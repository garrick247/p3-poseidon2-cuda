//! Batched GPU Merkle commit for BabyBear-16 Poseidon2, with mixed-matrix
//! injection support — byte-identical to Plonky3's binary-arity default
//! `MerkleTreeMmcs<_, _, PaddingFreeSponge<16, 8, 8>, TruncatedPermutation<2, 8, 16>, 2, 8>`.
//!
//! Builds the standard FRI-style commit:
//!   - Tallest matrices' rows form the leaf layer (concatenated per row).
//!   - Each step up halves the layer via binary `compress(left, right)`.
//!   - At any layer whose length matches a shorter matrix's height, that
//!     matrix's rows are hashed, then `compress(intermediate, row_digest)`
//!     replaces each node's digest.
//!   - Continue until 1 element remains (the root).
//!
//! Constraints (MVP):
//!   - All matrix heights must be powers of two.
//!   - All heights must be distinct after collapsing same-height groups
//!     (matrices of equal height are concatenated horizontally, matching
//!     Plonky3's `tallest_matrices` collection and shorter-matrix injection).
//!   - Binary arity (N=2) — Plonky3's standard config; N-ary support not
//!     in this MVP.

use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField32;
use p3_field::integers::QuotientMap;

use crate::Poseidon2GpuSession;

const WIDTH: usize = 16;
const RATE: usize = 8;
const DIGEST: usize = 8;

/// A height × width matrix of BabyBear elements, row-major.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: Vec<Vec<BabyBear>>,
}

impl Matrix {
    pub fn height(&self) -> usize { self.rows.len() }
    pub fn width(&self) -> usize { self.rows.first().map(|r| r.len()).unwrap_or(0) }
}

/// Hash N row-vectors via PaddingFreeSponge<16, 8, 8>, batched on GPU.
/// Each input is an arbitrary-length BabyBear sequence; output is one
/// 8-element digest per input. Inputs may have DIFFERENT lengths — the
/// number of absorb steps is max(ceil(len / RATE)) and rows shorter than
/// the max get a no-op tail (their state stays at the last permute output).
fn batched_sponge_hash(
    session: &mut Poseidon2GpuSession,
    inputs: &[Vec<BabyBear>],
) -> Vec<[BabyBear; DIGEST]> {
    let n = inputs.len();
    if n == 0 { return Vec::new(); }

    let max_len = inputs.iter().map(|r| r.len()).max().unwrap_or(0);
    if max_len == 0 {
        // All empty: PaddingFreeSponge with no input never permutes; state stays zero.
        return vec![[BabyBear::ZERO; DIGEST]; n];
    }

    let n_steps = (max_len + RATE - 1) / RATE;
    let mut state_flat: Vec<u32> = vec![0u32; n * WIDTH];

    for step in 0..n_steps {
        let chunk_start = step * RATE;
        let mut anyone_absorbed = false;

        for (row_idx, row) in inputs.iter().enumerate() {
            if chunk_start >= row.len() {
                // This row already fully absorbed; its state stays put.
                continue;
            }
            anyone_absorbed = true;
            let chunk_end = (chunk_start + RATE).min(row.len());
            let actual_rate = chunk_end - chunk_start;
            let state_offset = row_idx * WIDTH;
            // Overwrite first `actual_rate` rate slots; preserve the rest.
            for i in 0..actual_rate {
                state_flat[state_offset + i] = row[chunk_start + i].as_canonical_u32();
            }
        }

        if !anyone_absorbed { break; }

        // Dispatch one big batch of N permutations.
        // Note: rows that didn't absorb get permuted too. This is wasteful
        // GPU work for those rows but correctness requires NOT permuting
        // their state. We'd need to either skip them or accept the extra
        // perm and then ROLL BACK their state. Easiest: track which rows
        // are "done" and after the batched dispatch, RESTORE their
        // pre-perm state from a saved copy.
        let mut saved_inactive: Vec<(usize, [u32; WIDTH])> = Vec::new();
        for (row_idx, row) in inputs.iter().enumerate() {
            if chunk_start >= row.len() {
                let state_offset = row_idx * WIDTH;
                let mut snap = [0u32; WIDTH];
                snap.copy_from_slice(&state_flat[state_offset..state_offset + WIDTH]);
                saved_inactive.push((row_idx, snap));
            }
        }

        let mut out_flat = vec![0u32; n * WIDTH];
        let rc = session.permute_batch_canonical(&state_flat, &mut out_flat);
        assert_eq!(rc, 0, "session permute failed: {rc}");
        state_flat = out_flat;

        // Restore states for rows that shouldn't have been permuted this step.
        for (row_idx, snap) in saved_inactive {
            let state_offset = row_idx * WIDTH;
            state_flat[state_offset..state_offset + WIDTH].copy_from_slice(&snap);
        }
    }

    // Squeeze: digest[row] = state[row][0..DIGEST].
    (0..n)
        .map(|row_idx| {
            let off = row_idx * WIDTH;
            let mut d = [BabyBear::ZERO; DIGEST];
            for i in 0..DIGEST {
                d[i] = BabyBear::from_int(state_flat[off + i]);
            }
            d
        })
        .collect()
}

/// Binary compress: pair up consecutive digests in `layer`, apply
/// TruncatedPermutation<2, 8, 16> to each pair, return the new layer.
fn batched_binary_compress(
    session: &mut Poseidon2GpuSession,
    layer: &[[BabyBear; DIGEST]],
) -> Vec<[BabyBear; DIGEST]> {
    let half = layer.len() / 2;
    if half == 0 { return Vec::new(); }

    let mut state_flat: Vec<u32> = vec![0u32; half * WIDTH];
    for i in 0..half {
        let off = i * WIDTH;
        for j in 0..DIGEST {
            state_flat[off + j] = layer[2 * i][j].as_canonical_u32();
            state_flat[off + DIGEST + j] = layer[2 * i + 1][j].as_canonical_u32();
        }
    }

    let mut out_flat = vec![0u32; half * WIDTH];
    let rc = session.permute_batch_canonical(&state_flat, &mut out_flat);
    assert_eq!(rc, 0);

    (0..half)
        .map(|i| {
            let off = i * WIDTH;
            let mut d = [BabyBear::ZERO; DIGEST];
            for j in 0..DIGEST {
                d[j] = BabyBear::from_int(out_flat[off + j]);
            }
            d
        })
        .collect()
}

/// Pairwise compress two equal-length digest layers: out[i] = compress(a[i], b[i]).
/// Used for the injection step: `compress(intermediate_from_prev_layer, row_digest)`.
fn batched_pairwise_compress(
    session: &mut Poseidon2GpuSession,
    a: &[[BabyBear; DIGEST]],
    b: &[[BabyBear; DIGEST]],
) -> Vec<[BabyBear; DIGEST]> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 { return Vec::new(); }

    let mut state_flat: Vec<u32> = vec![0u32; n * WIDTH];
    for i in 0..n {
        let off = i * WIDTH;
        for j in 0..DIGEST {
            state_flat[off + j] = a[i][j].as_canonical_u32();
            state_flat[off + DIGEST + j] = b[i][j].as_canonical_u32();
        }
    }

    let mut out_flat = vec![0u32; n * WIDTH];
    let rc = session.permute_batch_canonical(&state_flat, &mut out_flat);
    assert_eq!(rc, 0);

    (0..n)
        .map(|i| {
            let off = i * WIDTH;
            let mut d = [BabyBear::ZERO; DIGEST];
            for j in 0..DIGEST {
                d[j] = BabyBear::from_int(out_flat[off + j]);
            }
            d
        })
        .collect()
}

/// Backward-compatible single-matrix entry. Calls into the full Mmcs-style
/// commit with a one-element matrix list. Kept for the existing tests/bench.
pub fn commit_root_via_gpu(
    session: &mut Poseidon2GpuSession,
    data: &[Vec<BabyBear>],
) -> [BabyBear; DIGEST] {
    let matrix = Matrix { rows: data.to_vec() };
    commit_root_via_gpu_mixed(session, vec![matrix])
}

/// Full Mmcs-style commit with mixed-matrix injection support.
///
/// Replicates the byte-for-byte hashing semantics of Plonky3's default
/// `MerkleTreeMmcs<_, _, PaddingFreeSponge<16, 8, 8>,
/// TruncatedPermutation<2, 8, 16>, 2, 8>::commit(matrices).root()`.
///
/// All matrix heights must be powers of two. Matrices of the same height
/// are concatenated horizontally for the leaf hash (matching Plonky3's
/// `tallest_matrices`). Shorter matrices are injected at the layer whose
/// length matches their height.
pub fn commit_root_via_gpu_mixed(
    session: &mut Poseidon2GpuSession,
    matrices: Vec<Matrix>,
) -> [BabyBear; DIGEST] {
    assert!(!matrices.is_empty(), "no matrices given");
    for m in &matrices {
        assert!(m.height().is_power_of_two(), "matrix height must be power of two (got {})", m.height());
    }

    // Group by height, sorted descending.
    let mut by_height: std::collections::BTreeMap<usize, Vec<Matrix>> =
        std::collections::BTreeMap::new();
    for m in matrices {
        by_height.entry(m.height()).or_default().push(m);
    }
    let groups: Vec<(usize, Vec<Matrix>)> = by_height.into_iter().rev().collect();

    let max_height = groups[0].0;

    // Layer 0: hash the tallest group's rows (concatenated horizontally per row).
    let tallest_inputs: Vec<Vec<BabyBear>> = (0..max_height)
        .map(|row_idx| {
            groups[0].1.iter().flat_map(|m| m.rows[row_idx].iter().copied()).collect()
        })
        .collect();
    let mut layer = batched_sponge_hash(session, &tallest_inputs);

    // Walk up, halving each step. At each new layer length, check whether
    // any shorter group lands there; if so, inject.
    let mut group_idx = 1; // index into `groups` for the next shorter matrices
    loop {
        if layer.len() <= 1 { break; }

        // Binary compress: layer halves.
        let intermediate = batched_binary_compress(session, &layer);
        let new_len = intermediate.len();

        // Does any pending group land at this layer length?
        if group_idx < groups.len() && groups[group_idx].0 == new_len {
            let inject_matrices = &groups[group_idx].1;
            let inject_inputs: Vec<Vec<BabyBear>> = (0..new_len)
                .map(|row_idx| {
                    inject_matrices.iter().flat_map(|m| m.rows[row_idx].iter().copied()).collect()
                })
                .collect();
            let row_digests = batched_sponge_hash(session, &inject_inputs);
            layer = batched_pairwise_compress(session, &intermediate, &row_digests);
            group_idx += 1;
        } else {
            layer = intermediate;
        }
    }

    // Sanity: all groups must have been consumed.
    // (If a group's height never matches any halving step, the tree
    // shape is incompatible — Plonky3's MerkleTreeMmcs::commit asserts
    // this with the "matrix height compatible with tallest height" check.)
    assert_eq!(
        group_idx, groups.len(),
        "matrix heights {:?} are incompatible with the tallest height {} \
         (not all injected during the binary tree walk)",
        groups.iter().skip(group_idx).map(|(h, _)| h).collect::<Vec<_>>(),
        max_height,
    );

    layer[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::default_babybear_poseidon2_16;
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::{
        PaddingFreeSponge, TruncatedPermutation, CryptographicHasher, PseudoCompressionFunction,
    };

    fn make_matrix(n_rows: usize, row_width: usize, seed: u64) -> Matrix {
        let mut x: u64 = seed;
        let rows = (0..n_rows)
            .map(|_| {
                (0..row_width)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        BabyBear::from_int((x as u32) % 2_013_265_921)
                    })
                    .collect()
            })
            .collect();
        Matrix { rows }
    }

    /// Reference impl using Plonky3 primitives directly.
    /// Sorts by height desc; concatenates same-height groups; runs the
    /// same binary-compress + inject loop. Mirrors `commit_root_via_gpu_mixed`
    /// but uses CPU sponge/compress.
    fn reference_root_mixed(matrices: Vec<Matrix>) -> [BabyBear; 8] {
        let perm = default_babybear_poseidon2_16();
        let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(perm.clone());
        let compress = TruncatedPermutation::<_, 2, 8, 16>::new(perm);

        let mut by_height: std::collections::BTreeMap<usize, Vec<Matrix>> =
            std::collections::BTreeMap::new();
        for m in matrices {
            by_height.entry(m.height()).or_default().push(m);
        }
        let groups: Vec<(usize, Vec<Matrix>)> = by_height.into_iter().rev().collect();
        let max_height = groups[0].0;

        let mut layer: Vec<[BabyBear; 8]> = (0..max_height)
            .map(|row_idx| {
                let row: Vec<BabyBear> = groups[0]
                    .1
                    .iter()
                    .flat_map(|m| m.rows[row_idx].iter().copied())
                    .collect();
                sponge.hash_iter(row.into_iter())
            })
            .collect();

        let mut group_idx = 1;
        while layer.len() > 1 {
            let intermediate: Vec<[BabyBear; 8]> = layer
                .chunks_exact(2)
                .map(|pair| compress.compress([pair[0], pair[1]]))
                .collect();
            let new_len = intermediate.len();

            if group_idx < groups.len() && groups[group_idx].0 == new_len {
                let inject_matrices = &groups[group_idx].1;
                let row_digests: Vec<[BabyBear; 8]> = (0..new_len)
                    .map(|row_idx| {
                        let row: Vec<BabyBear> = inject_matrices
                            .iter()
                            .flat_map(|m| m.rows[row_idx].iter().copied())
                            .collect();
                        sponge.hash_iter(row.into_iter())
                    })
                    .collect();
                layer = intermediate
                    .iter()
                    .zip(row_digests.iter())
                    .map(|(a, b)| compress.compress([*a, *b]))
                    .collect();
                group_idx += 1;
            } else {
                layer = intermediate;
            }
        }
        layer[0]
    }

    #[test]
    fn gpu_mixed_single_matrix_w8_h64() {
        // Sanity: single-matrix path still works through commit_root_via_gpu_mixed.
        let m = make_matrix(64, 8, 0x111);
        let cpu = reference_root_mixed(vec![m.clone()]);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu_mixed(&mut session, vec![m]);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn gpu_mixed_two_matrices_same_height() {
        // Same-height matrices: concatenated horizontally at the leaf layer.
        let a = make_matrix(64, 8, 0x111);
        let b = make_matrix(64, 4, 0x222);
        let cpu = reference_root_mixed(vec![a.clone(), b.clone()]);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu_mixed(&mut session, vec![a, b]);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn gpu_mixed_two_matrices_diff_height_injection() {
        // Heights 64 and 32: 32-row matrix injects at the layer of length 32.
        let big = make_matrix(64, 8, 0x111);
        let small = make_matrix(32, 8, 0x222);
        let cpu = reference_root_mixed(vec![big.clone(), small.clone()]);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu_mixed(&mut session, vec![big, small]);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn gpu_mixed_three_matrices_chain_injection() {
        // Heights 256, 64, 16: two injection points (layer of len 64, layer of len 16).
        let a = make_matrix(256, 8, 0xA);
        let b = make_matrix(64, 8, 0xB);
        let c = make_matrix(16, 8, 0xC);
        let cpu = reference_root_mixed(vec![a.clone(), b.clone(), c.clone()]);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu_mixed(&mut session, vec![a, b, c]);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn gpu_mixed_diff_widths_injection() {
        // Heights 128, 32, with DIFFERENT widths — exercises sponge with variable input length.
        let big = make_matrix(128, 12, 0xD1);
        let small = make_matrix(32, 5, 0xD2);
        let cpu = reference_root_mixed(vec![big.clone(), small.clone()]);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu_mixed(&mut session, vec![big, small]);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn gpu_mixed_two_at_same_height_plus_injection() {
        // Two matrices at the tallest height (concatenated) plus one shorter for injection.
        let a = make_matrix(128, 4, 0xE1);
        let b = make_matrix(128, 6, 0xE2);
        let c = make_matrix(32, 8, 0xE3);
        let cpu = reference_root_mixed(vec![a.clone(), b.clone(), c.clone()]);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu_mixed(&mut session, vec![a, b, c]);
        assert_eq!(gpu, cpu);
    }

    /// Compute the root via Plonky3's actual MerkleTreeMmcs::commit.
    /// Returns the root (cap_height=0 gives a 1-element cap).
    fn plonky3_mmcs_root(matrices: Vec<Matrix>) -> [BabyBear; 8] {
        use p3_baby_bear::Poseidon2BabyBear;
        use p3_commit::Mmcs;
        use p3_matrix::dense::RowMajorMatrix;
        use p3_merkle_tree::MerkleTreeMmcs;

        let perm: Poseidon2BabyBear<16> = default_babybear_poseidon2_16();
        let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(perm.clone());
        let compress = TruncatedPermutation::<_, 2, 8, 16>::new(perm);

        type Mmcs_ = MerkleTreeMmcs<
            <BabyBear as p3_field::Field>::Packing,
            <BabyBear as p3_field::Field>::Packing,
            PaddingFreeSponge<Poseidon2BabyBear<16>, 16, 8, 8>,
            TruncatedPermutation<Poseidon2BabyBear<16>, 2, 8, 16>,
            2,
            8,
        >;
        let mmcs = Mmcs_::new(sponge, compress, 0);

        let p3_matrices: Vec<RowMajorMatrix<BabyBear>> = matrices
            .into_iter()
            .map(|m| {
                let height = m.height();
                let width = m.width();
                let flat: Vec<BabyBear> = m.rows.into_iter().flatten().collect();
                debug_assert_eq!(flat.len(), height * width);
                RowMajorMatrix::new(flat, width)
            })
            .collect();

        let (cap, _prover_data) = mmcs.commit(p3_matrices);
        // cap_height=0 means cap has 1 element = the root.
        cap[0]
    }

    #[test]
    fn gpu_root_matches_plonky3_actual_mmcs_single() {
        // Validate against Plonky3's MerkleTreeMmcs::commit (not just our ref).
        let m = make_matrix(64, 8, 0x1001);
        let p3 = plonky3_mmcs_root(vec![m.clone()]);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu_mixed(&mut session, vec![m]);
        assert_eq!(gpu, p3, "gpu_root != Plonky3 MerkleTreeMmcs root (single matrix)");
    }

    #[test]
    fn gpu_root_matches_plonky3_actual_mmcs_two_diff_height() {
        let big = make_matrix(64, 8, 0x2001);
        let small = make_matrix(32, 8, 0x2002);
        let p3 = plonky3_mmcs_root(vec![big.clone(), small.clone()]);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu_mixed(&mut session, vec![big, small]);
        assert_eq!(gpu, p3, "gpu_root != Plonky3 MerkleTreeMmcs root (mixed)");
    }

    #[test]
    fn gpu_root_matches_plonky3_actual_mmcs_three_chain() {
        let a = make_matrix(256, 8, 0x3001);
        let b = make_matrix(64, 8, 0x3002);
        let c = make_matrix(16, 8, 0x3003);
        let p3 = plonky3_mmcs_root(vec![a.clone(), b.clone(), c.clone()]);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu_mixed(&mut session, vec![a, b, c]);
        assert_eq!(gpu, p3, "gpu_root != Plonky3 MerkleTreeMmcs root (chain)");
    }

}
