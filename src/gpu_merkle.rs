//! Batched GPU Merkle commit for BabyBear-16 Poseidon2.
//!
//! Replicates the byte-for-byte hashing semantics of Plonky3's
//! `MerkleTreeMmcs<_, _, PaddingFreeSponge<Poseidon2-16, 16, 8, 8>,
//! TruncatedPermutation<Poseidon2-16, 2, 8, 16>, 2, 8>::commit` for a
//! single matrix of power-of-two height, but dispatches every layer's
//! permutations as one big GPU batch via `Poseidon2GpuSession`.
//!
//! The integration pattern this demonstrates is:
//!   1. Leaf hash layer: for each absorb step (ceil(W / RATE) steps for
//!      a matrix of width W), build a canonical-u32 batch covering all
//!      N rows' current state, dispatch via `permute_batch_canonical`.
//!   2. Inner-node compression: each level halves; for each level dispatch
//!      one batch of (prev_len/2) permutations, each applied to
//!      concat(left_digest, right_digest).
//!
//! Validated by comparing root vs Plonky3's default MerkleTreeMmcs.

use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField32;
use p3_field::integers::QuotientMap;

use crate::Poseidon2GpuSession;

const WIDTH: usize = 16;
const RATE: usize = 8;
const DIGEST: usize = 8;

/// Compute the Merkle root over `n_rows` rows of a width-`row_width` matrix
/// `data` stored row-major as `data[row][col]`. Returns the 8-element
/// digest at the root of a complete binary tree (n_rows must be a power
/// of two).
///
/// Hashing semantics match Plonky3's PaddingFreeSponge<16, 8, 8> + binary
/// TruncatedPermutation<2, 8, 16> on the same input.
pub fn commit_root_via_gpu(
    session: &mut Poseidon2GpuSession,
    data: &[Vec<BabyBear>],
) -> [BabyBear; DIGEST] {
    let n_rows = data.len();
    assert!(n_rows.is_power_of_two(), "n_rows must be a power of two");
    assert!(n_rows > 0, "empty matrix");
    let row_width = data[0].len();
    for row in data {
        assert_eq!(row.len(), row_width, "all rows must have the same width");
    }

    // ─── Layer 0: hash each row to a DIGEST-element digest ──────────
    // PaddingFreeSponge logic:
    //   state = [0; 16]
    //   for each RATE-sized chunk of the row:
    //     state[0..RATE] = chunk; permute(state)
    //   if final residual chunk had any elements: permute already happened.
    //   if exactly multiple of RATE: NO extra permute, state already permuted.
    //
    // We batch the absorb-then-permute step across all N rows in parallel:
    // each rate-block we overwrite the corresponding state's [0..RATE], then
    // dispatch one big GPU batch of N permutations.

    // Build initial state (zeros) for all rows.
    let mut state_flat: Vec<u32> = vec![0u32; n_rows * WIDTH];

    // Determine number of absorb-permute steps.
    // For W elements with RATE-sized blocks: the loop runs ceil(W/RATE)
    // times unless W == 0. Each full block ends with a permute. A partial
    // final block also ends with a permute (i != 0 case in PaddingFreeSponge).
    let n_steps = (row_width + RATE - 1) / RATE;

    // Each step: overwrite state[0..RATE] with the current rate-block's
    // worth of canonical u32 limbs, then permute all rows in one batch.
    for step in 0..n_steps {
        let chunk_start = step * RATE;
        let chunk_end = (chunk_start + RATE).min(row_width);
        let actual_rate = chunk_end - chunk_start;

        for row_idx in 0..n_rows {
            let state_offset = row_idx * WIDTH;
            // Overwrite the first `actual_rate` rate elements; leave
            // any not-yet-absorbed rate slots and the capacity untouched.
            for i in 0..actual_rate {
                state_flat[state_offset + i] = data[row_idx][chunk_start + i].as_canonical_u32();
            }
            // Note: if actual_rate < RATE, we do NOT zero out the slots
            // beyond actual_rate. Plonky3's PaddingFreeSponge preserves
            // the existing state for those slots — they hold whatever was
            // there from the previous permute output (after the first
            // block) or zero (on the first block).
            //
            // For the FIRST block: state is all-zero, so unwritten slots
            // remain zero. ✓ matches Plonky3.
            // For subsequent blocks: state[RATE..WIDTH] is capacity (we
            // don't overwrite that anyway). The rate slots that we WOULD
            // have written but the input is exhausted: Plonky3 break's
            // out of the loop after writing only actual_rate slots, then
            // permutes IF i != 0. So the slots from actual_rate..RATE
            // hold the PREVIOUS permute's output. Our code preserves that
            // (we only overwrite [0..actual_rate]). ✓
        }

        // Dispatch one big batch of N permutations.
        let mut out_flat = vec![0u32; n_rows * WIDTH];
        let rc = session.permute_batch_canonical(&state_flat, &mut out_flat);
        assert_eq!(rc, 0, "session permute failed: {rc}");
        state_flat = out_flat;
    }

    // Squeeze: digest[row] = state[row][0..DIGEST].
    let leaves: Vec<[BabyBear; DIGEST]> = (0..n_rows)
        .map(|row_idx| {
            let offset = row_idx * WIDTH;
            let mut d = [BabyBear::ZERO; DIGEST];
            for i in 0..DIGEST {
                d[i] = BabyBear::from_int(state_flat[offset + i]);
            }
            d
        })
        .collect();

    // ─── Inner layers: TruncatedPermutation<2, DIGEST, WIDTH> ────────
    //   pair = concat(left, right); permute(pair); digest = pair[0..DIGEST]
    //
    // Each layer halves the previous. Dispatch as one batch per layer.
    let mut layer = leaves;
    while layer.len() > 1 {
        let next_len = layer.len() / 2;

        // Build canonical batch: pair i is [left_i; right_i] packed into
        // WIDTH=16 u32s.
        let mut state_flat: Vec<u32> = vec![0u32; next_len * WIDTH];
        for i in 0..next_len {
            let off = i * WIDTH;
            for j in 0..DIGEST {
                state_flat[off + j] = layer[2 * i][j].as_canonical_u32();
                state_flat[off + DIGEST + j] = layer[2 * i + 1][j].as_canonical_u32();
            }
        }

        let mut out_flat = vec![0u32; next_len * WIDTH];
        let rc = session.permute_batch_canonical(&state_flat, &mut out_flat);
        assert_eq!(rc, 0, "session permute failed: {rc}");

        layer = (0..next_len)
            .map(|i| {
                let off = i * WIDTH;
                let mut d = [BabyBear::ZERO; DIGEST];
                for j in 0..DIGEST {
                    d[j] = BabyBear::from_int(out_flat[off + j]);
                }
                d
            })
            .collect();
    }

    layer[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::default_babybear_poseidon2_16;
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation, CryptographicHasher, PseudoCompressionFunction, Permutation};

    /// Reference Merkle root via Plonky3's hash + binary compression.
    /// Uses the same component constants (WIDTH=16, RATE=8, DIGEST=8) and
    /// the default Poseidon2-BB16 permutation.
    fn reference_root(data: &[Vec<BabyBear>]) -> [BabyBear; 8] {
        let perm = default_babybear_poseidon2_16();
        let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(perm.clone());
        let compress = TruncatedPermutation::<_, 2, 8, 16>::new(perm);

        // Layer 0: hash each row.
        let mut layer: Vec<[BabyBear; 8]> = data
            .iter()
            .map(|row| sponge.hash_iter(row.iter().copied()))
            .collect();

        // Inner layers.
        while layer.len() > 1 {
            layer = layer
                .chunks_exact(2)
                .map(|pair| compress.compress([pair[0], pair[1]]))
                .collect();
        }
        layer[0]
    }

    fn make_matrix(n_rows: usize, row_width: usize, seed: u64) -> Vec<Vec<BabyBear>> {
        let mut x: u64 = seed;
        (0..n_rows)
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
            .collect()
    }

    #[test]
    fn gpu_root_matches_plonky3_w8_h4() {
        // Simplest case: row_width = RATE = 8, height = 4. One absorb step per leaf.
        let data = make_matrix(4, 8, 0x1234_5678);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu(&mut session, &data);
        let cpu = reference_root(&data);
        assert_eq!(gpu, cpu, "GPU root != Plonky3 reference root (w=8, h=4)");
    }

    #[test]
    fn gpu_root_matches_plonky3_w8_h1024() {
        let data = make_matrix(1024, 8, 0xC0DE_F00D);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu(&mut session, &data);
        let cpu = reference_root(&data);
        assert_eq!(gpu, cpu, "GPU root != Plonky3 reference root (w=8, h=1024)");
    }

    #[test]
    fn gpu_root_matches_plonky3_w16_h256() {
        // Two absorb steps per leaf (row_width=16 > RATE=8).
        let data = make_matrix(256, 16, 0xDEADBEEF);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu(&mut session, &data);
        let cpu = reference_root(&data);
        assert_eq!(gpu, cpu, "GPU root != Plonky3 reference root (w=16, h=256)");
    }

    #[test]
    fn gpu_root_matches_plonky3_w5_h512() {
        // Partial-block case: row_width=5, less than RATE=8 — one permute,
        // unused rate slots stay zero.
        let data = make_matrix(512, 5, 0xFACE_F00D);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu(&mut session, &data);
        let cpu = reference_root(&data);
        assert_eq!(gpu, cpu, "GPU root != Plonky3 reference root (w=5, h=512)");
    }

    #[test]
    fn gpu_root_matches_plonky3_w17_h64() {
        // Three absorb steps (16=2 full + 1 partial of 1 element). Tests
        // the i!=0 partial-block-permute branch.
        let data = make_matrix(64, 17, 0x4242_4242);
        let mut session = Poseidon2GpuSession::new();
        let gpu = commit_root_via_gpu(&mut session, &data);
        let cpu = reference_root(&data);
        assert_eq!(gpu, cpu, "GPU root != Plonky3 reference root (w=17, h=64)");
    }
}
