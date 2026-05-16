//! Full `Mmcs<BabyBear>` trait impl backed by `Poseidon2GpuSession`.
//!
//! Drop-in replacement for Plonky3's
//! `MerkleTreeMmcs<P, P, PaddingFreeSponge<Poseidon2-16, 16, 8, 8>,
//!                 TruncatedPermutation<Poseidon2-16, 2, 8, 16>, 2, 8>`
//! for BabyBear with binary-arity, single-matrix or mixed-matrix-injection
//! commits.
//!
//! - `commit`: dispatches every layer's permutations as one GPU batch.
//! - `open_batch`: walks stored digest layers to collect sibling hashes.
//!   CPU-only (the work is pure pointer chasing).
//! - `verify_batch`: standard verifier — reconstructs root from openings.
//!   CPU-only.
//!
//! Byte-identical to Plonky3's `MerkleTreeMmcs` for the same config; tests
//! validate commit + open + verify cycle against the official API.

use core::cmp::Reverse;
use core::marker::PhantomData;
use std::sync::{Arc, Mutex};

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField32;
use p3_field::integers::QuotientMap;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{Hash, MerkleCap};
use p3_util::log2_ceil_usize;

use crate::Poseidon2GpuSession;

const WIDTH: usize = 16;
const RATE: usize = 8;
const DIGEST: usize = 8;

/// ProverData for `GpuPoseidon2Mmcs`. Stores the committed matrices and
/// every digest layer so `open_batch` can walk paths.
#[derive(Clone, Debug)]
pub struct GpuMerkleTree<M> {
    pub leaves: Vec<M>,
    pub digest_layers: Vec<Vec<[BabyBear; DIGEST]>>,
}

/// Errors returned by the Mmcs verifier.
#[derive(Debug, thiserror::Error)]
pub enum GpuMmcsError {
    #[error("wrong batch size")]
    WrongBatchSize,
    #[error("wrong width")]
    WrongWidth,
    #[error("wrong height (expected {expected_proof_len}, got {num_siblings})")]
    WrongHeight { expected_proof_len: usize, num_siblings: usize },
    #[error("root mismatch")]
    RootMismatch,
}

/// Mmcs<BabyBear> impl using the GPU session for hashing.
/// Session is held in an Arc<Mutex<_>> so the Mmcs is Clone (Mmcs trait requires Clone).
#[derive(Clone)]
pub struct GpuPoseidon2Mmcs {
    session: Arc<Mutex<Poseidon2GpuSession>>,
    cap_height: usize,
}

impl GpuPoseidon2Mmcs {
    pub fn new(cap_height: usize) -> Self {
        Self {
            session: Arc::new(Mutex::new(Poseidon2GpuSession::new())),
            cap_height,
        }
    }
}

// ----- Hashing primitives (CPU helpers used during open/verify) -----

fn cpu_sponge_hash(row: &[BabyBear]) -> [BabyBear; DIGEST] {
    use p3_baby_bear::default_babybear_poseidon2_16;
    use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
    let perm = default_babybear_poseidon2_16();
    let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm);
    sponge.hash_iter(row.iter().copied())
}

fn cpu_compress(a: [BabyBear; DIGEST], b: [BabyBear; DIGEST]) -> [BabyBear; DIGEST] {
    use p3_baby_bear::default_babybear_poseidon2_16;
    use p3_symmetric::{PseudoCompressionFunction, TruncatedPermutation};
    let perm = default_babybear_poseidon2_16();
    let compress = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(perm);
    compress.compress([a, b])
}

// ----- GPU batched hashing (used during commit) -----

fn batched_sponge_hash(
    session: &mut Poseidon2GpuSession,
    inputs: &[Vec<BabyBear>],
) -> Vec<[BabyBear; DIGEST]> {
    let n = inputs.len();
    if n == 0 { return Vec::new(); }
    let max_len = inputs.iter().map(|r| r.len()).max().unwrap_or(0);
    if max_len == 0 {
        return vec![[BabyBear::ZERO; DIGEST]; n];
    }
    let n_steps = (max_len + RATE - 1) / RATE;
    let mut state_flat: Vec<u32> = vec![0u32; n * WIDTH];

    for step in 0..n_steps {
        let chunk_start = step * RATE;
        let mut any = false;
        for (row_idx, row) in inputs.iter().enumerate() {
            if chunk_start >= row.len() { continue; }
            any = true;
            let chunk_end = (chunk_start + RATE).min(row.len());
            let actual_rate = chunk_end - chunk_start;
            let off = row_idx * WIDTH;
            for i in 0..actual_rate {
                state_flat[off + i] = row[chunk_start + i].as_canonical_u32();
            }
        }
        if !any { break; }

        let mut saved: Vec<(usize, [u32; WIDTH])> = Vec::new();
        for (row_idx, row) in inputs.iter().enumerate() {
            if chunk_start >= row.len() {
                let off = row_idx * WIDTH;
                let mut snap = [0u32; WIDTH];
                snap.copy_from_slice(&state_flat[off..off + WIDTH]);
                saved.push((row_idx, snap));
            }
        }

        let mut out_flat = vec![0u32; n * WIDTH];
        let rc = session.permute_batch_canonical(&state_flat, &mut out_flat);
        assert_eq!(rc, 0);
        state_flat = out_flat;

        for (row_idx, snap) in saved {
            let off = row_idx * WIDTH;
            state_flat[off..off + WIDTH].copy_from_slice(&snap);
        }
    }

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
            state_flat[off + j] = layer[2*i][j].as_canonical_u32();
            state_flat[off + DIGEST + j] = layer[2*i + 1][j].as_canonical_u32();
        }
    }
    let mut out_flat = vec![0u32; half * WIDTH];
    let rc = session.permute_batch_canonical(&state_flat, &mut out_flat);
    assert_eq!(rc, 0);
    (0..half).map(|i| {
        let off = i * WIDTH;
        let mut d = [BabyBear::ZERO; DIGEST];
        for j in 0..DIGEST {
            d[j] = BabyBear::from_int(out_flat[off + j]);
        }
        d
    }).collect()
}

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
    (0..n).map(|i| {
        let off = i * WIDTH;
        let mut d = [BabyBear::ZERO; DIGEST];
        for j in 0..DIGEST {
            d[j] = BabyBear::from_int(out_flat[off + j]);
        }
        d
    }).collect()
}

impl Mmcs<BabyBear> for GpuPoseidon2Mmcs {
    type ProverData<M> = GpuMerkleTree<M>;
    type Commitment = MerkleCap<BabyBear, [BabyBear; DIGEST]>;
    type Proof = Vec<[BabyBear; DIGEST]>;
    type Error = GpuMmcsError;

    fn commit<M: Matrix<BabyBear>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        assert!(!inputs.is_empty(), "no matrices given");
        // Group by height descending (matching Plonky3's tallest-first traversal).
        let mut indexed: Vec<(usize, &M)> = inputs.iter().enumerate().collect();
        indexed.sort_by_key(|(_, m)| Reverse(m.height()));

        let max_height = indexed[0].1.height();
        assert!(max_height.is_power_of_two(), "max_height must be a power of two");

        // Collect tallest group (all matrices of height == max_height).
        let mut iter = indexed.into_iter().peekable();
        let tallest: Vec<&M> = iter
            .by_ref()
            .peeking_take_while(|(_, m)| m.height() == max_height)
            .map(|(_, m)| m)
            .collect();

        let mut session_guard = self.session.lock().unwrap();
        let session = &mut *session_guard;

        // Layer 0: hash each row of tallest matrices, concatenated horizontally.
        let leaf_inputs: Vec<Vec<BabyBear>> = (0..max_height)
            .map(|row_idx| {
                tallest.iter()
                    .flat_map(|m| m.row(row_idx).unwrap().into_iter())
                    .collect()
            })
            .collect();
        let layer0 = batched_sponge_hash(session, &leaf_inputs);
        let mut digest_layers: Vec<Vec<[BabyBear; DIGEST]>> = vec![layer0];

        // Walk up: halve at each step. If a shorter group's height matches
        // the new layer length, inject it.
        loop {
            let prev = digest_layers.last().unwrap();
            if prev.len() <= 1 { break; }

            let intermediate = batched_binary_compress(session, prev);
            let new_len = intermediate.len();

            // Peek for shorter matrices that land at this length.
            let inject: Vec<&M> = iter
                .by_ref()
                .peeking_take_while(|(_, m)| m.height() == new_len)
                .map(|(_, m)| m)
                .collect();

            let next = if inject.is_empty() {
                intermediate
            } else {
                let inject_inputs: Vec<Vec<BabyBear>> = (0..new_len)
                    .map(|row_idx| {
                        inject.iter()
                            .flat_map(|m| m.row(row_idx).unwrap().into_iter())
                            .collect()
                    })
                    .collect();
                let row_digests = batched_sponge_hash(session, &inject_inputs);
                batched_pairwise_compress(session, &intermediate, &row_digests)
            };
            digest_layers.push(next);
        }

        // Sanity: all groups consumed.
        assert!(
            iter.next().is_none(),
            "matrix heights incompatible with tallest height — not all injected"
        );

        // Compute cap from the digest_layers.
        let num_layers = digest_layers.len();
        let effective_cap_height = self.cap_height.min(num_layers.saturating_sub(1));
        let cap_layer_idx = num_layers - 1 - effective_cap_height;
        let cap_digests: Vec<[BabyBear; DIGEST]> = digest_layers[cap_layer_idx].clone();
        let cap = MerkleCap::from(cap_digests);

        let tree = GpuMerkleTree { leaves: inputs, digest_layers };
        (cap, tree)
    }

    fn open_batch<M: Matrix<BabyBear>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> p3_commit::BatchOpening<BabyBear, Self> {
        let max_height = self.get_max_height(prover_data);
        assert!(index < max_height, "index out of bounds");
        let log_max_height = log2_ceil_usize(max_height);

        let openings: Vec<Vec<BabyBear>> = prover_data.leaves.iter().map(|matrix| {
            let log2_height = log2_ceil_usize(matrix.height());
            let bits_reduced = log_max_height - log2_height;
            let reduced_index = index >> bits_reduced;
            matrix.row(reduced_index).unwrap().into_iter().collect()
        }).collect();

        // Binary tree: sibling at each level is the other element of the pair.
        let num_layers = prover_data.digest_layers.len();
        let effective_cap_height = self.cap_height.min(num_layers.saturating_sub(1));
        let proof_levels = num_layers.saturating_sub(1).saturating_sub(effective_cap_height);

        let mut proof = Vec::with_capacity(proof_levels);
        let mut idx = index;
        for layer_idx in 0..proof_levels {
            let sibling = idx ^ 1;
            proof.push(prover_data.digest_layers[layer_idx][sibling]);
            idx /= 2;
        }

        p3_commit::BatchOpening::new(openings, proof)
    }

    fn get_matrices<'a, M: Matrix<BabyBear>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.leaves.iter().collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        mut index: usize,
        batch_proof: p3_commit::BatchOpeningRef<'_, BabyBear, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, opening_proof) = batch_proof.unpack();
        if dimensions.len() != opened_values.len() {
            return Err(GpuMmcsError::WrongBatchSize);
        }

        // Group dimensions by height descending.
        let mut by_height_desc: Vec<(usize, &Dimensions)> = dimensions.iter().enumerate().collect();
        by_height_desc.sort_by_key(|(_, d)| Reverse(d.height));

        let max_height = by_height_desc[0].1.height;
        let log_max_height = log2_ceil_usize(max_height);

        // Tallest group: concat their opened rows for the leaf hash.
        let mut iter = by_height_desc.into_iter().peekable();
        let tallest_indices: Vec<usize> = iter
            .by_ref()
            .peeking_take_while(|(_, d)| d.height == max_height)
            .map(|(i, _)| i)
            .collect();

        let leaf_row: Vec<BabyBear> = tallest_indices.iter()
            .flat_map(|&i| opened_values[i].iter().copied())
            .collect();
        let mut digest = cpu_sponge_hash(&leaf_row);

        // Walk up. proof[layer_idx] is the sibling at level layer_idx (idx ^ 1).
        let num_layers_total = opening_proof.len() + 1;  // including the cap layer at the end
        let mut idx = index;
        let mut new_len = max_height;
        for layer_idx in 0..opening_proof.len() {
            let sibling = opening_proof[layer_idx];
            digest = if idx & 1 == 0 {
                cpu_compress(digest, sibling)
            } else {
                cpu_compress(sibling, digest)
            };
            idx /= 2;
            new_len /= 2;
            // Inject shorter matrices that land at this new length.
            let inject_indices: Vec<usize> = iter
                .by_ref()
                .peeking_take_while(|(_, d)| d.height == new_len)
                .map(|(i, _)| i)
                .collect();
            if !inject_indices.is_empty() {
                let inject_row: Vec<BabyBear> = inject_indices.iter()
                    .flat_map(|&i| opened_values[i].iter().copied())
                    .collect();
                let row_digest = cpu_sponge_hash(&inject_row);
                digest = cpu_compress(digest, row_digest);
            }
        }

        // The digest is now the entry at cap_idx of the cap layer.
        let cap_idx = index >> opening_proof.len();
        let cap_digest: [BabyBear; DIGEST] = (*commit)[cap_idx].into();
        if digest != cap_digest {
            return Err(GpuMmcsError::RootMismatch);
        }
        let _ = num_layers_total;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::{Poseidon2BabyBear, default_babybear_poseidon2_16};
    use p3_field::Field;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

    type StockMmcs = MerkleTreeMmcs<
        <BabyBear as Field>::Packing,
        <BabyBear as Field>::Packing,
        PaddingFreeSponge<Poseidon2BabyBear<16>, 16, 8, 8>,
        TruncatedPermutation<Poseidon2BabyBear<16>, 2, 8, 16>,
        2,
        8,
    >;

    fn stock_mmcs(cap_height: usize) -> StockMmcs {
        let perm: Poseidon2BabyBear<16> = default_babybear_poseidon2_16();
        let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(perm.clone());
        let compress = TruncatedPermutation::<_, 2, 8, 16>::new(perm);
        StockMmcs::new(sponge, compress, cap_height)
    }

    fn make_matrix(n_rows: usize, width: usize, seed: u64) -> RowMajorMatrix<BabyBear> {
        let mut x: u64 = seed;
        let mut data = Vec::with_capacity(n_rows * width);
        for _ in 0..(n_rows * width) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            data.push(BabyBear::from_int((x as u32) % 2_013_265_921));
        }
        RowMajorMatrix::new(data, width)
    }

    #[test]
    fn commit_matches_stock_single_matrix() {
        let m = make_matrix(64, 8, 0xAAAA);
        let stock = stock_mmcs(0);
        let gpu = GpuPoseidon2Mmcs::new(0);

        let (cap_stock, _pd_stock) = stock.commit(vec![m.clone()]);
        let (cap_gpu, _pd_gpu) = gpu.commit(vec![m]);

        assert_eq!(cap_stock.as_ref(), cap_gpu.as_ref(),
                   "commit cap mismatch (single matrix)");
    }

    #[test]
    fn commit_matches_stock_mixed() {
        let a = make_matrix(256, 8, 0xB1);
        let b = make_matrix(64, 8, 0xB2);
        let c = make_matrix(16, 8, 0xB3);
        let stock = stock_mmcs(0);
        let gpu = GpuPoseidon2Mmcs::new(0);

        let (cap_stock, _) = stock.commit(vec![a.clone(), b.clone(), c.clone()]);
        let (cap_gpu, _) = gpu.commit(vec![a, b, c]);

        assert_eq!(cap_stock.as_ref(), cap_gpu.as_ref(),
                   "commit cap mismatch (mixed)");
    }

    #[test]
    fn open_verify_roundtrip_single() {
        let m = make_matrix(64, 8, 0xCAFE);
        let gpu = GpuPoseidon2Mmcs::new(0);
        let (commit, prover_data) = gpu.commit(vec![m]);

        for &index in &[0, 1, 31, 32, 63] {
            let opening = gpu.open_batch(index, &prover_data);
            let dims = vec![Dimensions { width: 8, height: 64 }];
            let opening_ref = p3_commit::BatchOpeningRef::new(
                &opening.opened_values,
                &opening.opening_proof,
            );
            gpu.verify_batch(&commit, &dims, index, opening_ref)
                .expect("verify failed");
        }
    }

    #[test]
    fn open_verify_roundtrip_mixed() {
        let a = make_matrix(64, 8, 0xDEAD);
        let b = make_matrix(16, 8, 0xBEEF);
        let gpu = GpuPoseidon2Mmcs::new(0);
        let (commit, prover_data) = gpu.commit(vec![a, b]);

        for &index in &[0, 1, 7, 8, 15, 32, 63] {
            let opening = gpu.open_batch(index, &prover_data);
            let dims = vec![
                Dimensions { width: 8, height: 64 },
                Dimensions { width: 8, height: 16 },
            ];
            let opening_ref = p3_commit::BatchOpeningRef::new(
                &opening.opened_values,
                &opening.opening_proof,
            );
            gpu.verify_batch(&commit, &dims, index, opening_ref)
                .expect("verify failed");
        }
    }
}
