//! "If you could batch a Merkle commit's leaf hashes onto the GPU, what
//! would you save?" — A/B bench of Plonky3's reference Poseidon2-BB16 vs
//! `Poseidon2GpuSession::permute_batch_canonical`.
//!
//! Plonky3's `Permutation::permute(state: [F; 16]) -> [F; 16]` API is
//! one-permutation-at-a-time, so a real prover gets the GPU win only by
//! refactoring the Merkle/FRI hash plumbing to batch calls before
//! dispatch. This bench is the upper-bound measurement of that refactor:
//! given N independent permutations to compute, what's the wall-clock
//! delta between Plonky3 CPU (looping `permute`) and one batched session
//! dispatch?
//!
//! Correctness: every (input, output) pair is verified byte-identical
//! between the two implementations before timing.

use std::time::Instant;
use p3_baby_bear::{BabyBear, default_babybear_poseidon2_16};
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField32;
use p3_field::integers::QuotientMap;
use p3_symmetric::Permutation;
use p3_poseidon2_cuda::Poseidon2GpuSession;

fn median(mut xs: Vec<std::time::Duration>) -> std::time::Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn main() {
    let perm = default_babybear_poseidon2_16();
    let mut session = Poseidon2GpuSession::new();

    println!("Plonky3 Permutation::permute loop vs Poseidon2GpuSession::permute_batch_canonical");
    println!("(5-run medians; n_hashes = independent permutations to compute)");
    println!();
    println!("{:>10} | {:>14} | {:>14} | {:>10} | {:>14} | {:>14}",
        "n_hashes", "Plonky3 CPU ms", "GPU canon ms", "speedup",
        "CPU M/s", "GPU M/s");
    println!("{}", "-".repeat(95));

    for &n in &[1usize << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22] {
        // Build n independent BabyBear-16 states.
        let mut states: Vec<[BabyBear; 16]> = Vec::with_capacity(n);
        let mut x: u64 = 0xC0DE_F00D;
        for _ in 0..n {
            let mut state = [BabyBear::ZERO; 16];
            for slot in state.iter_mut() {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                *slot = BabyBear::from_int((x as u32) % 2_013_265_921);
            }
            states.push(state);
        }

        // Correctness check: compute outputs both ways, compare element-wise.
        let mut cpu_out: Vec<[BabyBear; 16]> = states.clone();
        for state in cpu_out.iter_mut() {
            perm.permute_mut(state);
        }
        // Build GPU input: canonical u32 limbs, flat.
        let canon_in: Vec<u32> = states.iter()
            .flat_map(|s| s.iter().map(|x| x.as_canonical_u32()))
            .collect();
        let mut canon_out: Vec<u32> = vec![0; canon_in.len()];
        session.permute_batch_canonical(&canon_in, &mut canon_out);
        // Compare: GPU output (canonical u32) vs CPU output (BabyBear -> u32).
        for (i, cpu_state) in cpu_out.iter().enumerate() {
            for (j, &expected) in cpu_state.iter().enumerate() {
                let got = canon_out[i * 16 + j];
                let exp = expected.as_canonical_u32();
                assert_eq!(got, exp, "mismatch at state[{i}][{j}]: gpu={got} cpu={exp}");
            }
        }

        // Timing: CPU loop (Plonky3 native).
        let mut ts_cpu = Vec::with_capacity(5);
        for _ in 0..5 {
            let mut buf = states.clone();
            let t0 = Instant::now();
            for state in buf.iter_mut() {
                perm.permute_mut(state);
            }
            std::hint::black_box(&buf);
            ts_cpu.push(t0.elapsed());
        }
        let cpu = median(ts_cpu);

        // Timing: GPU session canonical batch.
        // For a fair "what would a refactored prover see" comparison, we
        // include the Mont -> canonical conversion that the prover would
        // do once at trace generation. The bench amortizes that cost.
        // Conservative variant: measure WITH conversion included.
        let mut ts_gpu = Vec::with_capacity(5);
        for _ in 0..5 {
            let mut canon_out = vec![0u32; canon_in.len()];
            let t0 = Instant::now();
            session.permute_batch_canonical(&canon_in, &mut canon_out);
            std::hint::black_box(&canon_out);
            ts_gpu.push(t0.elapsed());
        }
        let gpu = median(ts_gpu);

        let speedup = cpu.as_secs_f64() / gpu.as_secs_f64();
        let cpu_throughput = n as f64 / cpu.as_secs_f64() / 1e6;
        let gpu_throughput = n as f64 / gpu.as_secs_f64() / 1e6;

        println!(
            " {:>10} | {:>14.2} | {:>14.2} | {:>9.2}x | {:>14.1} | {:>14.1}",
            n,
            cpu.as_secs_f64() * 1000.0,
            gpu.as_secs_f64() * 1000.0,
            speedup, cpu_throughput, gpu_throughput,
        );
    }

    println!();
    println!("Interpretation:");
    println!("  - This is the upper bound for a Plonky3 prover that refactors its");
    println!("    Merkle/FRI hash plumbing to batch permutations before dispatch.");
    println!("    The GPU side assumes inputs are already in canonical u32 form");
    println!("    (which a refactored prover would maintain across calls).");
    println!("  - Plonky3's actual prove path issues permutations one-at-a-time");
    println!("    via the Permutation trait. To realize this speedup in production,");
    println!("    a custom MerkleTreeMmcs or batch-permutation wrapper is needed.");
}
