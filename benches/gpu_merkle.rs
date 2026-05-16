//! End-to-end Merkle commit throughput: GPU-batched (`commit_root_via_gpu`)
//! vs Plonky3's reference (PaddingFreeSponge + TruncatedPermutation in a loop).
//!
//! Computes byte-identical Merkle roots over a synthetic matrix; reports
//! commit-time for each.

use std::time::Instant;
use p3_baby_bear::{BabyBear, default_babybear_poseidon2_16};
use p3_field::PrimeCharacteristicRing;
use p3_field::integers::QuotientMap;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation, CryptographicHasher, PseudoCompressionFunction};
use p3_poseidon2_cuda::{Poseidon2GpuSession, gpu_merkle::commit_root_via_gpu};

fn reference_root(data: &[Vec<BabyBear>]) -> [BabyBear; 8] {
    let perm = default_babybear_poseidon2_16();
    let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(perm.clone());
    let compress = TruncatedPermutation::<_, 2, 8, 16>::new(perm);

    let mut layer: Vec<[BabyBear; 8]> = data
        .iter()
        .map(|row| sponge.hash_iter(row.iter().copied()))
        .collect();

    while layer.len() > 1 {
        layer = layer
            .chunks_exact(2)
            .map(|pair| compress.compress([pair[0], pair[1]]))
            .collect();
    }
    layer[0]
}

fn median(mut xs: Vec<std::time::Duration>) -> std::time::Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn main() {
    let mut session = Poseidon2GpuSession::new();

    println!("Merkle commit (full tree, byte-identical roots verified)");
    println!("(5-run medians; Plonky3 CPU = PaddingFreeSponge+TruncatedPermutation in a loop)");
    println!();
    println!("{:>6} | {:>10} | {:>14} | {:>14} | {:>10}",
        "log_n", "row_width", "Plonky3 CPU ms", "GPU batched ms", "speedup");
    println!("{}", "-".repeat(75));

    for &log_n in &[10u32, 14, 16, 18, 20] {
        for &row_width in &[8usize, 16, 32] {
            let n_rows = 1usize << log_n;

            // Build matrix.
            let mut data: Vec<Vec<BabyBear>> = Vec::with_capacity(n_rows);
            let mut x: u64 = 0xC0DE_F00D ^ ((log_n as u64) << 8) ^ (row_width as u64);
            for _ in 0..n_rows {
                let mut row = Vec::with_capacity(row_width);
                for _ in 0..row_width {
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    row.push(BabyBear::from_int((x as u32) % 2_013_265_921));
                }
                data.push(row);
            }

            // Correctness.
            let cpu_root = reference_root(&data);
            let gpu_root = commit_root_via_gpu(&mut session, &data);
            assert_eq!(cpu_root, gpu_root,
                "ROOT MISMATCH at log_n={log_n} row_width={row_width}");

            // Warm.
            let _ = reference_root(&data);
            let _ = commit_root_via_gpu(&mut session, &data);

            // Time CPU.
            let mut ts = Vec::with_capacity(5);
            for _ in 0..5 {
                let t0 = Instant::now();
                let _ = reference_root(&data);
                ts.push(t0.elapsed());
            }
            let cpu = median(ts);

            // Time GPU.
            let mut ts = Vec::with_capacity(5);
            for _ in 0..5 {
                let t0 = Instant::now();
                let _ = commit_root_via_gpu(&mut session, &data);
                ts.push(t0.elapsed());
            }
            let gpu = median(ts);

            let speedup = cpu.as_secs_f64() / gpu.as_secs_f64();
            println!(
                " {:>6} | {:>10} | {:>14.2} | {:>14.2} | {:>9.2}x",
                log_n, row_width,
                cpu.as_secs_f64() * 1000.0,
                gpu.as_secs_f64() * 1000.0,
                speedup,
            );
        }
    }

    println!();
    println!("All sizes verified byte-identical between GPU-batched and Plonky3 reference.");
    println!("Speedups are end-to-end commit time (leaf hashing + inner-node compression).");
    println!("Includes per-call Mont<->canonical conversion at the BabyBear boundary.");
}
