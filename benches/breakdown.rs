//! Phase-by-phase breakdown of permute_batch.
//!
//! See M:\project_poseidon2_bb16.md for context: end-to-end is bottlenecked
//! at ~80% on host-side Mont<->canonical conversion (estimated). This bench
//! measures that ratio directly at sizes 2^16..2^22 by calling phase
//! helpers (mont_to_canonical / raw_gpu_call / canonical_to_mont) from the
//! library.

use std::time::Instant;
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::integers::QuotientMap;
use p3_poseidon2_cuda::{mont_to_canonical, raw_gpu_call, canonical_to_mont};

fn median(mut xs: Vec<std::time::Duration>) -> std::time::Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn main() {
    println!("Poseidon2 BB-16 phase breakdown (RTX 5090, 5-run medians)");
    println!();
    println!("{:>7} | {:>12} | {:>12} | {:>12} | {:>12} | {:>14} | {:>12}",
        "n", "mont2can ms", "gpu_call ms", "can2mont ms", "total ms",
        "kernel-only M/s", "conv % total");
    println!("{}", "-".repeat(105));

    for &n in &[1usize << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22] {
        // Build input in Montgomery form (Plonky3 native).
        let mut input = vec![BabyBear::ZERO; 16 * n];
        let mut x: u64 = 0xC0DE_F00D;
        for slot in input.iter_mut() {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            *slot = BabyBear::from_int((x as u32) % 2_013_265_921);
        }

        let mut canon_in: Vec<u32> = vec![0; 16 * n];
        let mut canon_out: Vec<u32> = vec![0; 16 * n];
        let mut out_mont: Vec<BabyBear> = vec![BabyBear::ZERO; 16 * n];

        // Warm.
        mont_to_canonical(&input, &mut canon_in);
        assert_eq!(raw_gpu_call(&canon_in, &mut canon_out), 0);

        // Phase 1: Mont -> canonical.
        let mut ts = Vec::with_capacity(5);
        for _ in 0..5 {
            let t0 = Instant::now();
            mont_to_canonical(&input, &mut canon_in);
            ts.push(t0.elapsed());
        }
        let m2c = median(ts);

        // Phase 2: GPU call (H->D + kernel + D->H + sync).
        let mut ts = Vec::with_capacity(5);
        for _ in 0..5 {
            let t0 = Instant::now();
            assert_eq!(raw_gpu_call(&canon_in, &mut canon_out), 0);
            ts.push(t0.elapsed());
        }
        let gpu = median(ts);

        // Phase 3: canonical -> Mont.
        let mut ts = Vec::with_capacity(5);
        for _ in 0..5 {
            let t0 = Instant::now();
            canonical_to_mont(&canon_out, &mut out_mont);
            ts.push(t0.elapsed());
        }
        let c2m = median(ts);

        let total = m2c + gpu + c2m;
        let conv = m2c + c2m;
        let conv_pct = conv.as_secs_f64() / total.as_secs_f64() * 100.0;
        let kernel_rate = n as f64 / gpu.as_secs_f64() / 1e6;

        println!(
            " {:>6} | {:>12.3} | {:>12.3} | {:>12.3} | {:>12.3} | {:>14.1} | {:>11.1}%",
            n,
            m2c.as_secs_f64() * 1000.0,
            gpu.as_secs_f64() * 1000.0,
            c2m.as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0,
            kernel_rate,
            conv_pct,
        );
    }

    println!();
    println!("conv = Mont<->canonical (host-side CPU work).");
    println!("gpu_call = H->D copy + kernel + D->H copy + sync (includes PCIe overhead).");
    println!("Kernel-only rate from `forge demos/2009` bench: ~1633 M/s.");
}
