//! A/B throughput bench for the session API. Three modes per (n, K):
//!
//!   single-call   : current API (permute_batch). Allocates per call, pageable PCIe,
//!                   Mont<->canonical conversion at both ends.
//!   session-mont  : Poseidon2GpuSession::permute_batch. Persistent buffers,
//!                   pinned PCIe, BUT still pays Mont<->canonical conversion.
//!   session-canon : Poseidon2GpuSession::permute_batch_canonical. Skips
//!                   conversion entirely; consumer hands us canonical u32s.
//!                   This is the "future path" where the prover keeps state
//!                   in canonical form between hash calls.

use std::time::Instant;
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField32;
use p3_field::integers::QuotientMap;
use p3_poseidon2_cuda::{permute_batch, Poseidon2GpuSession};

fn median(mut xs: Vec<std::time::Duration>) -> std::time::Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn main() {
    println!("Poseidon2 BB-16 session API: three modes");
    println!("(5-run medians; K = #batches per timing window)");
    println!();
    println!("{:>7} | {:>3} | {:>14} | {:>14} | {:>14} | {:>10} | {:>10}",
        "n", "K", "single-call ms", "session-mont ms", "session-canon ms",
        "mont vs s", "canon vs s");
    println!("{}", "-".repeat(100));

    for &n in &[1usize << 16, 1 << 18, 1 << 20, 1 << 22] {
        let mut input = vec![BabyBear::ZERO; 16 * n];
        let mut x: u64 = 0xC0DE_F00D;
        for slot in input.iter_mut() {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            *slot = BabyBear::from_int((x as u32) % 2_013_265_921);
        }
        // Pre-compute canonical form once (the session-canon path's input).
        let canon_in: Vec<u32> = input.iter().map(|x| x.as_canonical_u32()).collect();
        let mut canon_out: Vec<u32> = vec![0; canon_in.len()];
        let mut out_mont = vec![BabyBear::ZERO; 16 * n];

        let mut session = Poseidon2GpuSession::new();

        for &k in &[1usize, 10] {
            // Warm all three paths.
            permute_batch(&input, &mut out_mont);
            session.permute_batch(&input, &mut out_mont);
            session.permute_batch_canonical(&canon_in, &mut canon_out);

            // Single-call
            let mut ts = Vec::with_capacity(5);
            for _ in 0..5 {
                let t0 = Instant::now();
                for _ in 0..k { permute_batch(&input, &mut out_mont); }
                ts.push(t0.elapsed());
            }
            let single = median(ts);

            // Session-mont (still does the conversion)
            let mut ts = Vec::with_capacity(5);
            for _ in 0..5 {
                let t0 = Instant::now();
                for _ in 0..k { session.permute_batch(&input, &mut out_mont); }
                ts.push(t0.elapsed());
            }
            let sm = median(ts);

            // Session-canonical (skips conversion entirely)
            let mut ts = Vec::with_capacity(5);
            for _ in 0..5 {
                let t0 = Instant::now();
                for _ in 0..k {
                    let rc = session.permute_batch_canonical(&canon_in, &mut canon_out);
                    assert_eq!(rc, 0);
                }
                ts.push(t0.elapsed());
            }
            let sc = median(ts);

            let mont_vs_single = single.as_secs_f64() / sm.as_secs_f64();
            let canon_vs_single = single.as_secs_f64() / sc.as_secs_f64();

            println!(
                " {:>6} | {:>3} | {:>14.2} | {:>14.2} | {:>14.2} | {:>9.2}x | {:>9.2}x",
                n, k,
                single.as_secs_f64() * 1000.0,
                sm.as_secs_f64() * 1000.0,
                sc.as_secs_f64() * 1000.0,
                mont_vs_single, canon_vs_single,
            );
        }
    }

    println!();
    println!("Interpretation:");
    println!("  - mont vs s: gain from session machinery alone (persistent buffers +");
    println!("    pinned PCIe). Modest (~0.9x to ~1.04x) because conversion dominates.");
    println!("  - canon vs s: gain when consumer keeps state in canonical form between");
    println!("    calls. This is the real unlock - it eliminates the ~50-70% conversion");
    println!("    overhead the breakdown bench measured.");
}
