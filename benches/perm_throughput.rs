//! Compare the FORGE-verified GPU BB-16 Poseidon2 perm against Plonky3's
//! single-thread CPU AVX2 reference, end-to-end (including HtoD/DtoH copies
//! and Montgomery <-> canonical conversion overhead).

use std::time::Instant;
use p3_baby_bear::{BabyBear, default_babybear_poseidon2_16};
use p3_field::integers::QuotientMap;
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::Permutation;
use p3_poseidon2_cuda::permute_batch;

fn main() {
    let perm = default_babybear_poseidon2_16();

    println!("n_perms |   GPU end-to-end   |  CPU 1-thread    | speedup");
    println!("--------+--------------------+------------------+--------");

    for &n in &[1usize << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22] {
        let mut input = vec![BabyBear::ZERO; 16 * n];
        let mut x: u64 = 0xC0DE_F00D;
        for slot in input.iter_mut() {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            *slot = BabyBear::from_int((x as u32) % 2_013_265_921);
        }

        // Warm GPU once.
        let mut warm = vec![BabyBear::ZERO; 16 * n];
        permute_batch(&input, &mut warm);

        // Time GPU — median of 5 runs.
        let mut gpu_times = vec![];
        for _ in 0..5 {
            let mut out = vec![BabyBear::ZERO; 16 * n];
            let t0 = Instant::now();
            permute_batch(&input, &mut out);
            gpu_times.push(t0.elapsed());
        }
        gpu_times.sort();
        let gpu_t = gpu_times[2];

        // Time CPU — same input, same n.
        let mut cpu_times = vec![];
        for _ in 0..3 {
            let mut buf = input.clone();
            let t0 = Instant::now();
            for chunk in buf.chunks_exact_mut(16) {
                let arr: &mut [BabyBear; 16] = chunk.try_into().unwrap();
                perm.permute_mut(arr);
            }
            cpu_times.push(t0.elapsed());
        }
        cpu_times.sort();
        let cpu_t = cpu_times[1];

        let speedup = cpu_t.as_secs_f64() / gpu_t.as_secs_f64();
        println!(
            " {:>6} | {:>9.2} ms ({:>5.1} M/s) | {:>5.1} ms ({:>4.1} M/s) | {:>4.2}x",
            n,
            gpu_t.as_secs_f64() * 1000.0,
            n as f64 / gpu_t.as_secs_f64() / 1e6,
            cpu_t.as_secs_f64() * 1000.0,
            n as f64 / cpu_t.as_secs_f64() / 1e6,
            speedup,
        );
    }
}
