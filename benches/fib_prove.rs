//! End-to-end Plonky3 Fibonacci STARK prove + verify, GPU vs stock CPU.
//!
//! Same Fibonacci AIR + StarkConfig wiring as Plonky3's
//! batch-stark/benches/prove_batch.rs, but with TWO configurations:
//!   - Stock: `MerkleTreeMmcs<..., PaddingFreeSponge<16,8,8>, TruncatedPermutation<2,8,16>, 2, 8>`
//!   - GPU:   `GpuPoseidon2Mmcs` (this crate)
//!
//! Both use the EXACT same per-AIR config (Poseidon2-BB-16 throughout —
//! non-standard for the canonical examples/prove_prime_field_31.rs config
//! that uses BB-24 for sponge, but valid for unit-stark; matches the
//! batch-stark bench's config exactly).

use core::borrow::Borrow;
use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeField64};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::StdRng;

use p3_poseidon2_cuda::GpuPoseidon2Mmcs;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

// Two value-Mmcs configurations.
type StockValMmcs = MerkleTreeMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    MyHash,
    MyCompress,
    2,
    8,
>;
type GpuValMmcs = GpuPoseidon2Mmcs;

type StockChallengeMmcs = ExtensionMmcs<Val, Challenge, StockValMmcs>;
type GpuChallengeMmcs   = ExtensionMmcs<Val, Challenge, GpuValMmcs>;

type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;

type StockPcs = TwoAdicFriPcs<Val, Dft, StockValMmcs, StockChallengeMmcs>;
type GpuPcs   = TwoAdicFriPcs<Val, Dft, GpuValMmcs,   GpuChallengeMmcs>;

type StockConfig = StarkConfig<StockPcs, Challenge, Challenger>;
type GpuConfig   = StarkConfig<GpuPcs,   Challenge, Challenger>;

#[derive(Debug, Clone, Copy)]
struct FibonacciAir { log_height: usize }

impl<F: Field> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize { 2 }
    fn num_public_values(&self) -> usize { 3 }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct FibRow<F> { left: F, right: F }
impl<F> Borrow<FibRow<F>> for [F] {
    fn borrow(&self) -> &FibRow<F> {
        debug_assert_eq!(self.len(), 2);
        unsafe { &*(self.as_ptr() as *const FibRow<F>) }
    }
}

impl<AB: AirBuilder> Air<AB> for FibonacciAir where AB::F: Field {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (a0, b0, x) = (pis[0], pis[1], pis[2]);
        let local:  &FibRow<AB::Var> = main.current_slice().borrow();
        let next:   &FibRow<AB::Var> = main.next_slice().borrow();
        builder.when_first_row().assert_eq(local.left, a0);
        builder.when_first_row().assert_eq(local.right, b0);
        builder.when_transition().assert_eq(local.right, next.left);
        builder.when_transition().assert_eq(local.left + local.right, next.right);
        builder.when_last_row().assert_eq(local.right, x);
    }
}

fn fib_trace<F: PrimeField64>(n: usize) -> (RowMajorMatrix<F>, Vec<F>) {
    let mut vals = Vec::with_capacity(n * 2);
    let (mut a, mut b) = (F::ZERO, F::ONE);
    for _ in 0..n {
        vals.push(a);
        vals.push(b);
        let c = a + b;
        a = b;
        b = c;
    }
    let last = vals[2 * (n - 1) + 1];
    (RowMajorMatrix::new(vals, 2), vec![F::ZERO, F::ONE, last])
}

fn make_stock_config() -> StockConfig {
    let mut rng = StdRng::seed_from_u64(1337);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = StockValMmcs::new(hash, compress, 0);
    let challenge_mmcs = StockChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = StockPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

fn make_gpu_config() -> GpuConfig {
    let mut rng = StdRng::seed_from_u64(1337);
    let perm = Perm::new_from_rng_128(&mut rng);
    let val_mmcs = GpuValMmcs::new(0);
    let challenge_mmcs = GpuChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = GpuPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

fn median(mut xs: Vec<std::time::Duration>) -> std::time::Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn main() {
    println!("Fibonacci AIR STARK prove + verify, 3-run medians");
    println!("Config: Poseidon2-BB-16 throughout (matches batch-stark/benches/prove_batch.rs).");
    println!();
    println!("{:>10} | {:>14} | {:>14} | {:>10}",
        "log_height", "stock CPU ms", "GPU Mmcs ms", "speedup");
    println!("{}", "-".repeat(60));

    let stock_cfg = make_stock_config();
    let gpu_cfg = make_gpu_config();

    for &log_height in &[10usize, 12, 14, 16] {
        let n_rows = 1usize << log_height;
        let air = FibonacciAir { log_height };
        let (trace, pis) = fib_trace::<Val>(n_rows);

        // Validate both produce valid proofs.
        let proof_stock = prove(&stock_cfg, &air, trace.clone(), &pis);
        verify(&stock_cfg, &air, &proof_stock, &pis).expect("stock proof verifies");

        let proof_gpu = prove(&gpu_cfg, &air, trace.clone(), &pis);
        verify(&gpu_cfg, &air, &proof_gpu, &pis).expect("gpu proof verifies");

        // Time stock.
        let mut ts = Vec::with_capacity(3);
        for _ in 0..3 {
            let t0 = Instant::now();
            let p = prove(&stock_cfg, &air, trace.clone(), &pis);
            std::hint::black_box(&p);
            ts.push(t0.elapsed());
        }
        let stock = median(ts);

        // Time GPU.
        let mut ts = Vec::with_capacity(3);
        for _ in 0..3 {
            let t0 = Instant::now();
            let p = prove(&gpu_cfg, &air, trace.clone(), &pis);
            std::hint::black_box(&p);
            ts.push(t0.elapsed());
        }
        let gpu = median(ts);

        let speedup = stock.as_secs_f64() / gpu.as_secs_f64();
        println!(
            " {:>10} | {:>14.2} | {:>14.2} | {:>9.2}x",
            log_height,
            stock.as_secs_f64() * 1000.0,
            gpu.as_secs_f64() * 1000.0,
            speedup,
        );
    }

    println!();
    println!("Both configurations produce valid proofs (asserted by verify() before timing).");
    println!("StockValMmcs uses Plonky3's reference MerkleTreeMmcs (multi-threaded via rayon).");
    println!("GpuValMmcs uses Poseidon2GpuSession via the new gpu_mmcs::GpuPoseidon2Mmcs trait impl.");
}
