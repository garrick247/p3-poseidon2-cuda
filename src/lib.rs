//! FORGE-verified Plonky3 BabyBear-16 Poseidon2 permutation, on GPU.
//!
//! Wraps the FORGE/Z3-verified factored kernel
//! (`baby_bear_fused_perm_factored_kernel` from
//! `forge/demos/2009_bench_baby_bear_fused_perm_factored.cu`) behind a
//! batch FFI. Output is byte-identical to Plonky3's
//! `default_babybear_poseidon2_16` on the same inputs.
//!
//! NOTE: Plonky3 stores BabyBear in Montgomery form internally; the GPU
//! kernel works on canonical-form u32 limbs. We convert at the boundary.

use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use p3_field::integers::QuotientMap;


unsafe extern "C" {
    fn cuda_poseidon2_bb16_perm_batch(
        state: *const u32,
        out:   *mut u32,
        n:     u64,
    ) -> i32;
}

/// Permute `n` BabyBear-16 states in one CUDA dispatch.
///
/// `state` and `out` are flat slices of length `16 * n`. State `i` lives at
/// indices `[16*i .. 16*(i+1))`. Each output state is byte-identical to
/// `default_babybear_poseidon2_16().permute(state)` on the matching input.
pub fn permute_batch(state: &[BabyBear], out: &mut [BabyBear]) {
    assert_eq!(state.len(), out.len(), "state and out must match");
    assert!(state.len() % 16 == 0, "len must be multiple of 16");
    let n = (state.len() / 16) as u64;
    if n == 0 { return; }

    // Convert Montgomery -> canonical u32 limbs.
    let canon_in: Vec<u32> = state.iter().map(|x| x.as_canonical_u32()).collect();
    let mut canon_out: Vec<u32> = vec![0; canon_in.len()];

    let rc = unsafe {
        cuda_poseidon2_bb16_perm_batch(canon_in.as_ptr(), canon_out.as_mut_ptr(), n)
    };
    assert_eq!(rc, 0, "cuda_poseidon2_bb16_perm_batch failed: rc={rc}");

    // Convert canonical u32 -> Montgomery BabyBear.
    for (slot, &x) in out.iter_mut().zip(canon_out.iter()) {
        *slot = BabyBear::from_int(x);
    }
}

/// Phase 1: convert Montgomery-form BabyBear states to canonical u32 limbs.
/// Exposed so benches can measure the conversion cost in isolation.
pub fn mont_to_canonical(state: &[BabyBear], canon: &mut [u32]) {
    assert_eq!(state.len(), canon.len(), "state and canon must match length");
    for (c, m) in canon.iter_mut().zip(state.iter()) {
        *c = m.as_canonical_u32();
    }
}

/// Phase 2: raw GPU dispatch on canonical-form u32 limbs. Includes H->D copy,
/// kernel launch, D->H copy, and sync. Returns the kernel's exit code (0 = ok).
pub fn raw_gpu_call(canon_in: &[u32], canon_out: &mut [u32]) -> i32 {
    assert_eq!(canon_in.len(), canon_out.len(), "in and out must match length");
    assert!(canon_in.len() % 16 == 0, "len must be multiple of 16");
    let n = (canon_in.len() / 16) as u64;
    if n == 0 { return 0; }
    unsafe { cuda_poseidon2_bb16_perm_batch(canon_in.as_ptr(), canon_out.as_mut_ptr(), n) }
}

/// Phase 3: convert canonical u32 limbs back to Montgomery-form BabyBear.
pub fn canonical_to_mont(canon: &[u32], out: &mut [BabyBear]) {
    assert_eq!(canon.len(), out.len(), "canon and out must match length");
    for (slot, &x) in out.iter_mut().zip(canon.iter()) {
        *slot = BabyBear::from_int(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::default_babybear_poseidon2_16;
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;

    /// Plonky3's published BB-16 test vector (matches the same vector
    /// our forge bench harness validates against).
    const PLONKY3_BB16_INPUT_RAW: [u32; 16] = [
        894848333, 1437655012, 1200606629, 1690012884,  71131202, 1749206695,
       1717947831,  120589055,  19776022,   42382981, 1831865506,  724844064,
        171220207, 1299207443,  227047920, 1783754913,
    ];
    const PLONKY3_BB16_EXPECTED_RAW: [u32; 16] = [
        516096821,   90309867, 1101817252, 1660784290,  360715097, 1789519026,
       1788910906,  563338433,  319524748, 1741414159, 1650859320,  894311162,
       1121347488, 1692793758, 1052633829, 1344246938,
    ];

    fn to_bb_arr(raw: [u32; 16]) -> [BabyBear; 16] {
        let mut out = [BabyBear::ZERO; 16];
        for (i, x) in raw.iter().enumerate() { out[i] = BabyBear::from_int(*x); }
        out
    }

    #[test]
    fn single_perm_byte_identical_to_plonky3() {
        let input = to_bb_arr(PLONKY3_BB16_INPUT_RAW);
        let expected = to_bb_arr(PLONKY3_BB16_EXPECTED_RAW);
        let mut got = [BabyBear::ZERO; 16];
        permute_batch(&input, &mut got);
        assert_eq!(got, expected, "GPU output differs from Plonky3 KAT");

        let perm = default_babybear_poseidon2_16();
        let mut p3 = input;
        perm.permute_mut(&mut p3);
        assert_eq!(got, p3, "GPU output differs from Plonky3 runtime perm");
    }

    #[test]
    fn batch_byte_identical_to_plonky3() {
        let n = 1024;
        let perm = default_babybear_poseidon2_16();
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut input = vec![BabyBear::ZERO; 16 * n];
        for x in input.iter_mut() {
            *x = BabyBear::from_int(rng.gen_range(0..2_013_265_921u32));
        }

        let mut p3_out = input.clone();
        for chunk in p3_out.chunks_exact_mut(16) {
            let arr: &mut [BabyBear; 16] = chunk.try_into().unwrap();
            perm.permute_mut(arr);
        }

        let mut gpu_out = vec![BabyBear::ZERO; 16 * n];
        permute_batch(&input, &mut gpu_out);

        assert_eq!(gpu_out, p3_out, "batched GPU output differs from Plonky3");
    }
}

