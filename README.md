# p3-poseidon2-cuda

FORGE-verified CUDA implementation of the Plonky3 BabyBear-16 Poseidon2
permutation. Output is byte-identical to Plonky3's
`default_babybear_poseidon2_16` on every input.

## What

The kernel comes from the [forge](https://github.com/garrick247/forge) repo
at `demos/2009_bench_baby_bear_fused_perm_factored.{fg,cu}`. The `.fg` source
is verified by FORGE/Z3: 9692 proof obligations discharged. The factored form
(per-output-element `#[device]` helpers) keeps per-function SMT state
bounded.

This crate wraps the kernel in a batch FFI and a Montgomery <-> canonical
conversion at the boundary so it slots into Plonky3 code that uses
`p3_baby_bear::BabyBear`.

## Numbers (RTX 5090, single 285K CPU thread for reference)

End-to-end via the FFI (cudaMalloc + HtoD + Mont->canonical + kernel +
canonical->Mont + DtoH + cudaFree):

| n perms | GPU end-to-end | CPU 1-thread (Plonky3 AVX2) | speedup |
|---:|---:|---:|---:|
| 1024  |   0.11 ms ( 9.4 M/s) |   0.5 ms (1.9 M/s) |  4.86x |
| 4096  |   0.18 ms (23.0 M/s) |   2.1 ms (2.0 M/s) | 11.80x |
| 16384 |   0.50 ms (33.0 M/s) |   8.2 ms (2.0 M/s) | 16.58x |
| 65536 |   3.31 ms (19.8 M/s) |  33.8 ms (1.9 M/s) | 10.20x |
| 262144 | 14.27 ms (18.4 M/s) | 131.0 ms (2.0 M/s) |  9.18x |
| 1048576| 56.46 ms (18.6 M/s) | 540.3 ms (1.9 M/s) |  9.57x |
| 4194304|242.06 ms (17.3 M/s) |2160.2 ms (1.9 M/s) |  8.92x |

Bare kernel (no FFI overhead, no Mont conversion) hits **1.63 G-perms/sec**
on the same GPU, so the dominant cost is host-side Mont <-> canonical
conversion (~80% of wall clock at 64K+ perms). To unlock that, a real
prover integration would keep state on-device in canonical form between
calls — that is the next step.

## Build

Requires CUDA toolkit (nvcc) with `sm_120` target and a checked-out forge
repo at `/home/garrick/forge` (path is hard-coded in `build.rs` for now).

```
cargo test  --release      # KAT + 1024-perm batch byte-id vs Plonky3
cargo bench --bench perm_throughput
```

## License

MIT or Apache-2.0 at your option.
