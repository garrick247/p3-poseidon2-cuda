// FFI glue: expose a flat-pointer C ABI on top of the FORGE-emitted
// span-struct kernel (demos/2009_*).
//
// The verified kernel signature is:
//   __global__ void baby_bear_fused_perm_factored_kernel(
//       forge_span_u32_t out, forge_span_u32_t state, uint64_t n);
// where forge_span_u32_t = { uint32_t* data; uintptr_t len; }.

#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

typedef struct { uint32_t* data; uintptr_t len; } forge_span_u32_t;

extern __global__ void baby_bear_fused_perm_factored_kernel(
    forge_span_u32_t out, forge_span_u32_t state, uint64_t n);

extern "C" int cuda_poseidon2_bb16_perm_batch(
    const uint32_t* state_h, uint32_t* out_h, uint64_t n)
{
    if (n == 0) return 0;
    const size_t bytes = (size_t)16 * (size_t)n * sizeof(uint32_t);

    uint32_t *state_d = nullptr, *out_d = nullptr;
    cudaError_t err;

    err = cudaMalloc(&state_d, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc state: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMalloc(&out_d, bytes);
    if (err != cudaSuccess) { cudaFree(state_d); return -2; }

    err = cudaMemcpy(state_d, state_h, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(state_d); cudaFree(out_d); return -3; }

    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);

    forge_span_u32_t out_span   = { out_d,   (uintptr_t)(16 * n) };
    forge_span_u32_t state_span = { state_d, (uintptr_t)(16 * n) };
    baby_bear_fused_perm_factored_kernel<<<blocks, threads>>>(out_span, state_span, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(state_d); cudaFree(out_d); return -4; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(state_d); cudaFree(out_d); return -5; }

    err = cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { cudaFree(state_d); cudaFree(out_d); return -6; }

    cudaFree(state_d);
    cudaFree(out_d);
    return 0;
}


// ==============================================================
// Session-based extension (added 2026-05-15): persistent device +
// pinned-host buffers reused across calls, eliminating per-call
// cudaMalloc/cudaFree and unlocking the pinned-memory PCIe fast path.
// ==============================================================

#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t* state_d;
    uint32_t* out_d;
    uint32_t* state_h_pin;
    uint32_t* out_h_pin;
    size_t    capacity_n;
} cuda_poseidon2_session_t;

static void session_free_buffers(cuda_poseidon2_session_t* s) {
    if (!s) return;
    if (s->state_d)     { cudaFree(s->state_d);         s->state_d = nullptr; }
    if (s->out_d)       { cudaFree(s->out_d);           s->out_d = nullptr; }
    if (s->state_h_pin) { cudaFreeHost(s->state_h_pin); s->state_h_pin = nullptr; }
    if (s->out_h_pin)   { cudaFreeHost(s->out_h_pin);   s->out_h_pin = nullptr; }
    s->capacity_n = 0;
}

static int session_ensure_capacity(cuda_poseidon2_session_t* s, size_t n) {
    if (n <= s->capacity_n) return 0;
    size_t new_cap = s->capacity_n + s->capacity_n / 2;
    if (new_cap < n) new_cap = n;
    const size_t bytes = new_cap * 16 * sizeof(uint32_t);

    session_free_buffers(s);

    cudaError_t err;
    err = cudaMalloc(&s->state_d, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "session cudaMalloc state_d: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMalloc(&s->out_d, bytes);
    if (err != cudaSuccess) { session_free_buffers(s); return -2; }
    err = cudaMallocHost((void**)&s->state_h_pin, bytes);
    if (err != cudaSuccess) { session_free_buffers(s); return -3; }
    err = cudaMallocHost((void**)&s->out_h_pin, bytes);
    if (err != cudaSuccess) { session_free_buffers(s); return -4; }
    s->capacity_n = new_cap;
    return 0;
}

extern "C" int cuda_poseidon2_session_create(cuda_poseidon2_session_t** out_session) {
    cuda_poseidon2_session_t* s = (cuda_poseidon2_session_t*)calloc(1, sizeof(*s));
    if (!s) return -1;
    *out_session = s;
    return 0;
}

extern "C" int cuda_poseidon2_session_destroy(cuda_poseidon2_session_t* s) {
    if (!s) return 0;
    session_free_buffers(s);
    free(s);
    return 0;
}

extern "C" int cuda_poseidon2_session_perm_batch(
    cuda_poseidon2_session_t* s,
    const uint32_t* state_h, uint32_t* out_h, uint64_t n)
{
    if (n == 0) return 0;
    if (!s) return -100;
    int rc = session_ensure_capacity(s, (size_t)n);
    if (rc != 0) return rc;

    const size_t bytes = (size_t)16 * (size_t)n * sizeof(uint32_t);
    cudaError_t err;

    memcpy(s->state_h_pin, state_h, bytes);
    err = cudaMemcpy(s->state_d, s->state_h_pin, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -5;

    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);
    forge_span_u32_t out_span   = { s->out_d,   (uintptr_t)(16 * n) };
    forge_span_u32_t state_span = { s->state_d, (uintptr_t)(16 * n) };
    baby_bear_fused_perm_factored_kernel<<<blocks, threads>>>(out_span, state_span, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) return -6;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -7;

    err = cudaMemcpy(s->out_h_pin, s->out_d, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -8;

    memcpy(out_h, s->out_h_pin, bytes);
    return 0;
}
