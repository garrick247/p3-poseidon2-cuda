//! Build the FORGE-emitted, FORGE-verified BB-16 Poseidon2 perm kernel
//! (demos/2009_*.cu) into a static lib that this crate then links.
//!
//! The .cu is vendored at vendor/forge_2009_bench_baby_bear_fused_perm_factored.cu
//! (originally generated from forge demos/2009_*.fg by FORGE/Z3 at commit
//! 4a13f03 of garrick247/forge). It contains a demo `main` that we
//! silently rename via -Dmain=_forge_demo_main_2009.

use std::process::Command;
use std::path::{Path, PathBuf};

fn main() {
    let forge_cu = Path::new(
        "vendor/forge_2009_bench_baby_bear_fused_perm_factored.cu"
    );
    let glue_cu = Path::new("src/glue.cu");
    println!("cargo:rerun-if-changed={}", forge_cu.display());
    println!("cargo:rerun-if-changed={}", glue_cu.display());

    let out_dir: PathBuf = std::env::var("OUT_DIR").unwrap().into();
    let kernel_o = out_dir.join("forge_kernel.o");
    let glue_o = out_dir.join("glue.o");

    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| {
        if Path::new("/usr/local/cuda/bin/nvcc").exists() {
            "/usr/local/cuda/bin/nvcc".to_string()
        } else {
            "nvcc".to_string()
        }
    });

    // Compile the FORGE-emitted .cu (rename its demo main).
    let st = Command::new(&nvcc)
        .args(["-O3", "-arch=sm_120", "-Xcompiler", "-fPIC",
               "-Dmain=_forge_demo_main_2009", "-c"])
        .arg(forge_cu)
        .arg("-o").arg(&kernel_o)
        .status().expect("nvcc failed (is CUDA installed?)");
    assert!(st.success(), "nvcc failed for forge kernel");

    // Compile the glue (extern-C wrappers around the kernel).
    let st = Command::new(&nvcc)
        .args(["-O3", "-arch=sm_120", "-Xcompiler", "-fPIC", "-c"])
        .arg(glue_cu)
        .arg("-o").arg(&glue_o)
        .status().expect("nvcc failed");
    assert!(st.success(), "nvcc failed for glue");

    // Bundle into a static lib.
    let lib = out_dir.join("libforge_poseidon2.a");
    let st = Command::new("ar")
        .args(["rcs"]).arg(&lib).arg(&kernel_o).arg(&glue_o)
        .status().expect("ar failed");
    assert!(st.success());

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=forge_poseidon2");

    // Link CUDA runtime.
    let cuda_lib = if Path::new("/usr/local/cuda/lib64").exists() {
        "/usr/local/cuda/lib64"
    } else {
        "/usr/lib/x86_64-linux-gnu"
    };
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
