# Code Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** Correct normalization boundaries, GPU FFT plan typing, the CPU Laplacian fallback, and the release version identified by the code review.

**Architecture:** Transform kernels continue to use the canonical orthonormal convention with the Condon–Shortley phase. Public transform boundaries convert configured coefficient conventions through shared helpers and retain an identity fast path. CUDA plan storage is parametric over the concrete forward and inverse plan types, while the CPU Laplacian implementation lives in the core package and is reused by the GPU wrapper.

**Tech Stack:** Julia 1.12, Test, FFTW, CUDA extension APIs, Aqua, JET.

### Task 1: Normalization boundaries

**Files:**
- Modify: `test/serial/test_normalization.jl`
- Modify: `test/serial/test_turbo.jl`
- Modify: `test/serial/test_rotation_gradients.jl`
- Modify: `test/serial/test_vorticity_inverse.jl`
- Modify: `src/normalization.jl`
- Modify: scalar, vector, QST, packed, batch, plan, and diagnostic transform boundary files under `src/`
- Modify: normalization boundaries in the LoopVectorization, CUDA, advanced-AD, and parallel extensions

1. Add regression tests proving configured and canonical transforms describe the same field while returning differently scaled coefficients.
2. Run the focused test and confirm it fails because transforms ignore `cfg.norm` and `cfg.cs_phase`.
3. Add shared configured↔canonical helpers with an identity fast path and use them at public boundaries.
4. Re-run focused normalization and transform tests.

### Task 2: CPU Laplacian fallback

**Files:**
- Modify: `test/serial/test_operators.jl`
- Modify: `src/operators.jl`
- Modify: `ext/SHTnsKitGPUExt.jl`

1. Add a failing dense-matrix Laplacian regression test.
2. Add the core in-place implementation and route the CUDA wrapper's CPU branch to it.
3. Re-run the focused operator tests.

### Task 3: CUDA plan type compatibility

**Files:**
- Modify: `test/serial/test_cleanup_contract.jl`
- Modify: `ext/SHTnsKitGPUExt.jl`

1. Add a failing structural contract test that does not require CUDA hardware.
2. Parameterize `CuFFTPlan` over its actual plan and buffer types.
3. Run the serial contract test; verify the extension parses and document that a CUDA runtime allocation could not be exercised locally.

### Task 4: Release version

**Files:**
- Modify: `test/serial/test_cleanup_contract.jl`
- Modify: `Project.toml`
- Modify: `CHANGELOG.md`

1. Add a failing release-contract assertion for the documented breaking release.
2. Change the planned release and package version to `2.0.0`.
3. Re-run the contract test.

### Task 5: Verification

1. Run focused tests after each fix.
2. Run `Pkg.test()` with JET and Aqua enabled.
3. Run `git diff --check` and inspect the final diff without changing unrelated files.
