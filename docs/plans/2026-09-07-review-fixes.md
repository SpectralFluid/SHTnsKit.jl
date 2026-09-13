# Code Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct the five numerical/API defects confirmed in the codebase review.

**Architecture:** Keep existing transform kernels and public APIs. Correct element types and missing convention factors at their current boundaries, preserve rotation data needed by pullbacks, and honor the configured order stride in GPU kernels.

**Tech Stack:** Julia, FFTW, ChainRulesCore, ForwardDiff, Zygote, CUDA and KernelAbstractions.

## Validation environment

Use the installed Julia binary directly because the Juliaup launcher cannot read its root-owned configuration:

`/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. --startup-file=no`

An isolated dependency environment at `/private/tmp/shtnskit-review-ad-N7veoU` supplies cached ChainRulesCore and ForwardDiff without modifying this project's dependency files. Load it with `push!(LOAD_PATH, "/private/tmp/shtnskit-review-ad-N7veoU")`.

## Task 1: Scalar synthesis with real coefficients

Files: `src/core_transforms.jl`, `test/serial/test_basic_transforms.jl`.

1. Add tests comparing real/integer coefficients with equivalent complex coefficients through ordinary, in-place, RFFT and complex-output synthesis.
2. Run the test file and confirm the existing real scratch buffers cause method errors.
3. Use `complex(float(eltype(alm)))` for FFT buffers in scalar synthesis, including the truncated helper.
4. Rerun the tests and verify output types and input preservation.

## Task 2: Vorticity-loss synthesis scaling

Files: `src/vorticity_diagnostics.jl`, `test/serial/test_vorticity_inverse.jl`.

1. Add directional finite-difference checks under explicit and environment-selected quadrature scaling, for canonical and configured normalization.
2. Confirm both loss-gradient helpers disagree with finite differences by the missing scale.
3. Multiply their analysis-based gradients by `phi_inv_scale(cfg) / cfg.nlon`, preserving the established Hermitian-weighted gradient convention.
4. Run the entire vorticity inverse test file.

## Task 3: Robert-form local vector evaluation

Files: `src/local.jl`, `ext/ParallelLocal.jl`, `test/serial/test_local.jl`, `test/parallel/test_mpi_parallel_local_correctness.jl`.

1. Compare QST point and latitude evaluations with full-grid synthesis under Robert form, including pole nodes and degree truncation.
2. Confirm tangential components disagree while radial components agree.
3. Apply the same `sin(theta)` factor to local tangential components, including scalar-gradient point evaluation and distributed QST point/latitude evaluation.
4. Run local evaluation and pole/point regression tests, plus the distributed local correctness file with two MPI ranks.

## Task 4: Aliased rotation pullbacks

Files: `ext/SHTnsKitAdvancedADExt.jl`, `ext/SHTnsKitZygoteExt.jl`, `test/serial/test_rotation_gradients.jl`.

1. Add finite-difference checks for a Z rotation whose input/output buffers alias.
2. Confirm the angle pullback applies the rotation phase twice.
3. Preserve the primal information needed by both pullback implementations.
4. Check repeated pullbacks, separate buffers and available Zygote execution.

## Task 5: GPU order masks

Files: `ext/SHTnsKitGPUExt.jl`, `test/gpu/test_mres.jl`, `test/gpu/runtests.jl`.

1. Add numerical checks that excluded orders contribute zero and allowed orders agree with CPU transforms.
2. Observe failures with available kernel execution; document hardware limitations where applicable.
3. Pass `mres` to scalar and vector analysis/synthesis kernels and skip excluded orders.
4. Run GPU tests on available backends and retain CUDA coverage for GPU-capable CI.

## Final verification

Run focused regressions first, then `include("test/serial/runtests.jl")` with available AD dependencies. Check the complete diff and whitespace. Report all unexecuted optional backend tests; leave the changes uncommitted for review.

## Completion and results

All five tasks are implemented with regressions that failed before their fixes and passed afterward.

- Full serial suite: **67,636 / 67,636 assertions passed** with two Julia threads and ChainRulesCore, ForwardDiff, and Zygote loaded from isolated cached environments.
- Distributed local correctness: **137 / 137 assertions passed per rank** with two MPI ranks. The new Robert-form block accounts for 108 assertions per rank; 36 failed per rank before the fix.
- GPU numerical verification: **84 / 84 assertions passed** by executing the production KernelAbstractions kernels on the CPU backend. Permanent CUDA integration tests were added, but could not run on this Mac because CUDA hardware is unavailable and the cached CUDA dependency chain lacks LLVMExtra.
- Focused scalar, vorticity, and local regression files: **395 / 395 assertions passed**. Rotation gradient tests: **130 / 130 assertions passed**, including both ChainRules and Zygote.
- Full diff reviewed and `git diff --check` passed. Dependency files were unchanged. The serial runner skipped optional FFT plan cache and LoopVectorization tests because their extensions were not loaded.

Changes remain uncommitted in the existing workspace.
