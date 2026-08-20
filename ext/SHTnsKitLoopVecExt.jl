module SHTnsKitLoopVecExt

using LoopVectorization
using SHTnsKit
using Base.Threads: @threads

# Turbo-optimized variants live under the SHTnsKit namespace so users can call
# SHTnsKit.analysis_turbo, synthesis_turbo, etc., when LoopVectorization is loaded.

"""
    SHTnsKit.analysis_turbo(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)

Forward transform with LoopVectorization-optimized inner loops. Same API and
output as `SHTnsKit.analysis`.
"""
function SHTnsKit.analysis_turbo(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))

    fC = complex.(f)
    Fφ = SHTnsKit.fft_phi(fC)

    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(Fφ)
    alm = Matrix{CT}(undef, lmax + 1, mmax + 1)
    fill!(alm, 0.0 + 0.0im)

    scaleφ = cfg.cphi
    # Bind cfg fields to locals so the @tturbo loops below operate on plain arrays.
    # LoopVectorization can't analyze property access (cfg.Nlm) inside @tturbo, and cfg is mutable.
    xv = cfg.x; wv = cfg.w
    # Adaptive threading: use nested parallelism for better load balancing
    n_threads = Threads.nthreads()
    if mmax + 1 < n_threads ÷ 2 && nlat > 32
        # Few m modes: parallelize over latitude points with thread-local accumulators
        n_tid = Threads.maxthreadid()
        CT = eltype(alm)  # eltype-derived accumulators (this branch uses plain loops, not @tturbo)
        thread_alm = [Vector{CT}(undef, lmax + 1) for _ in 1:n_tid]
        thread_P_bufs = [Vector{Float64}(undef, lmax + 1) for _ in 1:n_tid]  # per-thread Legendre scratch (hoisted out of the latitude loop)
        for m in 0:mmax
            col = m + 1
            for t in 1:n_tid
                fill!(thread_alm[t], zero(CT))
            end
            if cfg.use_plm_tables && length(cfg.NP_tables) == mmax + 1
                # NP_tables[col][l+1, i] = P̄_l^m already; no extra Nlm multiply
                tbl = cfg.NP_tables[m + 1]
                @threads :static for i in 1:nlat   # :static pins iterations → threadid() stable (no buffer race under task migration)
                    tid = Threads.threadid()
                    local_acc = thread_alm[tid]
                    Fi = Fφ[i, col]
                    wi = wv[i]
                    @inbounds for l in m:lmax
                        local_acc[l + 1] += (wi * tbl[l + 1, i]) * Fi
                    end
                end
            else
                @threads :static for i in 1:nlat   # :static pins iterations → threadid() stable (no buffer race under task migration)
                    tid = Threads.threadid()
                    local_acc = thread_alm[tid]
                    thread_P = thread_P_bufs[tid]
                    SHTnsKit.Plm_norm_row!(thread_P, xv[i], lmax, m)
                    Fi = Fφ[i, col]
                    wi = wv[i]
                    @inbounds for l in m:lmax
                        local_acc[l + 1] += (wi * thread_P[l + 1]) * Fi
                    end
                end
            end
            # Merge thread-local accumulators and apply φ scaling (Nlm already in P̄)
            @inbounds for l in m:lmax
                acc = zero(CT)
                for t in 1:n_tid
                    acc += thread_alm[t][l + 1]
                end
                alm[l + 1, col] = acc * scaleφ
            end
        end
    else
        # Standard m-parallel approach with dynamic scheduling
        @threads :dynamic for m in 0:mmax
            col = m + 1
            if cfg.use_plm_tables && length(cfg.NP_tables) == mmax + 1
                # NP_tables[col][l+1, i] = P̄_l^m already; no extra Nlm multiply
                tbl = cfg.NP_tables[m + 1]
                for i in 1:nlat
                    Fi = Fφ[i, col]
                    wi = wv[i]
                    @tturbo warn_check_args=false for l in m:lmax
                        alm[l + 1, col] += (wi * tbl[l + 1, i]) * Fi
                    end
                end
            else
                thread_P = Vector{Float64}(undef, lmax + 1)
                for i in 1:nlat
                    SHTnsKit.Plm_norm_row!(thread_P, xv[i], lmax, m)
                    Fi = Fφ[i, col]
                    wi = wv[i]
                    @tturbo warn_check_args=false for l in m:lmax
                        alm[l + 1, col] += (wi * thread_P[l + 1]) * Fi
                    end
                end
            end
            # Apply φ scaling only (Nlm already baked into P̄ from Plm_norm_row!)
            @tturbo warn_check_args=false for l in m:lmax
                alm[l + 1, col] *= scaleφ
            end
        end
    end
    return SHTnsKit._externalize_coefficients!(alm, cfg)
end

"""
    SHTnsKit.synthesis_turbo(cfg::SHTnsKit.SHTConfig, alm::AbstractMatrix; real_output::Bool=true)

Inverse transform with LoopVectorization-optimized inner loops. Same API and
output as `SHTnsKit.synthesis`.
"""
function SHTnsKit.synthesis_turbo(cfg::SHTnsKit.SHTConfig, alm::AbstractMatrix; real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))

    alm_int = SHTnsKit._internal_coefficients(alm, cfg)
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm_int)
    Fφ = Matrix{CT}(undef, nlat, nlon)
    fill!(Fφ, 0.0 + 0.0im)

    inv_scaleφ = SHTnsKit.phi_inv_scale(cfg)
    # Bind cfg fields to locals so the @tturbo loops below operate on plain arrays.
    # LoopVectorization can't analyze property access (cfg.Nlm) inside @tturbo, and cfg is mutable.
    xv = cfg.x

    # Adaptive threading: use nested parallelism for better load balancing
    n_threads = Threads.nthreads()
    if mmax + 1 < n_threads ÷ 2 && nlat > 32
        # Few m modes: parallelize over latitude points instead
        n_tid = Threads.maxthreadid()
        thread_P_bufs = [Vector{Float64}(undef, lmax + 1) for _ in 1:n_tid]  # per-thread Legendre scratch (hoisted out of the latitude loop)
        G = Vector{CT}(undef, nlat)  # shared latitude scratch (few-m branch: outer m-loop is serial)
        for m in 0:mmax
            col = m + 1
            if cfg.use_plm_tables && length(cfg.NP_tables) == mmax + 1
                # NP_tables[col][l+1, i] = P̄_l^m already; no extra Nlm multiply
                tbl = cfg.NP_tables[m + 1]
                @threads for i in 1:nlat
                    g_re = 0.0
                    g_im = 0.0
                    @tturbo warn_check_args=false for l in m:lmax
                        c = tbl[l + 1, i]  # P̄ already normalized; no extra Nlm
                        a = alm_int[l + 1, col]
                        g_re += c * real(a)
                        g_im += c * imag(a)
                    end
                    G[i] = complex(g_re, g_im)
                end
            else
                @threads :static for i in 1:nlat   # :static pins iterations → threadid() stable
                    thread_P = thread_P_bufs[Threads.threadid()]
                    SHTnsKit.Plm_norm_row!(thread_P, xv[i], lmax, m)
                    g_re = 0.0
                    g_im = 0.0
                    @tturbo warn_check_args=false for l in m:lmax
                        c = thread_P[l + 1]  # P̄ already normalized; no extra Nlm
                        a = alm_int[l + 1, col]
                        g_re += c * real(a)
                        g_im += c * imag(a)
                    end
                    G[i] = complex(g_re, g_im)
                end
            end
            @tturbo warn_check_args=false for i in 1:nlat
                Fφ[i, col] = inv_scaleφ * G[i]
            end
            if real_output && m > 0
                conj_index = nlon - m + 1
                @inbounds for i in 1:nlat
                    Fφ[i, conj_index] = conj(Fφ[i, col])
                end
            end
        end
    else
        # Standard m-parallel approach with dynamic scheduling.
        # Per-m latitude scratch as columns of one pre-allocated matrix (each m owns
        # a distinct column → race-free, no per-iteration allocation).
        Gcols = Matrix{CT}(undef, nlat, mmax + 1)
        @threads :dynamic for m in 0:mmax
            col = m + 1
            thread_G = view(Gcols, :, col)
            if cfg.use_plm_tables && length(cfg.NP_tables) == mmax + 1
                # NP_tables[col][l+1, i] = P̄_l^m already; no extra Nlm multiply
                tbl = cfg.NP_tables[m + 1]
                for i in 1:nlat
                    g_re = 0.0
                    g_im = 0.0
                    @tturbo warn_check_args=false for l in m:lmax
                        c = tbl[l + 1, i]  # P̄ already normalized; no extra Nlm
                        a = alm_int[l + 1, col]
                        g_re += c * real(a)
                        g_im += c * imag(a)
                    end
                    thread_G[i] = complex(g_re, g_im)
                end
            else
                thread_P = Vector{Float64}(undef, lmax + 1)
                for i in 1:nlat
                    SHTnsKit.Plm_norm_row!(thread_P, xv[i], lmax, m)
                    g_re = 0.0
                    g_im = 0.0
                    @tturbo warn_check_args=false for l in m:lmax
                        c = thread_P[l + 1]  # P̄ already normalized; no extra Nlm
                        a = alm_int[l + 1, col]
                        g_re += c * real(a)
                        g_im += c * imag(a)
                    end
                    thread_G[i] = complex(g_re, g_im)
                end
            end
            @tturbo warn_check_args=false for i in 1:nlat
                Fφ[i, col] = inv_scaleφ * thread_G[i]
            end
            if real_output && m > 0
                conj_index = nlon - m + 1
                @tturbo warn_check_args=false for i in 1:nlat
                    Fφ[i, conj_index] = conj(Fφ[i, col])
                end
            end
        end
    end

    f = SHTnsKit.ifft_phi(Fφ)
    return real_output ? real.(f) : f
end

"""
    SHTnsKit.turbo_apply_laplacian!(cfg::SHTnsKit.SHTConfig, alm::AbstractMatrix)

In-place multiplication by -l(l+1) in spectral space (spherical Laplacian eigenvalue) using
LoopVectorization. The spherical Laplacian satisfies Δ Y_l^m = -l(l+1) Y_l^m.
"""
function SHTnsKit.turbo_apply_laplacian!(cfg::SHTnsKit.SHTConfig, alm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))
    @threads for m in 0:mmax
        col = m + 1
        @tturbo warn_check_args=false for l in m:lmax
            alm[l + 1, col] *= -(l * (l + 1))
        end
    end
    return alm
end

"""
    SHTnsKit.turbo_apply_laplacian!(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex})

In-place Laplacian factor application for packed coefficients (SHTns LM order).
The spherical Laplacian satisfies Δ Y_l^m = -l(l+1) Y_l^m.
"""
function SHTnsKit.turbo_apply_laplacian!(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex})
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    @tturbo warn_check_args=false for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0 + 1]
        Qlm[lm0 + 1] *= -(l * (l + 1))
    end
    return Qlm
end

"""
    SHTnsKit.benchmark_turbo_vs_simd(cfg; trials=3)

Run a simple timing comparison between `analysis`/`synthesis` and their turbo
variants. Returns a NamedTuple with timings and speedup estimates.
"""
function SHTnsKit.benchmark_turbo_vs_simd(cfg::SHTnsKit.SHTConfig; trials::Integer=3)
    nlat, nlon = cfg.nlat, cfg.nlon
    f = randn(nlat, nlon)
    alm = randn(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)

    # Warmup
    A1 = SHTnsKit.analysis(cfg, f)
    S1 = SHTnsKit.synthesis(cfg, A1)
    A2 = SHTnsKit.analysis_turbo(cfg, f)
    S2 = SHTnsKit.synthesis_turbo(cfg, A2)
    GC.gc()

    function timeit(fn)
        tmin = Inf
        for _ in 1:trials
            GC.gc()
            t = @elapsed fn()
            tmin = min(tmin, t)
        end
        return tmin
    end

    t_analysis_baseline = timeit(() -> SHTnsKit.analysis(cfg, f))
    t_analysis_turbo    = timeit(() -> SHTnsKit.analysis_turbo(cfg, f))
    t_synth_baseline    = timeit(() -> SHTnsKit.synthesis(cfg, alm))
    t_synth_turbo       = timeit(() -> SHTnsKit.synthesis_turbo(cfg, alm))

    return (;
        analysis_baseline=t_analysis_baseline,
        analysis_turbo=t_analysis_turbo,
        synthesis_baseline=t_synth_baseline,
        synthesis_turbo=t_synth_turbo,
        analysis_speedup = t_analysis_baseline / max(t_analysis_turbo, eps()),
        synthesis_speedup = t_synth_baseline / max(t_synth_turbo, eps()),
        speedup = t_analysis_baseline / max(t_analysis_turbo, eps()),
    )
end

end # module SHTnsKitLoopVecExt
