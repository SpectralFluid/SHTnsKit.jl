module SHTnsKitAdvancedADExt

using ChainRulesCore
using SHTnsKit
using SHTnsKit: LM_index, LM_cplx_index

    # Materialize Thunk/InplaceableThunk tangents before we try to collect/index.
    # ChainRules 1.x passes lazy tangents into pullbacks (e.g. from sum_abs2)
    # and expects downstream code to `unthunk` before consuming.
    _unthunk(A) = ChainRulesCore.unthunk(A)

    # Helper to ensure array eltype is complex for adjoints when needed.
    # Accepts Thunk tangents by unthunking first.
    _to_complex(A) = let B = _unthunk(A); eltype(B) <: Complex ? B : complex.(B); end

    # Adjoint of analysis now lives in SHTnsKit proper (src/core_transforms.jl).
    # Local alias kept for backward compat with any users touching this symbol.
    const _adjoint_analysis = SHTnsKit._adjoint_analysis

    # ---- normalization in the adjoint -------------------------------------
    #
    # Public transforms emit and consume the convention configured on `cfg`,
    # while their Legendre kernels remain canonical orthonormal+CS. Dense scalar
    # and vector adjoint helpers own their boundary maps. Specialized adjoints
    # that operate directly on packed storage must apply the corresponding
    # transpose map themselves before or after their canonical kernel.
    #
    # `convert_alm_norm!` reaches `_ensure_norm_scale_matrix!`, which lazily
    # builds and caches a
    # constant (l,m) table on the config. That `setindex!` is invisible to a
    # caller but fatal to Zygote ("Mutating arrays is not supported") if a
    # differentiated function ever reaches it. The table does not depend on any
    # differentiated value, so keep the builder declared non-differentiable.
    ChainRulesCore.@non_differentiable SHTnsKit._ensure_norm_scale_matrix!(::Any)

    # A loss that consumes only ONE of a two-output transform hands the other slot
    # a `ZeroTangent`. The `_adjoint_*` kernels take arrays, so materialise it to
    # an explicit zero matrix of the right shape rather than letting it reach them.
    @inline _coeff_zeros(cfg) = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    @inline _materialize_coeff(A, cfg) =
        A isa ChainRulesCore.AbstractZero ? _coeff_zeros(cfg) : _to_complex(A)

    # An output the loss never uses arrives as an AbstractZero cotangent; its
    # pullback owes zero tangents to every differentiable argument. Pullbacks
    # return exactly one tangent per positional argument (keywords get none):
    # an extra entry breaks ChainRulesTestUtils and Diffractor.
    @inline _zero_tangents(n::Int) =
        (NoTangent(), NoTangent(), ntuple(_ -> ZeroTangent(), n)...)


    # `fft_scratch` / `use_rfft` pick a different FFT implementation of the SAME
    # linear operator, so the adjoint is unchanged — but a pullback must still
    # ACCEPT them. Declaring fewer kwargs than the primal made ChainRules skip
    # this rule entirely the moment a caller passed one, even at its default.
    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis), cfg::SHTnsKit.SHTConfig, f;
                                  fft_scratch=nothing, use_rfft::Bool=false)
        y = SHTnsKit.analysis(cfg, f; fft_scratch, use_rfft)
        project_f = ProjectTo(f)
        function pullback(ȳ)
            ȳ = _unthunk(ȳ)
            ȳ isa ChainRulesCore.AbstractZero && return _zero_tangents(1)
            ȳA = _to_complex(ȳ)
            f̄ = project_f(_adjoint_analysis(cfg, ȳA))
            return NoTangent(), NoTangent(), f̄
        end
        return y, pullback
    end

# synthesis(cfg, alm; real_output=true) :: (lmax+1)×(mmax+1) -> (nlat×nlon)
    #
    # The mathematical adjoint of `synthesis` is NOT `analysis`: analysis
    # carries Gauss-Legendre quadrature weights and the cphi azimuthal factor,
    # neither of which appears in the synthesis adjoint. Use the dedicated
    # `_adjoint_synthesis` helper instead. (See `test_adjoint_consistency`
    # in the test suite for an FD verification.)
    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis), cfg::SHTnsKit.SHTConfig,
                                alm; real_output::Bool=true,
                                fft_scratch=nothing, use_rfft::Bool=false)
        y = SHTnsKit.synthesis(cfg, alm; real_output, fft_scratch, use_rfft)
        project_alm = ProjectTo(alm)
        function pullback(ȳ)
            ȳ_mat = ChainRulesCore.unthunk(ȳ)      # materialize Thunk/InplaceableThunk
            ȳ_mat isa ChainRulesCore.AbstractZero && return _zero_tangents(1)
            ȳA = ȳ_mat isa AbstractMatrix ? ȳ_mat : collect(ȳ_mat)
            alm̄ = project_alm(SHTnsKit._adjoint_synthesis(cfg, ȳA; real_output=real_output))
            return NoTangent(), NoTangent(), alm̄
        end
        return y, pullback
    end

    # Batch scalar transforms: analysis_batch, synthesis_batch
    # Each field is an independent scalar transform; adjoint applies the scalar
    # adjoint per slice. Simple and correct; not allocation-optimal (doesn't
    # share Legendre work across fields in the backward pass).
    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis_batch), cfg::SHTnsKit.SHTConfig,
                                 fields::AbstractArray{<:Real,3}; use_rfft::Bool=false)
        y = SHTnsKit.analysis_batch(cfg, fields; use_rfft)
        project_fields = ProjectTo(fields)
        function pullback(ȳ)
            ȳ = _unthunk(ȳ)
            ȳ isa ChainRulesCore.AbstractZero && return _zero_tangents(1)
            ȳA = eltype(ȳ) <: Complex ? ȳ : complex.(ȳ)
            nfields = size(ȳA, 3)
            f̄_raw = Array{complex(float(eltype(ȳA))),3}(undef, cfg.nlat, cfg.nlon, nfields)
            @inbounds for k in 1:nfields
                f̄_raw[:, :, k] .= _adjoint_analysis(cfg, @view ȳA[:, :, k])
            end
            f̄ = project_fields(f̄_raw)
            return NoTangent(), NoTangent(), f̄
        end
        return y, pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_batch), cfg::SHTnsKit.SHTConfig,
                                 alm_batch::AbstractArray{<:Complex,3};
                                 real_output::Bool=true, use_rfft::Bool=false)
        y = SHTnsKit.synthesis_batch(cfg, alm_batch; real_output, use_rfft)
        project_alm_batch = ProjectTo(alm_batch)
        function pullback(ȳ)
            ȳ = _unthunk(ȳ)
            ȳ isa ChainRulesCore.AbstractZero && return _zero_tangents(1)
            nfields = size(ȳ, 3)
            ālm = zeros(complex(float(eltype(ȳ))), cfg.lmax + 1, cfg.mmax + 1, nfields)
            @inbounds for k in 1:nfields
                # Adjoint of synthesis is `_adjoint_synthesis` (NO Gauss quadrature
                # weights), matching the non-batch synthesis rrule. Using `analysis`
                # here was wrong — off by the w_i·cphi factors (FD-checked).
                ālm[:, :, k] .= SHTnsKit._adjoint_synthesis(cfg, @view(ȳ[:, :, k]); real_output=real_output)
            end
            return NoTangent(), NoTangent(), project_alm_batch(ālm)
        end
        return y, pullback
    end

    # Packed scalar transforms: analysis_packed, synthesis_packed
    #
    # `analysis_packed = pack ∘ analysis ∘ reshape` and
    # `synthesis_packed = vec ∘ synthesis ∘ unpack`. Packing is pure re-indexing
    # (it drops the m not divisible by mres, which unpack leaves at zero), so
    # `adjoint(pack) = unpack` and `adjoint(unpack) = pack`; the transform half of
    # each adjoint is the corresponding `_adjoint_*` helper. Using the *inverse*
    # transform instead — `analysis_packed` as the adjoint of `synthesis_packed`
    # and vice versa — is wrong for exactly the reason spelled out above the
    # dense `synthesis` rrule: analysis carries the Gauss weights `w_i·cphi` that
    # the synthesis adjoint must not, and misses the `wm = 2` doubling for m > 0.

    # Dense (l+1, m+1) matrix ↔ packed LM-order vector, skipping m % mres ≠ 0.
    # Thin aliases over the canonical pair in src/layout.jl — this file used to
    # carry its own copy of both loops.
    const _unpack_lm = SHTnsKit.unpack_lm
    const _pack_lm = SHTnsKit.pack_lm

    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis_packed), cfg::SHTnsKit.SHTConfig, Vr)
        y = SHTnsKit.analysis_packed(cfg, Vr)
        project_Vr = ProjectTo(Vr)
        function pullback(ȳ)
            ȳ = _unthunk(ȳ)
            ȳ isa ChainRulesCore.AbstractZero && return _zero_tangents(1)
            Ā = _unpack_lm(cfg, _to_complex(ȳ))
            Vr̄ = project_Vr(vec(_adjoint_analysis(cfg, Ā)))
            return NoTangent(), NoTangent(), Vr̄
        end
        return y, pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_packed), cfg::SHTnsKit.SHTConfig, Qlm)
        y = SHTnsKit.synthesis_packed(cfg, Qlm)
        project_Qlm = ProjectTo(Qlm)
        function pullback(ȳ)
            ȳ = _unthunk(ȳ)
            ȳ isa ChainRulesCore.AbstractZero && return _zero_tangents(1)
            f̄ = reshape(ȳ, cfg.nlat, cfg.nlon)
            Qlm̄ = project_Qlm(_pack_lm(cfg, SHTnsKit._adjoint_synthesis(cfg, f̄; real_output=true)))
            return NoTangent(), NoTangent(), Qlm̄
        end
        return y, pullback
    end

    # Vector sphtor transforms
    # Helper: exact adjoint of analysis_sphtor (analogous to _adjoint_analysis for scalar)
    #
    # Forward analysis_sphtor does:
    #   Fθ, Fφ = fft_phi(Vt), fft_phi(Vp)
    #   S_lm = sum_i { w_i * scaleφ / ll1 * (dθY * Fθ + conj(term) * Fφ) }
    #   T_lm = sum_i { w_i * scaleφ / ll1 * (-conj(term) * Fθ + dθY * Fφ) }
    # where term = i*m*Y/sinθ, scaleφ = 2π/nlon
    #
    # The adjoint maps (S̄, T̄) → (V̄t, V̄p):
    #   F̄θ[i,m] = φadj * w_i * sum_l { (1/ll1) * (dθY * S̄ - conj(term) * T̄) }
    #   F̄φ[i,m] = φadj * w_i * sum_l { (1/ll1) * (conj(term) * S̄ + dθY * T̄) }
    #   V̄t, V̄p = ifft_phi(F̄θ), ifft_phi(F̄φ)
    # followed by projection onto each spatial primal's tangent space.
    # where φadj = nlon * scaleφ = 2π (same as scalar adjoint)
    # sphtor adjoint analysis now lives in SHTnsKit proper (src/sphtor_transforms.jl).
    # Keep local alias for any direct callers of the ext symbol.
    const _adjoint_analysis_sphtor = SHTnsKit._adjoint_analysis_sphtor

    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis_sphtor), cfg::SHTnsKit.SHTConfig, Vt, Vp;
                                  use_rfft::Bool=false)
        Slm, Tlm = SHTnsKit.analysis_sphtor(cfg, Vt, Vp; use_rfft)
        project_Vt = ProjectTo(Vt)
        project_Vp = ProjectTo(Vp)
        function pullback(ṠTl)
            ṠTl = _unthunk(ṠTl)
            ṠTl isa ChainRulesCore.AbstractZero && return _zero_tangents(2)
            Slm̄, Tlm̄ = ṠTl
            # analysis-like: the primal divides by M on the way out, so the
            # cotangent is divided before the internal-convention adjoint.
            # `_materialize_coeff` turns a ZeroTangent slot into explicit zeros.
            S̄ = _materialize_coeff(Slm̄, cfg)
            T̄ = _materialize_coeff(Tlm̄, cfg)
            V̄t, V̄p = _adjoint_analysis_sphtor(cfg, S̄, T̄)
            return NoTangent(), NoTangent(), project_Vt(V̄t), project_Vp(V̄p)
        end
        return (Slm, Tlm), pullback
    end

    # synthesis_sphtor adjoint now lives in SHTnsKit proper (src/core_transforms.jl,
    # parametrized over θ_globals so the distributed AD ext reuses the same kernel).
    # Local alias kept for backward compat with any users touching this symbol.
    const _adjoint_synthesis_sphtor = SHTnsKit._adjoint_synthesis_sphtor

    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_sphtor), cfg::SHTnsKit.SHTConfig,
                                Slm, Tlm; real_output::Bool=true, use_rfft::Bool=false)
        Vt, Vp = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output, use_rfft)
        project_Slm = ProjectTo(Slm)
        project_Tlm = ProjectTo(Tlm)
        function pullback(Ṽ)
            # Materialize (possibly Inplaceable)Thunk components before indexing —
            # a sum(abs2,·) loss delivers thunked cotangents in a Tangent tuple.
            V̄t = ChainRulesCore.unthunk(Ṽ[1])
            V̄p = ChainRulesCore.unthunk(Ṽ[2])
            # A loss touching only Vt (or only Vp) leaves the other a ZeroTangent.
            zsp() = zeros(Float64, cfg.nlat, cfg.nlon)
            V̄t = V̄t isa ChainRulesCore.AbstractZero ? zsp() : V̄t
            V̄p = V̄p isa ChainRulesCore.AbstractZero ? zsp() : V̄p
            S̄, T̄ = SHTnsKit._adjoint_synthesis_sphtor(cfg, V̄t, V̄p; real_output=real_output)
            return NoTangent(), NoTangent(), project_Slm(S̄), project_Tlm(T̄)
        end
        return (Vt, Vp), pullback
    end

    # QST (3-component) transforms. `synthesis_qst` is the scalar synthesis of Q
    # alongside the sphtor synthesis of (S,T), and `analysis_qst` is the mirror,
    # so each adjoint is just the two existing adjoints side by side. Without
    # these, differentiating a QST pipeline fell through to Zygote's source
    # tracing and crashed inside FFTW.
    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_qst), cfg::SHTnsKit.SHTConfig,
                                  Qlm, Slm, Tlm; real_output::Bool=true,
                                  use_rfft::Bool=false)
        Vr, Vt, Vp = SHTnsKit.synthesis_qst(cfg, Qlm, Slm, Tlm; real_output, use_rfft)
        project_Q = ProjectTo(Qlm); project_S = ProjectTo(Slm); project_T = ProjectTo(Tlm)
        function pullback(V̄)
            zsp() = zeros(Float64, cfg.nlat, cfg.nlon)
            V̄r = ChainRulesCore.unthunk(V̄[1]); V̄r = V̄r isa ChainRulesCore.AbstractZero ? zsp() : V̄r
            V̄t = ChainRulesCore.unthunk(V̄[2]); V̄t = V̄t isa ChainRulesCore.AbstractZero ? zsp() : V̄t
            V̄p = ChainRulesCore.unthunk(V̄[3]); V̄p = V̄p isa ChainRulesCore.AbstractZero ? zsp() : V̄p
            Q̄ = SHTnsKit._adjoint_synthesis(cfg, V̄r; real_output=real_output)
            S̄, T̄ = SHTnsKit._adjoint_synthesis_sphtor(cfg, V̄t, V̄p; real_output=real_output)
            return NoTangent(), NoTangent(), project_Q(Q̄), project_S(S̄), project_T(T̄)
        end
        return (Vr, Vt, Vp), pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis_qst), cfg::SHTnsKit.SHTConfig,
                                  Vr, Vt, Vp; use_rfft::Bool=false)
        Qlm, Slm, Tlm = SHTnsKit.analysis_qst(cfg, Vr, Vt, Vp; use_rfft)
        project_Vr = ProjectTo(Vr); project_Vt = ProjectTo(Vt); project_Vp = ProjectTo(Vp)
        function pullback(Ā)
            Q̄ = _materialize_coeff(Ā[1], cfg)
            S̄ = _materialize_coeff(Ā[2], cfg)
            T̄ = _materialize_coeff(Ā[3], cfg)
            V̄r = _adjoint_analysis(cfg, Q̄)
            V̄t, V̄p = _adjoint_analysis_sphtor(cfg, S̄, T̄)
            return NoTangent(), NoTangent(), project_Vr(V̄r), project_Vt(V̄t), project_Vp(V̄p)
        end
        return (Qlm, Slm, Tlm), pullback
    end

    # Energy diagnostics are quadratic forms whose gradients the core already
    # provides (`grad_energy_scalar_alm`, ...). Without rules Zygote traced their
    # scalar loops and recorded a closure per coefficient: about 33 GB for
    # `zgrad_scalar_energy` at lmax=255. The primal operands are copied because
    # callers may reuse the coefficient buffer before the pullback runs.
    function ChainRulesCore.rrule(::typeof(SHTnsKit.energy_scalar), cfg::SHTnsKit.SHTConfig,
                                  alm::AbstractMatrix; real_field::Bool=true)
        E = SHTnsKit.energy_scalar(cfg, alm; real_field)
        alm0 = copy(alm)
        project_alm = ProjectTo(alm)
        function energy_scalar_pullback(Ē)
            Ē = _unthunk(Ē)
            Ē isa ChainRulesCore.AbstractZero &&
                return NoTangent(), NoTangent(), ZeroTangent()
            ḡ = SHTnsKit.grad_energy_scalar_alm(cfg, alm0; real_field)
            return NoTangent(), NoTangent(), project_alm(Ē .* ḡ)
        end
        return E, energy_scalar_pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.energy_vector), cfg::SHTnsKit.SHTConfig,
                                  Slm::AbstractMatrix, Tlm::AbstractMatrix;
                                  real_field::Bool=true)
        E = SHTnsKit.energy_vector(cfg, Slm, Tlm; real_field)
        S0 = copy(Slm); T0 = copy(Tlm)
        project_S = ProjectTo(Slm); project_T = ProjectTo(Tlm)
        function energy_vector_pullback(Ē)
            Ē = _unthunk(Ē)
            Ē isa ChainRulesCore.AbstractZero &&
                return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
            gS, gT = SHTnsKit.grad_energy_vector_Slm_Tlm(cfg, S0, T0; real_field)
            return NoTangent(), NoTangent(), project_S(Ē .* gS), project_T(Ē .* gT)
        end
        return E, energy_vector_pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.enstrophy), cfg::SHTnsKit.SHTConfig,
                                  Tlm::AbstractMatrix; real_field::Bool=true)
        Z = SHTnsKit.enstrophy(cfg, Tlm; real_field)
        T0 = copy(Tlm)
        project_T = ProjectTo(Tlm)
        function enstrophy_pullback(Z̄)
            Z̄ = _unthunk(Z̄)
            Z̄ isa ChainRulesCore.AbstractZero &&
                return NoTangent(), NoTangent(), ZeroTangent()
            return NoTangent(), NoTangent(),
                   project_T(Z̄ .* SHTnsKit.grad_enstrophy_Tlm(cfg, T0; real_field))
        end
        return Z, enstrophy_pullback
    end

    # Complex packed (LM_cplx layout — both signs of m stored explicitly).
    #
    # Both transforms are ℂ-linear in their argument (no `real()` anywhere), so
    # each adjoint is the conjugate transpose of the forward operator — NOT the
    # other transform, which differs by the quadrature weights `w_i·cphi` exactly
    # as in the real packed pair above.

    """
        _adjoint_synthesis_packed_cplx(cfg, z̄) -> packed LM_cplx cotangent

    Adjoint of `synthesis_packed_cplx`. The forward writes DFT bin `am+1` from
    `a_{l,+am}` and bin `nlon-am+1` from `a_{l,-am}`, both through the SAME real
    row `P̄_l^{|m|}` and the same real norm·CS scale `M[l,|m|]`. `_adjoint_synthesis`
    (with `real_output=false`, i.e. no `wm` doubling) already delivers
    `Σ_i P̄ · fft(·)[i, am+1]`, so the +m half is a direct call; the −m half uses
    `fft(conj z̄)[i, am+1] = conj(fft(z̄)[i, nlon-am+1])` to reach the mirrored bins
    with the same kernel.
    """
    function _adjoint_synthesis_packed_cplx(cfg::SHTnsKit.SHTConfig, z̄::AbstractMatrix)
        cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
        lmax, mmax = cfg.lmax, cfg.mmax
        Ap = SHTnsKit._adjoint_synthesis(cfg, z̄;        real_output=false)
        Am = SHTnsKit._adjoint_synthesis(cfg, conj.(z̄); real_output=false)
        ā = zeros(eltype(Ap), SHTnsKit.nlm_cplx_calc(lmax, mmax, 1))
        @inbounds for l in 0:lmax
            ā[LM_cplx_index(lmax, mmax, l, 0) + 1] = Ap[l+1, 1]
            for m in 1:min(l, mmax)
                ā[LM_cplx_index(lmax, mmax, l,  m) + 1] = Ap[l+1, m+1]
                ā[LM_cplx_index(lmax, mmax, l, -m) + 1] = conj(Am[l+1, m+1])
            end
        end
        return ā
    end

    """
        _adjoint_analysis_packed_cplx(cfg, ā) -> spatial cotangent (nlat × nlon)

    Adjoint of `analysis_packed_cplx`. Mirrors `SHTnsKit._adjoint_analysis`
    (`F̄ = w_i·cphi·Σ_l P̄·ā`, then `adjoint(fft) = nlon·ifft`) but keeps BOTH DFT
    bins per `|m|` and does not take `real()` — the forward input is a complex
    field, so its cotangent is complex too.
    """
    function _adjoint_analysis_packed_cplx(cfg::SHTnsKit.SHTConfig, ā::AbstractVector)
        cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
        lmax, mmax = cfg.lmax, cfg.mmax
        nlat, nlon = cfg.nlat, cfg.nlon
        length(ā) == SHTnsKit.nlm_cplx_calc(lmax, mmax, 1) || throw(DimensionMismatch("ā length"))
        ā_int = SHTnsKit._analysis_cotangent_to_canonical(ā, cfg)
        CT = complex(float(eltype(ā_int)))
        F̄ = zeros(CT, nlat, nlon)
        P = Vector{Float64}(undef, lmax + 1)
        scaleφ = SHTnsKit._analysis_phi_scale(cfg)
        xv = cfg.x; wv = cfg.w
        for am in 0:mmax
            colp = am + 1
            coln = nlon - am + 1
            for i in 1:nlat
                SHTnsKit.Plm_norm_row!(P, xv[i], lmax, am)
                wi = wv[i] * scaleφ
                gp = zero(CT); gn = zero(CT)
                @inbounds for l in am:lmax
                    base = wi * P[l+1]
                    gp += base * ā_int[LM_cplx_index(lmax, mmax, l, am) + 1]
                    if am > 0
                        gn += base * ā_int[LM_cplx_index(lmax, mmax, l, -am) + 1]
                    end
                end
                F̄[i, colp] += gp
                am > 0 && (F̄[i, coln] += gn)
            end
        end
        return nlon .* SHTnsKit.ifft_phi(F̄)
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis_packed_cplx), cfg::SHTnsKit.SHTConfig, z)
        y = SHTnsKit.analysis_packed_cplx(cfg, z)
        function pullback(ȳ)
            ȳ = _unthunk(ȳ)
            ȳ isa ChainRulesCore.AbstractZero && return _zero_tangents(1)
            z̄ = _adjoint_analysis_packed_cplx(cfg, _to_complex(ȳ))
            return NoTangent(), NoTangent(), z̄
        end
        return y, pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_packed_cplx), cfg::SHTnsKit.SHTConfig, alm)
        y = SHTnsKit.synthesis_packed_cplx(cfg, alm)
        function pullback(ȳ)
            ȳ = _unthunk(ȳ)
            ȳ isa ChainRulesCore.AbstractZero && return _zero_tangents(1)
            alm̄ = _adjoint_synthesis_packed_cplx(cfg, _to_complex(ȳ))
            return NoTangent(), NoTangent(), alm̄
        end
        return y, pullback
    end

# Rotations of packed real-field coefficients are ℝ-linear (negative orders are
# conjugates of the stored ones), so their adjoint is not the inverse rotation:
# `W·R⁻¹·W⁻¹` is only correct for cotangents with real m=0 entries. The exact
# adjoint lives in SHTnsKit (`_rotation_apply_real_adjoint`) and is shared with
# the Zygote extension. The coefficient and angle pullbacks below are
# finite-difference verified in test/serial/test_rotation_gradients.jl.
#
# `rotation` is the FORWARD rotation of the configured primal F = C⁻¹ R C, whose
# adjoint is C Rᵀ C⁻¹.
function _configured_rotation_adjoint!(cfg, rotation, ȳ, Q̄)
    ȳ_canonical = SHTnsKit._analysis_cotangent_to_canonical(ȳ, cfg)
    Q̄_canonical = SHTnsKit._rotation_apply_real_adjoint(rotation, ȳ_canonical)
    if SHTnsKit._uses_canonical_convention(cfg)
        copyto!(Q̄, Q̄_canonical)
    else
        SHTnsKit.convert_alm_norm!(Q̄, Q̄_canonical, cfg; to_internal=true)
    end
    return Q̄
end

# The forward rotations behind the configured axis wrappers.
function _axis_rotation(cfg, α, β, γ)
    rotation = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax)
    SHTnsKit.shtns_rotation_set_angles_ZYZ(rotation, α, β, γ)
    return rotation
end

# Rotation kernels consume canonical ZYZ angles, which can differ from the
# public/stored fields because setter-created rotations reverse the outer
# angles and ZXZ adds constant phase offsets.  Pullbacks differentiate in the
# canonical coordinates, then return the tangent in the stored parameter
# order expected by callers of the setters.
@inline _rotation_rrule_angles(r::SHTnsKit.SHTRotation) =
    SHTnsKit._rotation_zyz_angles(r, Float64)

@inline function _rotation_rrule_tangent(r::SHTnsKit.SHTRotation, gα, gβ, gγ)
    αbar, βbar, γbar = r.reverse_outer ? (gγ, gβ, gα) : (gα, gβ, gγ)
    return Tangent{SHTnsKit.SHTRotation}(; α=αbar, β=βbar, γ=γbar)
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Zrotate), cfg::SHTnsKit.SHTConfig, Qlm, alpha::Real, Rlm)
    y = SHTnsKit.SH_Zrotate(cfg, Qlm, alpha, Rlm)
    # In-place rotation overwrites Qlm, and callers may reuse either buffer
    # before the pullback. Preserve the primal values needed for dR/dα.
    rotated = copy(y)
    function pullback(ȳ)
        ȳ = _unthunk(ȳ)
        ȳ isa ChainRulesCore.AbstractZero &&
            return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent()
        # Diagonal rotation Rlm = Qlm·e^{-imα} ⇒ Q̄ = ȳ·e^{imα} = SH_Zrotate(ȳ, -α).
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Zrotate(cfg, ȳ, -alpha, Q̄)
        # angle gradient: dR/dα = -i m R
        dα = zero(float(alpha))
        for m in 0:cfg.mmax
            (m % cfg.mres == 0) || continue
            for l in m:cfg.lmax
                lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                dα += real(conj(ȳ[lm]) * (-im * m * rotated[lm]))
            end
        end
        return NoTangent(), NoTangent(), Q̄, dα, ZeroTangent()
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Yrotate), cfg::SHTnsKit.SHTConfig, Qlm, alpha::Real, Rlm)
    # The primal permits overlapping input/output storage. Save the input in
    # the canonical basis before it is overwritten or reused by the caller.
    Qlm_canonical = copy(SHTnsKit._internal_coefficients(Qlm, cfg))
    y = SHTnsKit.SH_Yrotate(cfg, Qlm, alpha, Rlm)
    function pullback(ȳ)
        ȳ = _unthunk(ȳ)
        ȳ isa ChainRulesCore.AbstractZero &&
            return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent()
        Q̄ = similar(Qlm)
        _configured_rotation_adjoint!(cfg, _axis_rotation(cfg, 0.0, float(alpha), 0.0), ȳ, Q̄)
        # angle gradient via d/dβ of Wigner-d at β=alpha
        dα = zero(float(alpha))
        lmax, mmax = cfg.lmax, cfg.mmax
        ȳ_canonical = SHTnsKit._analysis_cotangent_to_canonical(ȳ, cfg)
        dl = Matrix{typeof(float(alpha))}(undef, 2lmax + 1, 2lmax + 1)
        dwork = similar(dl); dd = similar(dl)
        for l in 0:lmax
            mm = min(l, mmax)
            b = zeros(eltype(ȳ_canonical), 2l+1)
            # b = A because γ=0, A from packed Qlm
            for mp in -mm:mm
                idxp = LM_index(lmax, 1, l, abs(mp)) + 1
                # reconstruct complex A using hermitian symmetry for real field
                if mp == 0
                    b[mp + l + 1] = Qlm_canonical[idxp]
                elseif mp > 0
                    b[mp + l + 1] = Qlm_canonical[idxp]
                    b[-mp + l + 1] = (-1)^mp * conj(Qlm_canonical[idxp])
                end
            end
            SHTnsKit._wigner_d_advance!(dl, dwork, l, float(alpha))
            SHTnsKit._wigner_d_deriv!(dd, dl, l)
            # ∂R_m = (dd * b)_m for m>=0 (no left/right phases)
            for m in 0:mm
                lm = LM_index(lmax, 1, l, m) + 1
                s = zero(eltype(ȳ_canonical))
                for mp in -l:l
                    s += dd[m + l + 1, mp + l + 1] * b[mp + l + 1]
                end
                dα += real(conj(ȳ_canonical[lm]) * s)
            end
        end
        return NoTangent(), NoTangent(), Q̄, dα, ZeroTangent()
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Yrotate90), cfg::SHTnsKit.SHTConfig, Qlm, Rlm)
    y = SHTnsKit.SH_Yrotate90(cfg, Qlm, Rlm)
    function pullback(ȳ)
        ȳ = _unthunk(ȳ)
        ȳ isa ChainRulesCore.AbstractZero &&
            return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
        Q̄ = similar(Qlm)
        _configured_rotation_adjoint!(cfg, _axis_rotation(cfg, 0.0, π/2, 0.0), ȳ, Q̄)
        return NoTangent(), NoTangent(), Q̄, ZeroTangent()
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Xrotate90), cfg::SHTnsKit.SHTConfig, Qlm, Rlm)
    y = SHTnsKit.SH_Xrotate90(cfg, Qlm, Rlm)
    function pullback(ȳ)
        ȳ = _unthunk(ȳ)
        ȳ isa ChainRulesCore.AbstractZero &&
            return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
        # Same setter call as the primal (`SH_Xrotate90` in src/rotations.jl).
        Q̄ = similar(Qlm)
        _configured_rotation_adjoint!(cfg, _axis_rotation(cfg, π/2, π/2, -π/2), ȳ, Q̄)
        return NoTangent(), NoTangent(), Q̄, ZeroTangent()
    end
    return y, pullback
end

# Diagonal map from the configured coefficient convention to the canonical
# basis. The primal is C⁻¹ R C, so its adjoint is C R* C⁻¹.
function _rotation_rrule_scales(r::SHTnsKit.SHTRotation, coefficients;
                                 real_packed::Bool=false)
    RT = typeof(real(zero(eltype(coefficients))))
    scales = Vector{RT}(undef, length(coefficients))
    for l in 0:r.lmax
        mm = min(l, r.mmax)
        for m in (real_packed ? 0 : -mm):mm
            index = real_packed ? LM_index(r.lmax, 1, l, m) + 1 :
                                  LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            scales[index] = SHTnsKit._rotation_coefficient_scale(r, l, m)
        end
    end
    return scales
end

# Adjoint for complex rotation using conjugate-transpose of Wigner-D
function ChainRulesCore.rrule(::typeof(SHTnsKit.shtns_rotation_apply_cplx), r::SHTnsKit.SHTRotation, Zlm, Rlm)
    rotation = deepcopy(r)
    lmax, mmax = rotation.lmax, rotation.mmax
    α, β, γ = _rotation_rrule_angles(rotation)
    scales = _rotation_rrule_scales(rotation, Zlm)
    ε = SHTnsKit._lmcplx_ybasis_signs(lmax, mmax)
    # Materialize the forward intermediates before the alias-safe primal can
    # overwrite Zlm. These saved values also allow repeated pullback calls.
    Zε = ε .* scales .* Zlm
    y = SHTnsKit.shtns_rotation_apply_cplx(r, Zlm, Rlm)
    function pullback(ȳ)
        ȳ = _unthunk(ȳ)
        ȳ isa ChainRulesCore.AbstractZero &&
            return NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent()
        Z̄ = similar(Zlm)
        fill!(Z̄, zero(eltype(Z̄)))
        # This pullback reimplements the Wigner engine, which works in the Y_l^m
        # basis, while the primal is `ε ∘ engine ∘ ε` in the packed LM_cplx layout
        # (see SHTnsKit._lmcplx_ybasis_signs). Convert both the input and the
        # incoming cotangent into the engine's basis and convert the result back;
        # ε is real and self-inverse, so the angle gradients are unaffected.
        ȳε = ε .* (_unthunk(ȳ) ./ scales)
        gα = 0.0; gβ = 0.0; gγ = 0.0
        dbuf = Matrix{typeof(β)}(undef, 2lmax + 1, 2lmax + 1)
        dwork = similar(dbuf); ddbuf = similar(dbuf)
        for l in 0:lmax
            mm = min(l, mmax)
            n = 2l + 1
            # c̄_m = e^{+i m α} ȳ_m
            cbar = zeros(eltype(Zε), n)
            for m in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, m) + 1
                cbar[m + l + 1] = ȳε[idx] * cis(m * α)
                # α gradient uses -i m R_m -> inner product conj(ȳ_m) * (-i m R_m)
                # R_m = e^{-i m α} c_m
                # We need c_m; recompute below after d multiplication
            end
            # b̄ = d^T(β) c̄
            SHTnsKit._wigner_d_advance!(dbuf, dwork, l, β)
            dl = view(dbuf, 1:n, 1:n)
            bbar = zeros(eltype(Zε), n)
            for mp in -l:l
                s = zero(eltype(Zε))
                for m in -l:l
                    s += dl[m + l + 1, mp + l + 1] * cbar[m + l + 1]
                end
                bbar[mp + l + 1] = s
            end
            # compute forward intermediates for angle grads
            b = zeros(eltype(Zε), n)
            for mp in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, mp) + 1
                b[mp + l + 1] = Zε[idx] * cis(-mp * γ)
            end
            c = dl * b
            # α-grad: sum_m conj(ȳ_m) * (-i m) R_m = real(sum conj(ȳ_m) * (-i m) e^{-i m α} c_m )
            for m in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, m) + 1
                Rm = c[m + l + 1] * cis(-m * α)
                gα += real(conj(ȳε[idx]) * ((0 - 1im) * m * Rm))
            end
            # γ-grad: sum_m conj(ȳ_m) * phaseL * d * (-i m') b_{m'}
            gγ_l = 0.0
            for m in -mm:mm
                idxm = LM_cplx_index(lmax, mmax, l, m) + 1
                s = zero(eltype(Zε))
                for mp in -l:l
                    s += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
                end
                gγ_l += real(conj(ȳε[idxm]) * (s * cis(-m * α)))
            end
            gγ += gγ_l
            # β-grad: use derivative d'(β)
            ddl = SHTnsKit._wigner_d_deriv!(ddbuf, dl, l)
            gβ_l = 0.0
            for m in -mm:mm
                idxm = LM_cplx_index(lmax, mmax, l, m) + 1
                s = zero(eltype(Zε))
                for mp in -l:l
                    s += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
                end
                gβ_l += real(conj(ȳε[idxm]) * (s * cis(-m * α)))
            end
            gβ += gβ_l
            # Ā_m' = e^{+i m' γ} b̄_m'
            for mp in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, mp) + 1
                Z̄[idx] += bbar[mp + l + 1] * cis(mp * γ)
            end
        end
        Z̄ .*= ε .* scales   # back to the configured packed layout
        rt = _rotation_rrule_tangent(rotation, gα, gβ, gγ)
        return NoTangent(), rt, Z̄, ZeroTangent()
    end
    return y, pullback
end

# Adjoint for real packed rotation. The coefficient adjoint is the shared exact
# one (SHTnsKit._rotation_apply_real_adjoint); the angle gradients differentiate
# the canonical engine and are returned in the stored parameter order.
function ChainRulesCore.rrule(::typeof(SHTnsKit.shtns_rotation_apply_real), r::SHTnsKit.SHTRotation, Qlm, Rlm)
    rotation = deepcopy(r)
    lmax, mmax = rotation.lmax, rotation.mmax
    α, β, γ = _rotation_rrule_angles(rotation)
    scales = _rotation_rrule_scales(rotation, Qlm; real_packed=true)
    Qcanonical = scales .* Qlm
    y = SHTnsKit.shtns_rotation_apply_real(r, Qlm, Rlm)
    function pullback(ȳ)
        ȳ = _unthunk(ȳ)
        ȳ isa ChainRulesCore.AbstractZero &&
            return NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent()
        Q̄ = similar(Qlm)
        copyto!(Q̄, SHTnsKit._rotation_apply_real_adjoint(rotation, ȳ))
        ȳcanonical = ȳ ./ scales
        CT = promote_type(eltype(Qcanonical), ComplexF64)
        # Angle gradients (pack-domain contribution using m≥0 only)
        gα = 0.0; gβ = 0.0; gγ = 0.0
        dbuf = Matrix{typeof(β)}(undef, 2lmax + 1, 2lmax + 1)
        dwork = similar(dbuf); ddbuf = similar(dbuf)
        for l in 0:lmax
            mm = min(l, mmax)
            SHTnsKit._wigner_d_advance!(dbuf, dwork, l, β)
            dl = view(dbuf, 1:(2l + 1), 1:(2l + 1))
            ddl = SHTnsKit._wigner_d_deriv!(ddbuf, dl, l)  # depends only on (l,β); hoisted out of the m-loop below
            b = zeros(CT, 2l + 1)
            for mp in -mm:mm
                # Reconstruct the Wigner-basis input from packed Qlm.
                b[mp + l + 1] = mp >= 0 ?
                    Qcanonical[SHTnsKit.LM_index(lmax, 1, l, mp) + 1] :
                    (-1)^(-mp) * conj(Qcanonical[SHTnsKit.LM_index(lmax, 1, l, -mp) + 1])
                b[mp + l + 1] *= cis(-mp * γ)
            end
            c = dl * b
            for m in 0:mm
                idxp = SHTnsKit.LM_index(lmax, 1, l, m) + 1
                Rm = c[m + l + 1] * cis(-m * α)
                gα += real(conj(ȳcanonical[idxp]) * ((0 - 1im) * m * Rm))
                sβ = zero(CT)
                sγ = zero(CT)
                for mp in -l:l
                    sβ += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
                    sγ += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
                end
                gβ += real(conj(ȳcanonical[idxp]) * (sβ * cis(-m * α)))
                gγ += real(conj(ȳcanonical[idxp]) * (sγ * cis(-m * α)))
            end
        end
        rt = _rotation_rrule_tangent(rotation, gα, gβ, gγ)
        return NoTangent(), rt, Q̄, ZeroTangent()
    end
    return y, pullback
end

# Operator application: SH_mul_mx(cfg, mx, Qlm, Rlm)
# Forward: R[lm0] = mx[2*lm_prev+2]*Q[lm_prev] + mx[2*lm_next+1]*Q[lm_next]
# where lm_prev = LM_index(l-1,m) and lm_next = LM_index(l+1,m)
function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_mul_mx), cfg::SHTnsKit.SHTConfig, mx, Qlm, Rlm)
    # The pullback needs the primal operands, but callers routinely reuse the
    # buffers before it runs (`SH_mul_mx(cfg, mx, R, Q)` overwrites Q). The
    # primal itself still sees the caller's arrays, so its alias check applies.
    mx0 = copy(mx)
    Q0 = copy(Qlm)
    y = SHTnsKit.SH_mul_mx(cfg, mx, Qlm, Rlm)
    function pullback(ȳ)
        ȳ = _unthunk(ȳ)
        ȳ isa ChainRulesCore.AbstractZero &&
            return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent()
        lmax = cfg.lmax; mres = cfg.mres
        Q̄ = zeros(eltype(Q0), length(Q0))
        mx̄ = zeros(eltype(mx0), length(mx0))
        @inbounds for lm0 in 0:(cfg.nlm-1)
            l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
            rbar = ȳ[lm0 + 1]
            # Contribution from lower neighbor Y_{l-1}^m (uses mx0[2*lm_prev + 2])
            if l > m && l > 0
                lm_prev = LM_index(lmax, mres, l-1, m)
                c_from_below = mx0[2*lm_prev + 2]  # b_{l-1}^m coefficient
                Q̄[lm_prev + 1] += c_from_below * rbar  # mx is real, no conj needed
                mx̄[2*lm_prev + 2] += real(conj(rbar) * Q0[lm_prev + 1])
            end
            # Contribution from upper neighbor Y_{l+1}^m (uses mx0[2*lm_next + 1])
            if l < lmax
                lm_next = LM_index(lmax, mres, l+1, m)
                c_from_above = mx0[2*lm_next + 1]  # a_{l+1}^m coefficient
                Q̄[lm_next + 1] += c_from_above * rbar  # mx is real, no conj needed
                mx̄[2*lm_next + 1] += real(conj(rbar) * Q0[lm_next + 1])
            end
        end
        return NoTangent(), NoTangent(), mx̄, Q̄, ZeroTangent()
    end
    return y, pullback
end

end # module
