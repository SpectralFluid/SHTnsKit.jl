module SHTnsKitAdvancedADExt

using ChainRulesCore
using SHTnsKit
using SHTnsKit: LM_index, LM_cplx_index, wigner_d_matrix
import SHTnsKit: wigner_d_matrix_deriv

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


    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis), cfg::SHTnsKit.SHTConfig, f)
        y = SHTnsKit.analysis(cfg, f)
        function pullback(ȳ)
            ȳA = _to_complex(ȳ)
            f̄ = _adjoint_analysis(cfg, ȳA)
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
                                alm; real_output::Bool=true)
        y = SHTnsKit.synthesis(cfg, alm; real_output)
        function pullback(ȳ)
            ȳ_mat = ChainRulesCore.unthunk(ȳ)      # materialize Thunk/InplaceableThunk
            ȳA = ȳ_mat isa AbstractMatrix ? ȳ_mat : collect(ȳ_mat)
            alm̄ = SHTnsKit._adjoint_synthesis(cfg, ȳA; real_output=real_output)
            return NoTangent(), NoTangent(), alm̄, (; real_output=NoTangent())
        end
        return y, pullback
    end

    # Batch scalar transforms: analysis_batch, synthesis_batch
    # Each field is an independent scalar transform; adjoint applies the scalar
    # adjoint per slice. Simple and correct; not allocation-optimal (doesn't
    # share Legendre work across fields in the backward pass).
    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis_batch), cfg::SHTnsKit.SHTConfig, fields::AbstractArray{<:Real,3})
        y = SHTnsKit.analysis_batch(cfg, fields)
        function pullback(ȳ)
            ȳ = _unthunk(ȳ)
            ȳA = eltype(ȳ) <: Complex ? ȳ : complex.(ȳ)
            nfields = size(ȳA, 3)
            f̄ = Array{real(float(eltype(ȳA))),3}(undef, cfg.nlat, cfg.nlon, nfields)
            @inbounds for k in 1:nfields
                f̄[:, :, k] .= _adjoint_analysis(cfg, @view ȳA[:, :, k])
            end
            return NoTangent(), NoTangent(), f̄
        end
        return y, pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_batch), cfg::SHTnsKit.SHTConfig, alm_batch::AbstractArray{<:Complex,3}; real_output::Bool=true)
        y = SHTnsKit.synthesis_batch(cfg, alm_batch; real_output)
        function pullback(ȳ)
            ȳ = _unthunk(ȳ)
            nfields = size(ȳ, 3)
            ālm = zeros(complex(float(eltype(ȳ))), cfg.lmax + 1, cfg.mmax + 1, nfields)
            @inbounds for k in 1:nfields
                # Adjoint of synthesis is `_adjoint_synthesis` (NO Gauss quadrature
                # weights), matching the non-batch synthesis rrule. Using `analysis`
                # here was wrong — off by the w_i·cphi factors (FD-checked).
                ālm[:, :, k] .= SHTnsKit._adjoint_synthesis(cfg, @view(ȳ[:, :, k]); real_output=real_output)
            end
            return NoTangent(), NoTangent(), ālm, (; real_output=NoTangent())
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
        function pullback(ȳ)
            Ā = _unpack_lm(cfg, _to_complex(ȳ))
            Vr̄ = vec(_adjoint_analysis(cfg, Ā))
            return NoTangent(), NoTangent(), Vr̄
        end
        return y, pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_packed), cfg::SHTnsKit.SHTConfig, Qlm)
        y = SHTnsKit.synthesis_packed(cfg, Qlm)
        function pullback(ȳ)
            f̄ = reshape(_unthunk(ȳ), cfg.nlat, cfg.nlon)
            Qlm̄ = _pack_lm(cfg, SHTnsKit._adjoint_synthesis(cfg, f̄; real_output=true))
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
    #   V̄t, V̄p = real(ifft_phi(F̄θ)), real(ifft_phi(F̄φ))
    # where φadj = nlon * scaleφ = 2π (same as scalar adjoint)
    # sphtor adjoint analysis now lives in SHTnsKit proper (src/sphtor_transforms.jl).
    # Keep local alias for any direct callers of the ext symbol.
    const _adjoint_analysis_sphtor = SHTnsKit._adjoint_analysis_sphtor

    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis_sphtor), cfg::SHTnsKit.SHTConfig, Vt, Vp)
        Slm, Tlm = SHTnsKit.analysis_sphtor(cfg, Vt, Vp)
        function pullback(ṠTl)
            Slm̄, Tlm̄ = ṠTl
            # analysis-like: the primal divides by M on the way out, so the
            # cotangent is divided before the internal-convention adjoint.
            # `_materialize_coeff` turns a ZeroTangent slot into explicit zeros.
            S̄ = _materialize_coeff(Slm̄, cfg)
            T̄ = _materialize_coeff(Tlm̄, cfg)
            V̄t, V̄p = _adjoint_analysis_sphtor(cfg, S̄, T̄)
            return NoTangent(), NoTangent(), V̄t, V̄p
        end
        return (Slm, Tlm), pullback
    end

    # synthesis_sphtor adjoint now lives in SHTnsKit proper (src/core_transforms.jl,
    # parametrized over θ_globals so the distributed AD ext reuses the same kernel).
    # Local alias kept for backward compat with any users touching this symbol.
    const _adjoint_synthesis_sphtor = SHTnsKit._adjoint_synthesis_sphtor

    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_sphtor), cfg::SHTnsKit.SHTConfig,
                                Slm, Tlm; real_output::Bool=true)
        Vt, Vp = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output)
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
            return NoTangent(), NoTangent(), S̄, T̄, (; real_output=NoTangent())
        end
        return (Vt, Vp), pullback
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
        scaleφ = cfg.cphi
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
            z̄ = _adjoint_analysis_packed_cplx(cfg, _to_complex(ȳ))
            return NoTangent(), NoTangent(), z̄
        end
        return y, pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_packed_cplx), cfg::SHTnsKit.SHTConfig, alm)
        y = SHTnsKit.synthesis_packed_cplx(cfg, alm)
        function pullback(ȳ)
            alm̄ = _adjoint_synthesis_packed_cplx(cfg, _to_complex(ȳ))
            return NoTangent(), NoTangent(), alm̄
        end
        return y, pullback
    end

# Rotations: adjoints via inverse/adjoint rotation.
# The packed real-field rotations map coefficient vectors whose m>0 entries carry
# DOUBLE weight in the physical (field) inner product. Zygote/ChainRules use the
# STANDARD packed inner product ⟨a,b⟩=Σ conj(a)b, so the correct adjoint of a
# rotation R is Q̄ = W·R⁻¹·(W⁻¹ ȳ) with W = diag(wm), wm = 2 for m>0. (For the
# diagonal Z-rotation W cancels.) All four Q̄ formulas below are FD-verified in
# test/serial/test_rotation_gradients.jl; the angle (dα) gradients were already
# correct and are unchanged.
_rot_wm(cfg) = Float64[cfg.mi[k] == 0 ? 1.0 : 2.0 for k in 1:cfg.nlm]

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Zrotate), cfg::SHTnsKit.SHTConfig, Qlm, alpha::Real, Rlm)
    y = SHTnsKit.SH_Zrotate(cfg, Qlm, alpha, Rlm)
    function pullback(ȳ)
        # Diagonal rotation Rlm = Qlm·e^{imα} ⇒ Q̄ = ȳ·e^{-imα} = SH_Zrotate(ȳ, -α).
        # (Was conj.(SH_Zrotate(ȳ,+α)) = conj(ȳ)·e^{-imα} — wrong for complex ȳ.)
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Zrotate(cfg, ȳ, -alpha, Q̄)
        # angle gradient: dR/dα = i m R
        dα = 0.0
        for m in 0:cfg.mmax
            (m % cfg.mres == 0) || continue
            for l in m:cfg.lmax
                lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                # R = Q * e^{i m α}
                Rval = Qlm[lm] * cis(m * alpha)
                dα += real(conj(ȳ[lm]) * ((0 + 1im) * m * Rval))
            end
        end
        return NoTangent(), NoTangent(), Q̄, dα, ZeroTangent()
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Yrotate), cfg::SHTnsKit.SHTConfig, Qlm, alpha::Real, Rlm)
    y = SHTnsKit.SH_Yrotate(cfg, Qlm, alpha, Rlm)
    function pullback(ȳ)
        # Q̄ = W·R(-α)·(W⁻¹ ȳ). Bare R(-α) is the field-inner-product adjoint and
        # was off by the wm weighting.
        wm = _rot_wm(cfg)
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Yrotate(cfg, ȳ ./ wm, -alpha, Q̄)
        Q̄ .*= wm
        # angle gradient via d/dβ of Wigner-d at β=alpha
        dα = 0.0
        lmax, mmax = cfg.lmax, cfg.mmax
        for l in 0:lmax
            mm = min(l, mmax)
            b = zeros(eltype(ȳ), 2l+1)
            # b = A because γ=0, A from packed Qlm
            for mp in -mm:mm
                idxp = LM_index(lmax, 1, l, abs(mp)) + 1
                # reconstruct complex A using hermitian symmetry for real field
                if mp == 0
                    b[mp + l + 1] = Qlm[idxp]
                elseif mp > 0
                    b[mp + l + 1] = Qlm[idxp]
                    b[-mp + l + 1] = (-1)^mp * conj(Qlm[idxp])
                end
            end
            dd = wigner_d_matrix_deriv(l, float(alpha))
            # ∂R_m = (dd * b)_m for m>=0 (no left/right phases)
            for m in 0:mm
                lm = LM_index(lmax, 1, l, m) + 1
                s = zero(eltype(ȳ))
                for mp in -l:l
                    s += dd[m + l + 1, mp + l + 1] * b[mp + l + 1]
                end
                dα += real(conj(ȳ[lm]) * s)
            end
        end
        return NoTangent(), NoTangent(), Q̄, dα, ZeroTangent()
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Yrotate90), cfg::SHTnsKit.SHTConfig, Qlm, Rlm)
    y = SHTnsKit.SH_Yrotate90(cfg, Qlm, Rlm)
    function pullback(ȳ)
        wm = _rot_wm(cfg)
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Yrotate(cfg, ȳ ./ wm, -π/2, Q̄)
        Q̄ .*= wm
        return NoTangent(), NoTangent(), Q̄, ZeroTangent()
    end
    return y, pullback
end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Xrotate90), cfg::SHTnsKit.SHTConfig, Qlm, Rlm)
        y = SHTnsKit.SH_Xrotate90(cfg, Qlm, Rlm)
        function pullback(ȳ)
            # Forward Xrotate90 is ZYZ(π/2, π/2, -π/2); its inverse is
            # ZYZ(-γ,-β,-α) = ZYZ(π/2, -π/2, -π/2) (the old (-π/2,-π/2,π/2) was
            # not the inverse). Plus the wm weighting for the standard adjoint.
            wm = _rot_wm(cfg)
            r = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax)
            SHTnsKit.shtns_rotation_set_angles_ZYZ(r, π/2, -π/2, -π/2)
            Q̄ = similar(Qlm)
            SHTnsKit.shtns_rotation_apply_real(r, ȳ ./ wm, Q̄)
            Q̄ .*= wm
            return NoTangent(), NoTangent(), Q̄, ZeroTangent()
        end
        return y, pullback
    end

# Adjoint for complex rotation using conjugate-transpose of Wigner-D
function ChainRulesCore.rrule(::typeof(SHTnsKit.shtns_rotation_apply_cplx), r::SHTnsKit.SHTRotation, Zlm, Rlm)
    y = SHTnsKit.shtns_rotation_apply_cplx(r, Zlm, Rlm)
    function pullback(ȳ)
        lmax, mmax = r.lmax, r.mmax
        Z̄ = similar(Zlm)
        fill!(Z̄, zero(eltype(Z̄)))
        α, β, γ = r.α, r.β, r.γ
        # This pullback reimplements the Wigner engine, which works in the Y_l^m
        # basis, while the primal is `ε ∘ engine ∘ ε` in the packed LM_cplx layout
        # (see SHTnsKit._lmcplx_ybasis_signs). Convert both the input and the
        # incoming cotangent into the engine's basis and convert the result back;
        # ε is real and self-inverse, so the angle gradients are unaffected.
        # NOTE: bind to NEW locals. `Zlm` is captured from the enclosing rrule, and
        # assigning to a captured variable inside a closure rebinds its box — so
        # `Zlm = ε .* Zlm` would apply ε again on every subsequent call, leaving
        # the angle gradients wrong (by O(1)) from the second invocation onward.
        # `ȳ` is the closure's own argument and would be safe, but is renamed too
        # so the pair reads the same way.
        ε = SHTnsKit._lmcplx_ybasis_signs(lmax, mmax)
        Zε = ε .* Zlm
        ȳε = ε .* ȳ
        gα = 0.0; gβ = 0.0; gγ = 0.0
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
            dl = wigner_d_matrix(l, β)
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
            ddl = wigner_d_matrix_deriv(l, β)
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
        Z̄ .*= ε   # back to the packed layout
        rt = Tangent{SHTnsKit.SHTRotation}(; α=gα, β=gβ, γ=gγ)
        return NoTangent(), rt, Z̄, ZeroTangent()
    end
    return y, pullback
end

# Adjoint for real packed rotation: extend to full, apply cplx adjoint, fold back
function ChainRulesCore.rrule(::typeof(SHTnsKit.shtns_rotation_apply_real), r::SHTnsKit.SHTRotation, Qlm, Rlm)
    y = SHTnsKit.shtns_rotation_apply_real(r, Qlm, Rlm)
    function pullback(ȳ)
        lmax, mmax = r.lmax, r.mmax
        # Extend cotangent on packed to full complex
        Zbar_full = zeros(eltype(Qlm), SHTnsKit.nlm_cplx_calc(lmax, mmax, 1))
        for l in 0:lmax
            mm = min(l, mmax)
            # m = 0
            idxp0 = LM_index(lmax, 1, l, 0) + 1
            idxc0 = LM_cplx_index(lmax, mmax, l, 0) + 1
            Zbar_full[idxc0] = ȳ[idxp0]
            for m in 1:mm
                idxp = LM_index(lmax, 1, l, m) + 1
                idxc = LM_cplx_index(lmax, mmax, l, m) + 1
                Zbar_full[idxc] = ȳ[idxp]
                # negative m gets zero from packing adjoint
            end
        end
        # Compute adjoint of complex rotation (transpose of Wigner-d matrix)
        Z̄ = zeros(eltype(Zbar_full), length(Zbar_full))
        α, β, γ = r.α, r.β, r.γ
        for l in 0:lmax
            mm = min(l, mmax)
            n = 2l + 1
            cbar = zeros(eltype(Z̄), n)
            for m in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, m) + 1
                cbar[m + l + 1] = Zbar_full[idx] * cis(m * α)
            end
            dl = wigner_d_matrix(l, β)
            bbar = zeros(eltype(Z̄), n)
            for mp in -l:l
                s = zero(eltype(Z̄))
                for m in -l:l
                    s += dl[m + l + 1, mp + l + 1] * cbar[m + l + 1]
                end
                bbar[mp + l + 1] = s
            end
            for mp in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, mp) + 1
                Z̄[idx] += bbar[mp + l + 1] * cis(mp * γ)
            end
        end
        # Angle gradients (pack-domain contribution using m≥0 only)
        gα = 0.0; gβ = 0.0; gγ = 0.0
        for l in 0:lmax
            mm = min(l, mmax)
            dl = wigner_d_matrix(l, r.β)
            ddl = wigner_d_matrix_deriv(l, r.β)  # depends only on (l,β); hoisted out of the m-loop below
            b = zeros(eltype(Z̄), 2l + 1)
            for mp in -mm:mm
                idx = SHTnsKit.LM_cplx_index(lmax, mmax, l, mp) + 1
                b[mp + l + 1] = (SHTnsKit.LM_cplx_index(lmax, mmax, l, mp) >= 0) ? (begin
                    # reconstruct from packed Qlm
                    if mp == 0
                        Qlm[SHTnsKit.LM_index(lmax, 1, l, 0) + 1]
                    elseif mp > 0
                        Qlm[SHTnsKit.LM_index(lmax, 1, l, mp) + 1]
                    else
                        (-1)^(-mp) * conj(Qlm[SHTnsKit.LM_index(lmax, 1, l, -mp) + 1])
                    end
                end) : 0
                b[mp + l + 1] *= cis(-mp * r.γ)
            end
            c = dl * b
            for m in 0:mm
                idxp = SHTnsKit.LM_index(lmax, 1, l, m) + 1
                Rm = c[m + l + 1] * cis(-m * r.α)
                gα += real(conj(ȳ[idxp]) * ((0 - 1im) * m * Rm))
                # β
                sβ = zero(eltype(Z̄))
                sγ = zero(eltype(Z̄))
                for mp in -l:l
                    sβ += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
                    sγ += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
                end
                gβ += real(conj(ȳ[idxp]) * (sβ * cis(-m * r.α)))
                gγ += real(conj(ȳ[idxp]) * (sγ * cis(-m * r.α)))
            end
        end
        # Fold back to packed positive-m: q̄(m) = Z̄(m) + (-1)^m conj(Z̄(-m))
        Q̄ = zeros(eltype(Qlm), length(Qlm))
        for l in 0:lmax
            mm = min(l, mmax)
            # m=0
            idxp0 = LM_index(lmax, 1, l, 0) + 1
            idxc0 = LM_cplx_index(lmax, mmax, l, 0) + 1
            Q̄[idxp0] = Z̄[idxc0]
            for m in 1:mm
                idxp = LM_index(lmax, 1, l, m) + 1
                idxc_p = LM_cplx_index(lmax, mmax, l, m) + 1
                idxc_n = LM_cplx_index(lmax, mmax, l, -m) + 1
                Q̄[idxp] = Z̄[idxc_p] + (-1)^m * conj(Z̄[idxc_n])
            end
        end
        rt = Tangent{SHTnsKit.SHTRotation}(; α=gα, β=gβ, γ=gγ)
        return NoTangent(), rt, Q̄, ZeroTangent()
    end
    return y, pullback
end

# Operator application: SH_mul_mx(cfg, mx, Qlm, Rlm)
# Forward: R[lm0] = mx[2*lm_prev+2]*Q[lm_prev] + mx[2*lm_next+1]*Q[lm_next]
# where lm_prev = LM_index(l-1,m) and lm_next = LM_index(l+1,m)
function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_mul_mx), cfg::SHTnsKit.SHTConfig, mx, Qlm, Rlm)
    y = SHTnsKit.SH_mul_mx(cfg, mx, Qlm, Rlm)
    function pullback(ȳ)
        lmax = cfg.lmax; mres = cfg.mres
        Q̄ = zeros(eltype(Qlm), length(Qlm))
        mx̄ = zeros(eltype(mx), length(mx))
        @inbounds for lm0 in 0:(cfg.nlm-1)
            l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
            rbar = ȳ[lm0 + 1]
            # Contribution from lower neighbor Y_{l-1}^m (uses mx[2*lm_prev + 2])
            if l > m && l > 0
                lm_prev = LM_index(lmax, mres, l-1, m)
                c_from_below = mx[2*lm_prev + 2]  # b_{l-1}^m coefficient
                Q̄[lm_prev + 1] += c_from_below * rbar  # mx is real, no conj needed
                mx̄[2*lm_prev + 2] += real(conj(rbar) * Qlm[lm_prev + 1])
            end
            # Contribution from upper neighbor Y_{l+1}^m (uses mx[2*lm_next + 1])
            if l < lmax
                lm_next = LM_index(lmax, mres, l+1, m)
                c_from_above = mx[2*lm_next + 1]  # a_{l+1}^m coefficient
                Q̄[lm_next + 1] += c_from_above * rbar  # mx is real, no conj needed
                mx̄[2*lm_next + 1] += real(conj(rbar) * Qlm[lm_next + 1])
            end
        end
        return NoTangent(), NoTangent(), mx̄, Q̄, ZeroTangent()
    end
    return y, pullback
end

end # module
