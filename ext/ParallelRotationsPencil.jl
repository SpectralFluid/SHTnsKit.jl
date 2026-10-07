##########
# PencilArray rotations
##########

const _ROTATION_STATS_LOCK = ReentrantLock()
const _ROTATION_STATS = Dict{Symbol,Int}(
    :z_payload_sent_elements => 0,
    :general_payload_sent_elements => 0,
    :general_max_message_elements => 0,
)

function _reset_rotation_stats!()
    lock(_ROTATION_STATS_LOCK) do
        for key in keys(_ROTATION_STATS)
            _ROTATION_STATS[key] = 0
        end
    end
    return nothing
end

function _rotation_stats()
    return lock(_ROTATION_STATS_LOCK) do
        return (
            z_payload_sent_elements=_ROTATION_STATS[:z_payload_sent_elements],
            general_payload_sent_elements=_ROTATION_STATS[:general_payload_sent_elements],
            general_max_message_elements=_ROTATION_STATS[:general_max_message_elements],
        )
    end
end

function _record_rotation_payload!(sent::Int, maximum::Int)
    lock(_ROTATION_STATS_LOCK) do
        _ROTATION_STATS[:general_payload_sent_elements] += sent
        _ROTATION_STATS[:general_max_message_elements] = max(
            _ROTATION_STATS[:general_max_message_elements], maximum,
        )
    end
    return nothing
end

"""
An angle as Float64, or NaN when it is not a real number. Every `Real` is
accepted (`π`, integers, rationals), as in the serial rotations; checking the
value instead of the type also lets ranks pass `0.5f0` and `0.5` alike.
"""
_rotation_angle_value(angle::Real) = try
    Float64(angle)
catch
    NaN
end
_rotation_angle_value(angle) = NaN

"""Collectively require each angle to be finite and equal on every rank."""
function _validate_rotation_angles!(comm, angles::Tuple, operation::Symbol)
    flags = UInt32(0)
    for angle in angles
        value = _rotation_angle_value(angle)
        isfinite(value) || (flags |= 0x8000)
        reference = Ref(value)
        MPI.Bcast!(reference, 0, comm)
        isequal(value, reference[]) || (flags |= 0x8000)
    end
    _collective_validation_error(comm, flags, operation)
    return nothing
end

function _validate_rotation_pencils!(cfg, input::PencilArray,
                                     output::PencilArray, angles::Tuple,
                                     operation::Symbol; general::Bool,
                                     mixing_beta=nothing)
    # The first input is the communicator root of trust. Every rank in that
    # communicator must enter with a compatible input; candidate outputs are
    # then preflighted collectively before any mutation or data movement.
    comm = communicator(input)
    _validate_qst_pencil_communicators!(comm, (input, output), operation)
    _validate_cfg_replicated(cfg, comm)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    _validate_scalar_pencil!(
        cfg, input, expected, operation; comm, peer=output,
        require_full_first_dim=true, required_decomposition=(2,),
        require_complex_input=true,
    )
    _validate_scalar_pencil!(
        cfg, output, expected, operation; comm, peer=input,
        require_full_first_dim=true, required_decomposition=(2,),
        require_complex_input=true,
    )
    _validate_identical_pencil_layout!(input, output, operation; comm)
    flags = eltype(input) === eltype(output) ? UInt32(0) : UInt32(0x0004)
    general && cfg.mres != 1 && (flags |= 0x10000)
    # As in the serial engine: orders above mmax would be silently dropped.
    mixing_beta === nothing ||
        !SHTnsKit._rotation_exceeds_orders(cfg.lmax, cfg.mmax, mixing_beta) ||
        (flags |= 0x80000)
    _collective_validation_error(comm, flags, operation)
    _validate_rotation_angles!(comm, angles, operation)
    return comm
end

function _dist_zrotate_local!(cfg, input::PencilArray, angle::Real,
                              output::PencilArray)
    source = parent(input); destination = parent(output)
    orders = collect(Int, globalindices(input, 2))
    RT = typeof(real(zero(eltype(input))))
    @inbounds for (local_m, m_index) in pairs(orders)
        m = m_index - 1
        phase = cis(-RT(m) * RT(angle))
        for local_l in axes(source, 1)
            destination[local_l, local_m] = phase * source[local_l, local_m]
        end
    end
    return output
end

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig,
                                  input::PencilArray, angle::Real)
    output = similar(input)
    return SHTnsKit.dist_SH_Zrotate(cfg, input, angle, output)
end

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig,
                                  input::PencilArray, angle::Real,
                                  output::PencilArray)
    _validate_rotation_pencils!(
        cfg, input, output, (angle,), :dist_SH_Zrotate; general=false,
    )
    return _dist_zrotate_local!(cfg, input, angle, output)
end

function _dist_yrotate_rows!(cfg, input::PencilArray, beta::Real,
                             output::PencilArray, comm)
    RT = typeof(real(zero(eltype(input))))
    CT = eltype(input)
    source = parent(input); destination = parent(output)
    l_indices = collect(Int, globalindices(input, 1))
    m_indices = collect(Int, globalindices(input, 2))
    counts_m = collect(Int, MPI.Allgather(length(m_indices), comm))
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    counts_l = zeros(Int, nranks)
    local_buffer = Vector{CT}(undef, length(m_indices))
    full = Vector{CT}(undef, cfg.mmax + 1)
    b = Vector{CT}(undef, 2cfg.lmax + 1)
    c = similar(b)
    d = Matrix{RT}(undef, 2cfg.lmax + 1, 2cfg.lmax + 1)
    dwork = similar(d)
    previous_l = -2
    local_sent = 0
    local_maximum = 0

    for (local_l, l_index) in pairs(l_indices)
        l = l_index - 1
        mm = min(l, cfg.mmax)
        offset = 0
        @inbounds for r in 1:nranks
            counts_l[r] = max(0, min(offset + counts_m[r], mm + 1) - offset)
            offset += counts_m[r]
        end
        count_local = counts_l[rank + 1]
        @inbounds for k in 1:count_local
            local_buffer[k] = source[local_l, k]
        end
        MPI.Allgatherv!(
            view(local_buffer, 1:count_local),
            MPI.VBuffer(view(full, 1:(mm + 1)), counts_l), comm,
        )
        local_sent += count_local
        local_maximum = max(local_maximum, count_local)

        n = 2l + 1
        fill!(view(b, 1:n), zero(CT))
        @inbounds for m in 0:mm
            scale = RT(SHTnsKit.coefficient_scale_to_canonical(cfg, l, m))
            canonical = scale * full[m + 1]
            b[m + l + 1] = canonical
            if m > 0
                b[-m + l + 1] = (isodd(m) ? -one(RT) : one(RT)) * conj(canonical)
            end
        end
        # A rank's degrees are consecutive: build the first one, then advance
        # (O(l²) per degree instead of rebuilding each from d⁰ at O(l³)).
        if l == previous_l + 1
            SHTnsKit._wigner_d_advance!(d, dwork, l, RT(beta))
        else
            SHTnsKit.wigner_d_matrix!(d, l, RT(beta), dwork)
        end
        previous_l = l
        block = view(d, 1:n, 1:n)
        @inbounds for m in -l:l
            acc = zero(CT)
            for mp in -l:l
                acc += block[m + l + 1, mp + l + 1] * b[mp + l + 1]
            end
            c[m + l + 1] = acc
        end
        @inbounds for (local_m, m_index) in pairs(m_indices)
            m = m_index - 1
            destination[local_l, local_m] = m <= l ?
                c[m + l + 1] /
                RT(SHTnsKit.coefficient_scale_to_canonical(cfg, l, m)) :
                zero(CT)
        end
    end
    sent = MPI.Allreduce(local_sent, +, comm)
    maximum = MPI.Allreduce(local_maximum, max, comm)
    _record_rotation_payload!(sent, maximum)
    return output
end

function _dist_yrotate!(cfg, input::PencilArray, beta::Real,
                        output::PencilArray, operation::Symbol)
    comm = _validate_rotation_pencils!(
        cfg, input, output, (beta,), operation; general=true, mixing_beta=beta,
    )
    return _dist_yrotate_rows!(cfg, input, beta, output, comm)
end

SHTnsKit.dist_SH_Yrotate_allgatherm!(cfg::SHTnsKit.SHTConfig,
                                     input::PencilArray, beta::Real,
                                     output::PencilArray) =
    _dist_yrotate!(cfg, input, beta, output, :dist_SH_Yrotate_allgatherm!)

SHTnsKit.dist_SH_Yrotate_truncgatherm!(cfg::SHTnsKit.SHTConfig,
                                       input::PencilArray, beta::Real,
                                       output::PencilArray) =
    _dist_yrotate!(cfg, input, beta, output, :dist_SH_Yrotate_truncgatherm!)

SHTnsKit.dist_SH_Yrotate(cfg::SHTnsKit.SHTConfig,
                         input::PencilArray, beta::Real,
                         output::PencilArray) =
    _dist_yrotate!(cfg, input, beta, output, :dist_SH_Yrotate)

SHTnsKit.dist_SH_Yrotate90(cfg::SHTnsKit.SHTConfig,
                           input::PencilArray, output::PencilArray) =
    SHTnsKit.dist_SH_Yrotate(cfg, input, pi / 2, output)

function SHTnsKit.dist_SH_rotate_euler(cfg::SHTnsKit.SHTConfig,
                                      input::PencilArray,
                                      alpha::Real, beta::Real, gamma::Real,
                                      output::PencilArray)
    comm = _validate_rotation_pencils!(
        cfg, input, output, (alpha, beta, gamma),
        :dist_SH_rotate_euler; general=true, mixing_beta=beta,
    )
    first = similar(input); second = similar(input)
    _dist_zrotate_local!(cfg, input, alpha, first)
    _dist_yrotate_rows!(cfg, first, beta, second, comm)
    return _dist_zrotate_local!(cfg, second, gamma, output)
end

SHTnsKit.dist_SH_Xrotate90(cfg::SHTnsKit.SHTConfig,
                           input::PencilArray, output::PencilArray) =
    SHTnsKit.dist_SH_rotate_euler(
        # `dist_SH_rotate_euler` applies the first Z factor first, matching
        # the intrinsic/setter call order used by the serial X90 helper.
        cfg, input, pi / 2, pi / 2, -pi / 2, output,
    )

function _validate_packed_rotation!(cfg, coefficients, angles, prototype,
                                    operation; general)
    # Packed coefficients have no communicator, so the prototype is their
    # trusted communicator anchor. All replication checks stay within it.
    comm = communicator(prototype)
    _validate_parallel_storage!(comm, operation, coefficients, prototype)
    _validate_cfg_replicated(cfg, comm)
    flags = length(coefficients) == cfg.nlm ? UInt32(0) : UInt32(0x0001)
    code = _scalar_precision_code(eltype(coefficients))
    code in (2, 4) || (flags |= 0x0004)
    MPI.Allreduce(code, min, comm) == MPI.Allreduce(code, max, comm) ||
        (flags |= 0x0004)
    general && cfg.mres != 1 && (flags |= 0x10000)
    _collective_validation_error(comm, flags, operation)
    _validate_rotation_angles!(comm, angles, operation)
    return nothing
end

function SHTnsKit.dist_SH_Zrotate_packed(cfg::SHTnsKit.SHTConfig,
                                         coefficients::AbstractVector{<:Complex},
                                         angle::Real;
                                         prototype_lm::PencilArray)
    _validate_packed_rotation!(
        cfg, coefficients, (angle,), prototype_lm,
        :dist_SH_Zrotate_packed; general=false,
    )
    output = similar(coefficients)
    return SHTnsKit.SH_Zrotate(SHTnsKit.CPU(), cfg, coefficients, angle, output)
end

function SHTnsKit.dist_SH_Yrotate_packed(cfg::SHTnsKit.SHTConfig,
                                         coefficients::AbstractVector{<:Complex},
                                         beta::Real;
                                         prototype_lm::PencilArray)
    _validate_packed_rotation!(
        cfg, coefficients, (beta,), prototype_lm,
        :dist_SH_Yrotate_packed; general=true,
    )
    output = similar(coefficients)
    return SHTnsKit.SH_Yrotate(SHTnsKit.CPU(), cfg, coefficients, beta, output)
end

SHTnsKit.dist_SH_Yrotate90_packed(cfg::SHTnsKit.SHTConfig,
                                  coefficients::AbstractVector{<:Complex};
                                  prototype_lm::PencilArray) =
    SHTnsKit.dist_SH_Yrotate_packed(
        cfg, coefficients, pi / 2; prototype_lm,
    )

function SHTnsKit.dist_SH_Xrotate90_packed(cfg::SHTnsKit.SHTConfig,
                                           coefficients::AbstractVector{<:Complex};
                                           prototype_lm::PencilArray)
    _validate_packed_rotation!(
        cfg, coefficients, (pi / 2,), prototype_lm,
        :dist_SH_Xrotate90_packed; general=true,
    )
    output = similar(coefficients)
    return SHTnsKit.SH_Xrotate90(
        SHTnsKit.CPU(), cfg, coefficients, output,
    )
end
