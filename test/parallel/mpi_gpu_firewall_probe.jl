# Dispatch probe for the compound MPI + GPU extension firewall.  Included by
# test/parallel/test_mpi_gpu_vendor_boundary.jl (mock vendor storage) and by
# test/parity/mpi_gpu.jl (CUDA and ROCm types); the including module must
# have loaded MPI, PencilArrays and SHTnsKit.

"""
    firewall_dispatch_probe(compound, device_array) -> (; probes, failures)

Probe every method that the compound extension module `compound` adds to
SHTnsKit and SHTnsKitParallelExt functions, using vendor-backed argument types
built from `device_array(T, N)`, and return the probes that do not dispatch to
the probed method itself. A method that cannot be probed is also a failure.
`Test.detect_ambiguities` misses the ambiguities that a vendor PencilArray
alias causes when it is not a subtype of `PencilArray`, so this checks
dispatch directly, for positional and keyword calls.
"""
function firewall_dispatch_probe(compound::Module, device_array)
    parallel = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    pencil_name = Base.typename(PencilArrays.PencilArray)
    host_pencil = typeof(PencilArrays.Pencil((4, 4), (1,), MPI.COMM_SELF))
    device_pencil = host_pencil.name.wrapper{
        host_pencil.parameters[1:end-1]..., device_array(UInt8, 1),
    }
    pencil_array(T, N, A, P) = PencilArrays.PencilArray{T,N,A,2,N - 2,P}
    function argument(parameter, T, N, dense, host_pencil_allowed)
        body = Base.unwrap_unionall(parameter)
        candidates = if body isa DataType && body.name === pencil_name
            device = pencil_array(T, N, device_array(T, N), device_pencil)
            host = pencil_array(T, N, Array{T,N}, host_pencil)
            host_pencil_allowed ? (host, device) : (device,)
        else
            (Vector{Float64}, Matrix{T}, device_array(T, dense),
             device_array(T, N), SHTnsKit.SHTConfig, Int, Float64,
             SHTnsKit.CPU, parallel.DistTransposePlan,
             parallel.DistributedSpectralPlan, parallel.DistributedSpectralPlan2D)
        end
        for candidate in candidates
            candidate <: parameter && return candidate
        end
        # e.g. a keyword body's `typeof(f)` parameter
        return isconcretetype(parameter) ? parameter : nothing
    end

    probes = 0
    failures = String[]
    seen = Set{Method}()
    for owner in (SHTnsKit, parallel), name in names(owner; all=true)
        isdefined(owner, name) || continue
        f = getfield(owner, name)
        f isa Function || continue
        for method in methods(f)
            method.module === compound && !(method in seen) || continue
            push!(seen, method)
            parameters = Base.unwrap_unionall(method.sig).parameters[2:end]
            if any(parameter -> parameter isa Core.TypeofVararg, parameters)
                push!(failures, "unprobed vararg: $method")
                continue
            end
            keyword = !isempty(Base.kwarg_decl(method))
            probed = false
            # Dense (non-Pencil) vendor arguments alternate between vectors
            # and matrices across the element types.
            for (T, N, dense) in ((Float64, 2, 2), (ComplexF64, 2, 1),
                                  (Float32, 3, 2), (ComplexF32, 3, 1)),
                mixed in (false, true)
                # The first Pencil argument is the vendor-backed one; in the
                # mixed probe any later Pencil argument is host-backed.
                pencils = 0
                arguments = Any[]
                for parameter in parameters
                    parameter = Base.rewrap_unionall(
                        parameter isa TypeVar ? parameter.ub : parameter,
                        method.sig,
                    )
                    chosen = argument(
                        parameter, T, N, dense, mixed && pencils > 0,
                    )
                    chosen === nothing && break
                    chosen <: PencilArrays.PencilArray && (pencils += 1)
                    push!(arguments, chosen)
                end
                length(arguments) == length(parameters) || continue
                probed = true
                for call in (keyword ? (:positional, :keyword) : (:positional,))
                    probes += 1
                    outcome = try
                        if call === :positional
                            target = which(f, Tuple{arguments...})
                            target === method ? :ok : target
                        else
                            target = which(Core.kwcall, Tuple{
                                NamedTuple{(:probe,),Tuple{Int}},
                                typeof(f), arguments...,
                            })
                            target.module === compound ? :ok : target
                        end
                    catch error
                        first(split(sprint(showerror, error), '\n'))
                    end
                    outcome === :ok || push!(failures,
                        "$(method.name) $(call) T=$T N=$N mixed=$mixed " *
                        "($(method.file):$(method.line)): $outcome")
                end
            end
            probed || push!(failures, "unprobed: $method")
        end
    end
    return (; probes, failures)
end
