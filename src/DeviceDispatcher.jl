module DeviceDispatcher

# our only strong dependency
using Adapt

# public API
export dev,
       use_gpu!, use_cpu!, use_device!,
       current_device,
       @gpu, @cpu


########################################################################
##                        MODULE-INTERNAL TOOLS                       ##
########################################################################

# thread-local device flag
const _device = fill(:cpu, Threads.nthreads())  ::Vector{Symbol}

@inline current_device() = @inbounds _device[Threads.threadid()]

@inline function _set_device!(d::Symbol)
  @inbounds _device[Threads.threadid()] = d
  return nothing
end

const _backend_mod  = Ref{Union{Module,Nothing}}(nothing)
const _gpu_eltype   = Ref{Any}(nothing)
const _to_gpu_fun   = Ref{Function}(x -> error("No GPU backend loaded."))

# a helper to make the backend module name display more nicely
Base.show(io::IO, r::typeof(_backend_mod)) = show(io, r[])

"""
    _register_backend!(backend_module, array_type)

Called by an extension module (e.g. `CUDA`) to tell DeviceDispatcher how
to move data to that device.
"""
function _register_backend!(mod::Module, arrtype)
  _backend_mod[] = mod
  _gpu_eltype[]  = arrtype
  _to_gpu_fun[]  = x -> Adapt.adapt(arrtype, x)
  @info "DeviceDispatcher bound to $(mod)"
  return
end

@inline _to_gpu(x) = (_to_gpu_fun[])(x)


########################################################################
##                            MODE SWITCHES                           ##
########################################################################

"""
    use_gpu!()

Set the *current* device to `:gpu` (error if no GPU backends are present).
Remains in force until you call `use_cpu!()` or `use_device!(:cpu)`.
"""
function use_gpu!()
  # probe: try to convert a tiny array with the currently loaded
  # backend(s); if none succeed, emit a helpful error
  try
    _to_gpu([0.0f0]) # 1-element Float32; nearly free
  catch err
    error("""
          GPU requested but no GPU backend appears to be loaded.
          Install CUDA.jl, AMDGPU.jl, Metal.jl or oneAPI.jl and ensure
          the hardware/driver is functional. Original error:

          $(err)
          """)
  end
  _set_device!(:gpu)
  return nothing
end


"""
    use_cpu!()

Set the *current* device to `:cpu` (the default).
"""
use_cpu!() = (_set_device!(:cpu); nothing)


"""
    use_device!()

Set the *current* device to `:gpu` or `:cpu`.
"""
use_device!(d::Symbol) = d === :gpu ? use_gpu!() :
                         d === :cpu ? use_cpu!() :
                         error("Unknown device $d.  Use :cpu or :gpu")


########################################################################
##                      TYPE CONVERSION FUNCTIONS                     ##
########################################################################

"""
    dev(x)

Adapt `x` to the **current** device:

* When device == `:gpu` -> `gpu(x)` (vendor-specific array returned).
* When device == `:cpu` -> `Adapt.adapt(Array, x)` (host memory).
* Scalars and already-device arrays pass through unchanged.

Recurses through containers via Adapt.jl.
"""
dev(x) = current_device() === :gpu ?
         _to_gpu(x) :                # backend-agnostic copy to GPU
         Adapt.adapt(Array, x)


########################################################################
##                          CONVENIENCE MACROS                        ##
########################################################################

"""
    @gpu expr
    @gpu begin ... end

Temporarily switch the calling thread to **GPU mode** for the duration
of `expr`.

If no GPU back-end is available---e.g. CUDA.jl is not in the current
environment---`use_gpu!()` throws an informative error.
"""
macro gpu(block)
  quote
    DeviceDispatcher.use_gpu!()
    try
      $(esc(block))
    finally
      # leave things as we found them
      DeviceDispatcher.use_cpu!()
    end
  end
end

"""
    @cpu expr
    @cpu begin ... end

Temporarily switch the calling thread to **CPU mode** for the duration
of `expr`.
"""
macro cpu(block)
  quote
    DeviceDispatcher.use_cpu!()
    $(esc(block))
  end
end


end # module
