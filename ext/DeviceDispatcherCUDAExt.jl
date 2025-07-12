module DeviceDispatcherCUDAExt
using DeviceDispatcher
using CUDA, Adapt

function __init__()
  # register if (and only if) CUDA.jl is loadable in this environment
  DeviceDispatcher._register_backend!(CUDA, CUDA.CuArray)
end

end
