module DeviceDispatcherAMDGPUExt
using DeviceDispatcher
using AMDGPU, Adapt

function __init__()
  # register if (and only if) AMDGPU.jl is loadable in this environment
  DeviceDispatcher._register_backend!(AMDGPU, AMDGPU.ROCArray)
end

end
