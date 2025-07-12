module DeviceDispatcherMetalExt
using DeviceDispatcher
using Metal, Adapt

function __init__()
  # register if (and only if) Metal.jl is loadable in this environment
  DeviceDispatcher._register_backend!(Metal, Metal.MtlArray)
end

end
