module DeviceDispatcherOneAPIExt
using DeviceDispatcher
using oneAPI, Adapt

function __init__()
  # register if (and only if) oneAPI.jl is loadable in this environment
  DeviceDispatcher._register_backend!(oneAPI, oneAPI.OneArray)
end

end
