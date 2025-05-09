module Devices

using CUDA

export
  AbstractDevice,
  Device,
  IsCPU,
  IsCUDA,
  DeviceNotSpecified,
  AVAILABLE_DEVICES,
  obtain_device,
  get_available_memory

abstract type AbstractDevice end

struct IsCPU <: AbstractDevice end
struct IsCUDA <: AbstractDevice end
struct DeviceNotSpecified <: AbstractDevice end

const AVAILABLE_DEVICES = Dict{Symbol,AbstractDevice}(
  :cpu => IsCPU(),
  :cuda => IsCUDA(),
)

Device(::Any) = DeviceNotSpecified()

function obtain_device(device::Symbol)
  if !haskey(AVAILABLE_DEVICES, device)
    throw(ArgumentError("Invalid device: $device"))
  end

  return AVAILABLE_DEVICES[device]
end

function get_available_memory(::IsCPU)
  return Sys.free_memory() / 1024^2
end
function get_available_memory(::IsCUDA)
  return CUDA.memory_info()[1] / 1024^2
end


end
