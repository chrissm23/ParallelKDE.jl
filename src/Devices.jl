module Devices

using CUDA

export
  AbstractDevice,
  get_device,
  IsCPU,
  IsCUDA,
  DeviceNotSpecified,
  AVAILABLE_DEVICES,
  obtain_device,
  get_available_memory,
  CPU_SERIAL,
  CPU_THREADED,
  GPU_CUDA,
  ensure_valid_implementation

abstract type AbstractDevice end

struct IsCPU <: AbstractDevice end
struct IsCUDA <: AbstractDevice end
struct DeviceNotSpecified <: AbstractDevice end

const AVAILABLE_DEVICES = Dict{Symbol,AbstractDevice}(
  :cpu => IsCPU(),
  :cuda => IsCUDA(),
)

get_device(::Any) = DeviceNotSpecified()

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

const CPU_SERIAL = :serial
const CPU_THREADED = :threaded
const GPU_CUDA = :cuda

const DEVICE_IMPLEMENTATIONS = Dict{AbstractDevice,Set{Symbol}}(
  IsCPU() => Set([CPU_SERIAL, CPU_THREADED]),
  IsCUDA() => Set([GPU_CUDA]),
  DeviceNotSpecified() => Set()
)

is_valid_implementation(device::AbstractDevice, implementation::Symbol) =
  implementation in get(DEVICE_IMPLEMENTATIONS, device, Set())

function ensure_valid_implementation(device::AbstractDevice, implementation::Symbol)
  if !is_valid_implementation(device, implementation)
    throw(ArgumentError("Invalid implementation $implementation for device $device"))
  end

  return true
end
ensure_valid_implementation(device::Symbol, implementation::Symbol) =
  ensure_valid_implementation(obtain_device(device), implementation)

end
