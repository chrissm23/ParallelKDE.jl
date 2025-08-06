module Devices

using CUDA

export
  AbstractDevice,
  get_device,
  IsCPU,
  IsCUDA,
  DeviceNotSpecified,
  AVAILABLE_DEVICES,
  CPU_SERIAL,
  CPU_THREADED,
  GPU_CUDA,
  DEVICE_IMPLEMENTATIONS,
  DEFAULT_IMPLEMENTATIONS,
  get_available_memory,
  ensure_valid_implementation,
  convert32

"""
    AbstractDevice

Supertype for all device types.
"""
abstract type AbstractDevice end

"""
    IsCPU <: AbstractDevice

Type representing a CPU device.
"""
struct IsCPU <: AbstractDevice end

"""
    IsCUDA <: AbstractDevice

Type representing a CUDA-enabled GPU device.
"""
struct IsCUDA <: AbstractDevice end

"""
    DeviceNotSpecified <: AbstractDevice

Type representing a device that has not been specified or is unsupported.
"""
struct DeviceNotSpecified <: AbstractDevice end

"""
    AVAILABLE_DEVICES

Relate a `Symbol` to the corresponding device type.
"""
const AVAILABLE_DEVICES = Dict(
  :cpu => IsCPU(),
  :cuda => IsCUDA(),
)

const CPU_SERIAL = :serial
const CPU_THREADED = :threaded
const GPU_CUDA = :cuda

"""
    DEVICE_IMPLEMENTATIONS

Relate the device types to their available implementations.
"""
const DEVICE_IMPLEMENTATIONS = Dict(
  IsCPU() => Set([CPU_SERIAL, CPU_THREADED]),
  IsCUDA() => Set([GPU_CUDA]),
  DeviceNotSpecified() => Set()
)

"""
    DEFAULT_IMPLEMENTATIONS

Relate the device types to their default implementations.
"""
const DEFAULT_IMPLEMENTATIONS = Dict(
  IsCPU() => CPU_SERIAL,
  IsCUDA() => GPU_CUDA,
)

"""
    get_device(device::Any)

Obtain the device object for a given device type.

If the method is called with an unsupported type, it returns a `DeviceNotSpecified` object.
"""
get_device(::Any) = DeviceNotSpecified()
get_device(::IsCPU) = IsCPU()
get_device(::IsCUDA) = IsCUDA()

function get_device(device::Symbol)
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

"""
    is_valid_implementation(device::AbstractDevice, implementation::Symbol)

Check if the given implementation is valid for the specified device.
"""
is_valid_implementation(device::AbstractDevice, implementation::Symbol) =
  implementation in get(DEVICE_IMPLEMENTATIONS, device, Set())

"""
    ensure_valid_implementation(device::AbstractDevice, implementation::Symbol)

Ensure that the specified implementation is valid for the given device.

If the implementation is not valid, an `ArgumentError` is thrown.
"""
function ensure_valid_implementation(device::AbstractDevice, implementation::Symbol)
  if !is_valid_implementation(device, implementation)
    throw(ArgumentError("Invalid implementation $implementation for device $device"))
  end

  return true
end
ensure_valid_implementation(device::Symbol, implementation::Symbol) =
  ensure_valid_implementation(get_device(device), implementation)

"""
    convert32(x)

Converts into a 32-bit representation if `x` is a `Float64`, `CuArray{Float64}`, or `Array{Float64}`.
"""
function convert32(x)
  if x isa Float64
    return Float32(x)
  elseif x isa CuArray{Float64}
    return CuArray{Float32}(x)
  elseif x isa Array{Float64}
    return Array{Float32}(x)
  else
    return x
  end
end
"""
    convert32(args...; kwargs...)

If multiple arguments are provided, converts each argument to a 32-bit representation if applicable,
and returns a tuple of the converted arguments and a dictionary of converted keyword arguments.
"""
function convert32(args...; kwargs...)
  kwargs_32 = (; (k => convert32(v) for (k, v) in kwargs)...)
  args_32 = (convert32(arg) for arg in args)

  return (args_32, kwargs_32)
end

end
