module Devices

export Device, IsCPU, IsCUDA, DeviceNotSpecified, AVAILABLE_DEVICES

abstract type Device end

struct IsCPU <: Device end
struct IsCUDA <: Device end
struct DeviceNotSpecified <: Device end

const AVAILABLE_DEVICES = Dict{Symbol,Device}(
  :cpu => IsCPU(),
  :cuda => IsCUDA(),
)

Device(::Any) = DeviceNotSpecified()

end
