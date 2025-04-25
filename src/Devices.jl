module Devices

export Device, IsCPU, IsCUDA

abstract type Device end

struct IsCPU <: Device end
struct IsCUDA <: Device end
struct DeviceNotSpecified <: Device end

Device(::Any) = DeviceNotSpecified()

end
