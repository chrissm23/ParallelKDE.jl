module Devices

export Device, IsCPU, IsCUDA

abstract type Device end

struct IsCPU <: Device end
struct IsCUDA <: Device end
struct NotSpecified <: Device end

Device(::Any) = NotSpecified()

end
