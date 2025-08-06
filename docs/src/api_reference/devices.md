```@meta
CurrentModule = ParallelKDE
```

# [Devices Interface](@id APIDevices)

The devices interface lets you specify where computations should run for a specific object.

```@docs
AbstractDevice
IsCPU
IsCUDA
DeviceNotSpecified
```

Any object that is meant to run on a specific device should implement the `get_device` method to make use of this interface.

```@docs
get_device(::Any)
```

Additionally, a device may have one or more methods that it has available for use as well default methods when the user does not specify one. Furthermore, a `Symbol` may be used to identify a device type. These methods are registered in dictionaries in the `Devices.jl` module.

```@docs
AVAILABLE_DEVICES
DEVICE_IMPLEMENTATIONS
DEFAULT_IMPLEMENTATIONS
```

This dictionaries can be used to query the validity of a selected method for a device.

```@docs
ensure_valid_implementation
```

Since GPU devices may work more efficiently with single-precision numbers, the package provides a method to convert double precision numbers to single precision when possible.

```@docs
convert32
```
