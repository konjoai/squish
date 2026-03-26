# Compatibility shim — canonical implementation lives in squish.serving.power_monitor.
import sys as _sys, importlib as _il
_sys.modules[__name__] = _il.import_module("squish.serving.power_monitor")
