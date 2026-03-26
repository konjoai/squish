# Compatibility shim — canonical implementation lives in squish.speculative.conf_spec.
import sys as _sys, importlib as _il
_sys.modules[__name__] = _il.import_module("squish.speculative.conf_spec")
