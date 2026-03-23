"""squish/loaders — model file loader plugins.

Loader plugins read Squish-native (``.squizd``) or third-party model files
and return the weight dictionaries and config structures expected by
the Squish inference engine.

Available loaders
─────────────────
* :mod:`squish.loaders.coreml_loader` — CoreML appendix block loader; reads
  the ANE_COREML appendix embedded in a ``.squizd`` file and returns a
  :class:`~squish.loaders.coreml_loader.CoreMLRuntime` for ANE inference.
"""
