"""Module entry point so ``python -m squish ...`` mirrors the ``squish`` script.

Having a ``__main__`` makes the CLI invokable without depending on the
console-script shim being on ``PATH`` — used by the e2e harness and CI, which
launch the server via ``python -m squish serve``.
"""

from __future__ import annotations

from squish.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
