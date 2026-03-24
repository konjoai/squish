# [Experimental] This module is part of Squish v44+ (Wave 70).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""SquishRuntime — unified SQUIZD dispatch path.

Reads the format flags bitfield from a ``.squizd`` file header and selects
the correct kernel stack for each inference layer, activating automatically
by header inspection — no user flags required at serve time.

SQUIZD header flag bits
───────────────────────
Bit 0 — ASTC     : ASTC hardware-texture compressed weights
Bit 1 — TCA_TBE  : TCA-TBE lossless bitmap encoding
Bit 2 — INT4     : INT4-quantised weight blocks
Bit 3 — SPARSE   : structured FFN sparsity masks
Bit 4 — EAGLE    : trained EAGLE-3 draft head appendix present
Bit 5 — INT2     : INT2 sub-4-bit weights (hybrid blocks)
Bit 6 — ANE_COREML: ANE CoreML appendix present (Wave 69)

Dispatch priority
─────────────────
1. ASTC path (bit 0)            — hardware-decode on Apple Silicon GPU
2. TCA_TBE path (bit 1)         — bitmap lossless decode
3. INT4 / INT2 hybrid path      — fast Metal GEMV kernels
4. NumPy fallback               — always available (testing + CI)

Usage::

    from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags

    rt = SquishRuntime.from_file("model.squizd")
    print(rt.active_flags)          # SquizdFlags set
    tokens = rt.generate("Hello", max_new_tokens=32)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntFlag, auto
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

__all__ = [
    "SquizdFlags",
    "SquizdHeader",
    "KernelStack",
    "DispatchRecord",
    "SquishRuntime",
    "SQUIZD_MAGIC",
    "SQUIZD_VERSION",
]

# ---------------------------------------------------------------------------
# Magic bytes and version
# ---------------------------------------------------------------------------

SQUIZD_MAGIC: bytes = b"SQZD"
SQUIZD_VERSION: int = 1

# Header is 256 bytes total; first 12 bytes are fixed fields.
_HEADER_SIZE: int = 256
_MAGIC_OFFSET: int = 0
_VERSION_OFFSET: int = 4        # uint16 LE
_FLAGS_OFFSET: int = 6          # uint32 LE
_LAYER_COUNT_OFFSET: int = 10   # uint16 LE
_ARCH_ID_OFFSET: int = 12       # uint16 LE


# ---------------------------------------------------------------------------
# SQUIZD format flags
# ---------------------------------------------------------------------------

class SquizdFlags(IntFlag):
    """Bitfield of compression and feature flags stored in the file header."""

    NONE       = 0
    ASTC       = 1 << 0    # ASTC hardware-texture compressed weights
    TCA_TBE    = 1 << 1    # TCA-TBE lossless bitmap encoding
    INT4       = 1 << 2    # INT4 weight quantisation
    SPARSE     = 1 << 3    # Structured FFN sparsity
    EAGLE      = 1 << 4    # EAGLE-3 draft head appendix
    INT2       = 1 << 5    # INT2 sub-4-bit weight blocks
    ANE_COREML = 1 << 6    # ANE CoreML appendix (Wave 69)

    @classmethod
    def from_uint32(cls, value: int) -> "SquizdFlags":
        """Parse a 32-bit integer into a :class:`SquizdFlags` value."""
        return cls(value & 0xFFFFFFFF)

    def has(self, flag: "SquizdFlags") -> bool:
        """Return ``True`` if *flag* is set."""
        return bool(self & flag)


# ---------------------------------------------------------------------------
# Parsed header
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SquizdHeader:
    """Parsed contents of a SQUIZD file header.

    Attributes:
        magic: 4-byte magic string (``b"SQZD"`` for valid files).
        version: Format version integer (currently 1).
        flags: :class:`SquizdFlags` bitfield.
        layer_count: Number of transformer layers in the file.
        arch_id: Architecture identifier (0 = generic; used for routing hints).
        raw_bytes: Full 256-byte header bytes.
    """

    magic: bytes
    version: int
    flags: SquizdFlags
    layer_count: int
    arch_id: int
    raw_bytes: bytes

    @property
    def is_valid(self) -> bool:
        """Return ``True`` if magic bytes and version are recognised."""
        return self.magic == SQUIZD_MAGIC and self.version == SQUIZD_VERSION

    def summary(self) -> str:
        """Return a compact one-line summary string."""
        flag_names = [f.name for f in SquizdFlags if self.flags.has(f) and f != SquizdFlags.NONE]
        flags_str = "|".join(flag_names) if flag_names else "NONE"
        return (
            f"SQUIZD v{self.version} layers={self.layer_count} "
            f"arch={self.arch_id} flags=[{flags_str}]"
        )


# ---------------------------------------------------------------------------
# Kernel stack selection
# ---------------------------------------------------------------------------

class KernelStack:
    """String constants identifying available kernel stacks."""

    ASTC     = "astc"
    TCA_TBE  = "tca_tbe"
    INT4     = "int4"
    INT2     = "int2"
    NUMPY    = "numpy"
    COREML   = "coreml"


@dataclass
class DispatchRecord:
    """Dispatch decision for a single layer.

    Attributes:
        layer_idx: Zero-based transformer layer index.
        kernel_stack: One of the :class:`KernelStack` string constants.
        flags_active: Subset of header flags that influenced this choice.
        sparse_enabled: True if sparsity mask decoding is enabled.
        draft_enabled: True if this layer participates in EAGLE draft decode.
    """

    layer_idx: int
    kernel_stack: str
    flags_active: SquizdFlags
    sparse_enabled: bool = False
    draft_enabled: bool = False


# ---------------------------------------------------------------------------
# SquishRuntime
# ---------------------------------------------------------------------------

class SquishRuntime:
    """Unified inference runtime for SQUIZD-format models.

    Reads header flags and builds a per-layer dispatch table mapping each
    transformer layer to the correct kernel stack.  Inference is executed
    through :meth:`generate` (non-streaming) or :meth:`generate_stream`
    (token-by-token iterator).

    In production, each kernel stack invokes the appropriate Metal shader
    or coremltools delegate.  In this implementation each stack is simulated
    with deterministic NumPy operations to enable full test coverage without
    requiring Apple Silicon hardware.

    Parameters:
        header: Parsed :class:`SquizdHeader` from the model file.
        weights: Optional weight dictionary (``{name: np.ndarray}``).
        vocab_size: Output vocabulary size (default 32 000).
        _path: Source file path (informational; used in error messages).
    """

    def __init__(
        self,
        header: SquizdHeader,
        weights: Optional[Dict[str, np.ndarray]] = None,
        *,
        vocab_size: int = 32_000,
        _path: Optional[Path] = None,
    ) -> None:
        self.header = header
        self.weights = weights or {}
        self.vocab_size = vocab_size
        self._path = _path
        self._dispatch_table: List[DispatchRecord] = self._build_dispatch_table()

    # ------------------------------------------------------------------
    # Class-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> "SquishRuntime":
        """Load a :class:`SquishRuntime` from a ``.squizd`` file.

        For non-existent or non-SQUIZD files the method still succeeds by
        constructing a minimal header with ``NONE`` flags — callers should
        check ``runtime.header.is_valid``.

        Args:
            path: Filesystem path to the ``.squizd`` file.
            **kwargs: Forwarded to :class:`SquishRuntime`.

        Returns:
            A :class:`SquishRuntime` ready for inference.
        """
        p = Path(path)
        header = cls._read_header(p)
        return cls(header, _path=p, **kwargs)

    @classmethod
    def from_flags(
        cls,
        flags: Union[SquizdFlags, int],
        layer_count: int = 32,
        vocab_size: int = 32_000,
    ) -> "SquishRuntime":
        """Construct a runtime from explicit flags (for testing).

        Args:
            flags: :class:`SquizdFlags` or integer bitfield.
            layer_count: Pretend the model has this many layers.
            vocab_size: Vocabulary size.

        Returns:
            A :class:`SquishRuntime` with a synthetic header.
        """
        raw = SQUIZD_MAGIC + struct.pack("<HI HH", SQUIZD_VERSION, int(flags), layer_count, 0)
        raw = raw.ljust(_HEADER_SIZE, b"\x00")
        header = SquizdHeader(
            magic=SQUIZD_MAGIC,
            version=SQUIZD_VERSION,
            flags=SquizdFlags.from_uint32(int(flags)),
            layer_count=layer_count,
            arch_id=0,
            raw_bytes=raw,
        )
        return cls(header, vocab_size=vocab_size)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_flags(self) -> SquizdFlags:
        """Return the active :class:`SquizdFlags` from the file header."""
        return self.header.flags

    @property
    def layer_count(self) -> int:
        """Return the number of transformer layers."""
        return self.header.layer_count

    @property
    def dispatch_table(self) -> List[DispatchRecord]:
        """Return the per-layer dispatch table (read-only view)."""
        return list(self._dispatch_table)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a completion for *prompt* (non-streaming).

        Args:
            prompt: Plain-text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            seed: Deterministic RNG seed.

        Returns:
            Generated text as a plain string.
        """
        return "".join(
            tok for tok, _ in self.generate_stream(
                prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, seed=seed,
            )
        )

    def generate_stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> Iterator[tuple[str, Optional[str]]]:
        """Yield ``(token_text, finish_reason_or_None)`` pairs.

        The simulated forward pass runs through the dispatch table, applying
        the appropriate (simulated) kernel for each layer.

        Args:
            prompt: Plain-text prompt string.
            max_new_tokens: Token budget.
            temperature: Sampling temperature.
            seed: Deterministic seed.

        Yields:
            ``(token_str, finish_reason)`` where *finish_reason* is ``None``
            for intermediate tokens, ``"stop"`` on EOS, or ``"length"`` when
            the budget is exhausted.
        """
        rng = np.random.default_rng(seed=seed)
        input_ids = self._tokenise(prompt)

        for step in range(max_new_tokens):
            hidden = self._forward(input_ids, rng=rng)
            token_id = self._sample(hidden, rng=rng, temperature=temperature)
            token_text = f" t{token_id % 1000}"

            if token_id == 2:  # EOS
                yield (token_text, "stop")
                return
            if step == max_new_tokens - 1:
                yield (token_text, "length")
                return
            yield (token_text, None)
            input_ids = np.concatenate([input_ids, [[token_id]]], axis=1)

    # ------------------------------------------------------------------
    # Dispatch table construction
    # ------------------------------------------------------------------

    def _build_dispatch_table(self) -> List[DispatchRecord]:
        """Build the per-layer dispatch table from the header flags."""
        flags = self.header.flags
        records: List[DispatchRecord] = []

        for i in range(self.header.layer_count):
            stack = self._select_kernel(flags)
            records.append(
                DispatchRecord(
                    layer_idx=i,
                    kernel_stack=stack,
                    flags_active=flags,
                    sparse_enabled=flags.has(SquizdFlags.SPARSE),
                    draft_enabled=flags.has(SquizdFlags.EAGLE),
                )
            )
        return records

    @staticmethod
    def _select_kernel(flags: SquizdFlags) -> str:
        """Choose the best available kernel stack for *flags*."""
        if flags.has(SquizdFlags.ANE_COREML):
            return KernelStack.COREML
        if flags.has(SquizdFlags.ASTC):
            return KernelStack.ASTC
        if flags.has(SquizdFlags.TCA_TBE):
            return KernelStack.TCA_TBE
        if flags.has(SquizdFlags.INT2):
            return KernelStack.INT2
        if flags.has(SquizdFlags.INT4):
            return KernelStack.INT4
        return KernelStack.NUMPY

    # ------------------------------------------------------------------
    # Simulated forward pass
    # ------------------------------------------------------------------

    def _forward(self, input_ids: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        """Run the simulated forward pass for the current dispatch table."""
        hidden = input_ids.astype(np.float32)
        for rec in self._dispatch_table:
            hidden = self._layer_sim(hidden, rec, rng=rng)
        # Project the last token position to vocab logits.
        # hidden shape: (1, seq_len); take [:, -1:] → (1, 1) regardless of seq_len.
        proj = rng.standard_normal((1, self.vocab_size)).astype(np.float32)
        return (hidden[:, -1:] @ proj).reshape(1, self.vocab_size)

    @staticmethod
    def _layer_sim(
        x: np.ndarray, rec: DispatchRecord, *, rng: np.random.Generator
    ) -> np.ndarray:
        """Simulate a single transformer layer (deterministic scale + shift)."""
        scale = 1.0 + 0.01 * rec.layer_idx
        if rec.sparse_enabled:
            mask = rng.integers(0, 2, size=x.shape).astype(np.float32)
            return np.tanh(x * scale) * mask
        return np.tanh(x * scale)

    @staticmethod
    def _sample(logits: np.ndarray, *, rng: np.random.Generator, temperature: float) -> int:
        """Sample a token id from *logits*."""
        if temperature <= 0.0:
            return int(np.argmax(logits))
        scaled = logits / temperature
        shifted = scaled - scaled.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        return int(rng.choice(len(probs[0]), p=probs[0]))

    @staticmethod
    def _tokenise(text: str) -> np.ndarray:
        """Map text to a (1, seq_len) int64 array."""
        words = text.split() or ["<empty>"]
        ids = [(abs(hash(w)) % 29997) + 3 for w in words]
        return np.array([ids], dtype=np.int64)

    # ------------------------------------------------------------------
    # Header I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _read_header(path: Path) -> SquizdHeader:
        """Read and parse the SQUIZD header from *path*."""
        if not path.exists():
            return SquishRuntime._null_header()
        data = path.read_bytes()
        if len(data) < _HEADER_SIZE:
            return SquishRuntime._null_header()

        raw = data[:_HEADER_SIZE]
        magic = raw[_MAGIC_OFFSET:_MAGIC_OFFSET + 4]
        try:
            version, flags, layer_count, arch_id = struct.unpack_from(
                "<HI HH", raw, _VERSION_OFFSET
            )
        except struct.error:
            return SquishRuntime._null_header()

        return SquizdHeader(
            magic=magic,
            version=version,
            flags=SquizdFlags.from_uint32(flags),
            layer_count=int(layer_count),
            arch_id=int(arch_id),
            raw_bytes=raw,
        )

    @staticmethod
    def _null_header() -> SquizdHeader:
        """Return a header representing an absent or corrupt file."""
        return SquizdHeader(
            magic=b"\x00\x00\x00\x00",
            version=0,
            flags=SquizdFlags.NONE,
            layer_count=0,
            arch_id=0,
            raw_bytes=b"\x00" * _HEADER_SIZE,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_squizd_header(
        flags: Union[SquizdFlags, int],
        layer_count: int = 32,
        arch_id: int = 0,
    ) -> bytes:
        """Build a valid 256-byte SQUIZD header blob (for testing / writing).

        Args:
            flags: Compression flags.
            layer_count: Number of transformer layers.
            arch_id: Architecture identifier.

        Returns:
            256-byte ``bytes`` object representing the header.
        """
        raw = (
            SQUIZD_MAGIC
            + struct.pack("<HI HH", SQUIZD_VERSION, int(flags), layer_count, arch_id)
        )
        return raw.ljust(_HEADER_SIZE, b"\x00")
