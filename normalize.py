"""Per-crop GaussNN input normalization.

``minmax`` is a bit-exact NumPy port of Rheed-HW ``08_hls_design/pre_gaus_normalizer/
pre_gaus_normalizer.sv``, the block that sits between pCrop and the GAUS core:

    min, max  over the in-bounds pixels of the 48x48 crop (zero pads excluded)
    r         = max - min                          (8-bit)
    RECIP[r]  = round(2^18 / r),  RECIP[0] = 0     (flat crop -> all zeros)
    u         = ((p - min) * RECIP[r] + 128) >> 8  (0..1024, in-bounds pixels)
    u         = 0                                  (pads)

GAUS reads u as ap_fixed<16,6> (10 fractional bits), i.e. the network sees
u / 1024 ~= (p - min) / (max - min) in [0, 1]. Equivalent to the 0..255 form
pixel = (p - min) * 255 / (max - min), rescaled to the [0, 1] range of the GAUS input type
(255 itself does not fit in ap_fixed<16,6>).

``bg256`` is the previous normalization (subtract the per-crop minimum, divide by 256),
kept so older models can still be reproduced and evaluated.
"""

from __future__ import annotations

import numpy as np

MODES = ("minmax", "bg256")
DEFAULT_MODE = "minmax"

FRAC_BITS = 10  # GAUS input ap_fixed<16,6>
RECIP_BITS = 18

# RECIP[r] = round(2^18 / r), RECIP[0] = 0  (pre_gaus_normalizer.sv reciprocal ROM)
_r = np.arange(256, dtype=np.int64)
RECIP = np.where(_r > 0, ((1 << RECIP_BITS) + _r // 2) // np.maximum(_r, 1), 0).astype(np.int64)
del _r


def get_mode(cfg: dict | None) -> str:
    """Normalization mode from a config's ``normalization.mode`` (missing -> legacy bg256,
    so configs written before this option keep their old behaviour)."""
    mode = ((cfg or {}).get("normalization") or {}).get("mode", "bg256")
    if mode not in MODES:
        raise ValueError(f"normalization.mode must be one of {MODES}, got {mode!r}")
    return mode


CHUNK = 4096  # crops per block in minmax_normalize, bounds the int32 temporaries

def to_pixels(crops: np.ndarray) -> np.ndarray:
    """Round and clip to the 8-bit pixels pCrop hands the normalizer (int32: d * RECIP[r] is at
    most 255 * 2^18 < 2^31)."""
    return np.clip(np.rint(crops), 0, 255).astype(np.int32)


def crop_stats(pix: np.ndarray, inb: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-crop (min, range) over in-bounds pixels; crops are the leading axis.
    A crop without in-bounds pixels gets min 255, range 0 (hardware: run_min FF, run_max 00)."""
    n = pix.shape[0]
    flat = pix.reshape(n, -1)
    if inb is None:
        mn, mx = flat.min(axis=1), flat.max(axis=1)
    else:
        m = inb.reshape(n, -1).astype(bool)
        mn = np.where(m, flat, 255).min(axis=1)
        mx = np.where(m, flat, 0).max(axis=1)
    rng = np.where(mx >= mn, mx - mn, 0)
    return mn, rng


def minmax_normalize(crops: np.ndarray, inb: np.ndarray | None = None) -> np.ndarray:
    """Hardware min-max normalization of a batch of crops (leading axis = crop).

    crops: pixel values (any float/int dtype; rounded and clipped to 0..255).
    inb:   optional boolean mask, same shape, True = in-bounds pixel, False = zero pad.
    Returns float32 values u / 1024 in [0, 1], same shape as crops.
    """
    out = np.empty(crops.shape, dtype=np.float32)
    for s in range(0, crops.shape[0], CHUNK):
        sl = slice(s, s + CHUNK)
        out[sl] = _minmax_block(crops[sl], None if inb is None else inb[sl])
    return out


def _minmax_block(crops: np.ndarray, inb: np.ndarray | None) -> np.ndarray:
    pix = to_pixels(crops)
    mn, rng = crop_stats(pix, inb)
    bshape = (-1,) + (1,) * (pix.ndim - 1)
    d = pix - mn.reshape(bshape)
    if inb is not None:
        d = np.where(inb.astype(bool), d, 0)
    d = np.clip(d, 0, None)  # in-bounds pixels are always >= min; guards pads without a mask
    u = (d * RECIP[rng].astype(np.int32).reshape(bshape) + 128) >> 8
    return u.astype(np.float32) / np.float32(1 << FRAC_BITS)


def bg256_normalize(crops: np.ndarray) -> np.ndarray:
    """Legacy: subtract each crop's minimum, divide by 256."""
    n = crops.shape[0]
    bg = crops.reshape(n, -1).min(axis=1).reshape((-1,) + (1,) * (crops.ndim - 1))
    return (np.clip(crops - bg, 0.0, None) / 256.0).astype(np.float32)


def normalize(crops: np.ndarray, mode: str, inb: np.ndarray | None = None) -> np.ndarray:
    if mode == "minmax":
        return minmax_normalize(crops, inb)
    if mode == "bg256":
        return bg256_normalize(crops)
    raise ValueError(f"unknown normalization mode {mode!r}")
