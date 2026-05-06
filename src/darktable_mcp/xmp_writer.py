"""Write Darktable-compatible XMP sidecar files from an EditState.

Darktable stores per-module history items in its XMP sidecars.  Each item's
``darktable:params`` field is a little-endian hex-encoded binary blob whose
layout matches the module's C struct (version-dependent).

We implement only the modules whose struct layout is well-known and stable:
  - exposure   (modversion 6)
  - temperature (modversion 5, simplified to camera-WB coefficients)
  - brightness_contrast (modversion 1)
  - colorbalance saturation via ``colorin`` is complex – we skip it
  - crop/rotate via ``crop`` module (modversion 5)

Unknown adjustments are silently skipped in the XMP; they are still applied
when exporting via rawpy/Pillow.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

from .edits import EditState, AdjustmentState, CropState

# ---------------------------------------------------------------------------
# Blendop default (7 bytes + padding, version 7)
# Values from a vanilla Darktable export with no blending.
# ---------------------------------------------------------------------------
_BLENDOP_DEFAULT = (
    "00000000"  # mask_mode = DEVELOP_MASK_DISABLED (0)
    "00000000"  # blend_mode = DEVELOP_BLEND_NORMAL (0)
    "000000000000000000000000"  # opacity = 1.0 → we encode as float below
    "00000000"  # mask_id = 0
    "07000000"  # blendop_version = 7
    "00000000" * 16  # feathering / details padding
)

# Simpler constant pulled from real darktable output
_BLENDOP_V7 = (
    "0c000000"  # mask_mode DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK_CONDITIONAL
    "0c000000"  # DEVELOP_BLEND_NORMAL8
    + "0000803f"  # opacity 1.0 (float LE)
    + "00000000"  # mask_id
    + "00000000" * 30  # padding to 140 bytes
)

# Use a safe minimal blendop blob that darktable accepts
def _blendop() -> str:
    # 140-byte blendop_params_v7: just set opacity=1.0, rest zeros
    buf = bytearray(140)
    # mask_mode = 0 (DEVELOP_MASK_DISABLED)
    struct.pack_into("<I", buf, 0, 0)
    # blend_mode = 12 (DEVELOP_BLEND_NORMAL_UNBOUNDED in older DT; 0 = passthrough)
    struct.pack_into("<I", buf, 4, 0)
    # opacity = 1.0
    struct.pack_into("<f", buf, 8, 1.0)
    return buf.hex()


# ---------------------------------------------------------------------------
# Module param encoders
# ---------------------------------------------------------------------------

def _encode_exposure(exposure_ev: float, black: float = 0.0) -> str:
    """exposure module version 6.

    struct dt_iop_exposure_params_t {
        dt_iop_exposure_mode_t mode;  // int32, 0=MANUAL
        float black;
        float exposure;
        float deflicker_percentile;
        float deflicker_target_level;
    }
    """
    data = struct.pack("<iffff", 0, black, exposure_ev, 50.0, -4.0)
    return data.hex()


def _encode_temperature(temp_k: float, tint: float = 0.0) -> str:
    """temperature module version 5 (simplified).

    We write coefficients for a basic Planckian-locus approximation.
    struct dt_iop_temperature_params_t {
        float temp_out;       // colour temperature (informational)
        float coeffs[4][2];   // wb multipliers per illuminant
        gboolean adapt_3way;  // 4 bytes
        float g_mix;
    }  ≈ 4 + 32 + 4 + 4 = 44 bytes
    """
    ref_k = 5500.0
    ratio = temp_k / ref_k
    r = (1.0 / ratio) ** 1.2
    g = 1.0 + (tint / 100.0) * 0.15
    b = ratio ** 1.2
    mn = min(r, g, b)
    r, g, b = r / mn, g / mn, b / mn

    buf = bytearray(44)
    struct.pack_into("<f", buf, 0, temp_k)
    # coeffs[4][2] — we fill illuminant 0 (D65) only
    struct.pack_into("<ff", buf, 4, r, g)
    struct.pack_into("<ff", buf, 12, b, g)
    # adapt_3way = 0
    struct.pack_into("<I", buf, 36, 0)
    # g_mix = 0.5
    struct.pack_into("<f", buf, 40, 0.5)
    return buf.hex()


def _encode_brightness_contrast(brightness: float, contrast: float) -> str:
    """brightness_contrast module version 1.

    struct dt_iop_brightnesscontrast_params_t {
        float brightness;  // -1.0..1.0
        float contrast;    // -1.0..1.0
    }
    """
    b = brightness / 100.0
    c = contrast / 100.0
    data = struct.pack("<ff", b, c)
    return data.hex()


def _encode_crop(crop: CropState) -> str:
    """crop module version 5.

    struct dt_iop_crop_params_t {
        float cx, cy, cw, ch;   // normalised 0..1
        float angle;             // degrees
    }  = 5 floats = 20 bytes
    """
    data = struct.pack(
        "<fffff",
        crop.left,
        crop.top,
        crop.right,
        crop.bottom,
        -crop.rotation,  # darktable uses CCW convention
    )
    return data.hex()


def _encode_sharpen(amount: float) -> str:
    """sharpen module version 3.

    struct dt_iop_sharpen_params_t {
        float radius;
        float amount;
        float threshold;
    }
    """
    radius = 1.5 + amount / 100.0 * 4.0
    amt = amount / 100.0 * 0.5
    data = struct.pack("<fff", radius, amt, 0.01)
    return data.hex()


def _encode_denoiseprofile(strength: float) -> str:
    """denoise (profiled) module version 10 — very simplified.

    We only set strength; the rest uses darktable defaults (zeros map to
    no-op for most fields, and darktable will use the camera profile).
    struct size ≈ 32 floats for v10.
    """
    buf = bytearray(128)
    struct.pack_into("<f", buf, 0, strength / 100.0)
    return buf.hex()


def _encode_vignette(amount: float) -> str:
    """vignette module version 5.

    struct dt_iop_vignette_params_t {
        float scale;       // 0..1, size of the unvignetted centre
        float falloff_scale; // 0..1
        float brightness; // -1..1 (negative = darker)
        float saturation; // 0..1
        dt_iop_vignette_shape_t shape; // int, 0=ELLIPSE
        int   unbound;
    }  = 4 floats + 2 ints = 24 bytes
    """
    sat = 0.8
    brightness = -abs(amount) / 100.0 if amount < 0 else abs(amount) / 100.0
    scale = 0.5
    falloff = 0.5
    data = struct.pack("<ffffii", scale, falloff, brightness, sat, 0, 0)
    return data.hex()


# ---------------------------------------------------------------------------
# XMP builder
# ---------------------------------------------------------------------------

_XMP_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:darktable="http://darktable.sf.net/"
    darktable:xmp_version="4"
    darktable:raw_params="0"
    darktable:auto_presets_applied="1"
    darktable:history_end="{history_end}">
   <darktable:history>
    <rdf:Seq>
{history_items}    </rdf:Seq>
   </darktable:history>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

_HISTORY_ITEM = """\
     <rdf:li
      darktable:operation="{operation}"
      darktable:enabled="1"
      darktable:modversion="{modversion}"
      darktable:params="{params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="7"
      darktable:blendop_params="{blendop}"
      />
"""


def write_xmp(edit_state: EditState) -> Path:
    """Write a Darktable XMP sidecar for *edit_state* and return its path."""
    adj = edit_state.adjustments
    blendop = _blendop()
    items: list[str] = []

    def _item(op: str, ver: int, params: str) -> str:
        return _HISTORY_ITEM.format(
            operation=op, modversion=ver, params=params, blendop=blendop
        )

    # Exposure
    if adj.exposure_ev != 0.0 or adj.black_level != 0.0:
        items.append(_item("exposure", 6, _encode_exposure(adj.exposure_ev, adj.black_level)))

    # White balance (temperature)
    if adj.temperature_kelvin is not None:
        items.append(_item("temperature", 5, _encode_temperature(adj.temperature_kelvin, adj.tint)))

    # Brightness / contrast
    if adj.brightness != 0.0 or adj.contrast != 0.0:
        items.append(_item("brightness_contrast", 1, _encode_brightness_contrast(adj.brightness, adj.contrast)))

    # Sharpening
    if adj.sharpness > 0:
        items.append(_item("sharpen", 3, _encode_sharpen(adj.sharpness)))

    # Noise reduction
    if adj.noise_reduction > 0:
        items.append(_item("denoiseprofile", 10, _encode_denoiseprofile(adj.noise_reduction)))

    # Vignette
    if adj.vignette != 0.0:
        items.append(_item("vignette", 5, _encode_vignette(adj.vignette)))

    # Crop (darktable treats crop as a module too)
    if edit_state.crop:
        items.append(_item("crop", 5, _encode_crop(edit_state.crop)))

    xmp_content = _XMP_TEMPLATE.format(
        history_end=len(items),
        history_items="".join(items),
    )

    xmp_path = edit_state.source_path.with_suffix(".xmp")
    xmp_path.write_text(xmp_content, encoding="utf-8")
    return xmp_path
