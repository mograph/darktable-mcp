"""Write Darktable-compatible XMP sidecar files from an EditState.

Darktable stores per-module history items in its XMP sidecars.  Each item's
``darktable:params`` field is a little-endian hex-encoded binary blob whose
layout matches the module's C struct (version-dependent).

Global (unmasked) modules — blendop v7:
  - exposure   (modversion 6)
  - temperature (modversion 5, simplified to camera-WB coefficients)
  - brightness_contrast (modversion 1)
  - crop/rotate via ``crop`` module (modversion 5)
  - sharpen, denoiseprofile, vignette

Masked modules — blendop v14 (darktable 5.x struct):
  - rasterfile  (modversion 1) — loads a PFM mask file
  - exposure    (modversion 6) with raster mask reference
  - colorbalancergb (modversion 5) with raster mask reference

Blendop v14 layout (420 bytes, darktable 5.x dt_develop_blend_params_t):
  offset  0: uint32 mask_mode   — 0=disabled, 9=ENABLED|RASTER
  offset  4: int32  blend_cst
  offset  8: uint32 blend_mode
  offset 12: float  blend_parameter
  offset 16: float  opacity
  offset 20: uint32 mask_combine
  offset 24: int32  mask_id
  offset 28: uint32 blendif
  offset 32: float  feathering_radius
  offset 36: uint32 feathering_guide
  offset 40: float  blur_radius
  offset 44: float  contrast
  offset 48: float  brightness
  offset 52: float  details
  offset 56: uint32 feather_version
  offset 60: uint32 reserved[2]
  offset 68: float  blendif_parameters[64]   (256 bytes)
  offset 324: float blendif_boost_factors[16] (64 bytes)
  offset 388: char  raster_mask_source[20]   (dt_dev_operation_t)
  offset 408: int32 raster_mask_instance
  offset 412: int32 raster_mask_id
  offset 416: int32 raster_mask_invert
  total: 420 bytes

Mask mode flags (dt_develop_mask_mode_t):
  DEVELOP_MASK_DISABLED = 0
  DEVELOP_MASK_ENABLED  = 1
  DEVELOP_MASK_MASK     = 2
  DEVELOP_MASK_CONDITIONAL = 4
  DEVELOP_MASK_RASTER   = 8  → combined with ENABLED = 9
"""
from __future__ import annotations

import json
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
        float cx;    // x center of crop, normalised 0..1
        float cy;    // y center of crop, normalised 0..1
        float cw;    // crop width, normalised 0..1
        float ch;    // crop height, normalised 0..1
        float angle; // degrees, darktable CCW convention (negate our CW value)
    }  = 5 floats = 20 bytes
    """
    cx = (crop.left + crop.right) / 2.0
    cy = (crop.top + crop.bottom) / 2.0
    cw = crop.right - crop.left
    ch = crop.bottom - crop.top
    data = struct.pack("<fffff", cx, cy, cw, ch, -crop.rotation)
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


def _blendop_v14_default() -> str:
    """Blendop v14 (420 bytes) with no mask — opacity 1.0, all else default."""
    buf = bytearray(420)
    struct.pack_into("<I", buf, 0, 0)       # mask_mode = DISABLED
    struct.pack_into("<f", buf, 16, 1.0)    # opacity = 1.0
    return buf.hex()


def _blendop_v14_raster(source_op: str, instance: int = 0, invert: bool = False) -> str:
    """Blendop v14 (420 bytes) referencing a rasterfile module as the mask.

    source_op: the darktable:operation value of the rasterfile module, e.g. "rasterfile"
    instance:  multi_priority of the rasterfile module (0 for first, 1 for second, etc.)
    """
    buf = bytearray(420)
    # DEVELOP_MASK_ENABLED | DEVELOP_MASK_RASTER = 1 | 8 = 9
    struct.pack_into("<I", buf, 0, 9)
    struct.pack_into("<i", buf, 4, 0)       # blend_cst
    struct.pack_into("<I", buf, 8, 0)       # blend_mode = NORMAL
    struct.pack_into("<f", buf, 12, 0.0)    # blend_parameter
    struct.pack_into("<f", buf, 16, 1.0)    # opacity = 1.0
    struct.pack_into("<I", buf, 20, 0)      # mask_combine
    struct.pack_into("<i", buf, 24, 0)      # mask_id
    struct.pack_into("<I", buf, 28, 0)      # blendif
    struct.pack_into("<f", buf, 32, 0.0)    # feathering_radius
    struct.pack_into("<I", buf, 36, 0)      # feathering_guide
    struct.pack_into("<f", buf, 40, 0.0)    # blur_radius
    struct.pack_into("<f", buf, 44, 0.0)    # contrast
    struct.pack_into("<f", buf, 48, 0.0)    # brightness
    struct.pack_into("<f", buf, 52, 0.0)    # details
    struct.pack_into("<I", buf, 56, 0)      # feather_version
    # reserved[2] stays zero (offset 60)
    # blendif_parameters[64] stays zero (offset 68)
    # blendif_boost_factors[16] stays zero (offset 324)
    op_b = source_op.encode("utf-8")[:19]
    buf[388: 388 + len(op_b)] = op_b        # raster_mask_source[20]
    struct.pack_into("<i", buf, 408, instance)  # raster_mask_instance
    struct.pack_into("<i", buf, 412, 0)     # raster_mask_id
    struct.pack_into("<i", buf, 416, 1 if invert else 0)  # raster_mask_invert
    return buf.hex()


def _encode_rasterfile(pfm_dir: str, pfm_file: str) -> str:
    """rasterfile module params (version 1, 4100 bytes).

    struct dt_iop_rasterfile_params_t {
        int32_t mode;         // DT_RASTERFILE_MODE_ALL = 7
        char    path[2048];   // directory
        char    file[2048];   // filename
    }
    """
    buf = bytearray(4100)
    struct.pack_into("<i", buf, 0, 7)       # DT_RASTERFILE_MODE_ALL = 7
    path_b = str(pfm_dir).encode("utf-8")[:2047]
    file_b = str(pfm_file).encode("utf-8")[:2047]
    buf[4: 4 + len(path_b)] = path_b
    buf[2052: 2052 + len(file_b)] = file_b
    return buf.hex()


def _encode_colorbalancergb(
    shadows_Y: float = 0.0,
    shadows_C: float = 0.0,
    shadows_H: float = 0.0,
    midtones_Y: float = 0.0,
    midtones_C: float = 0.0,
    midtones_H: float = 0.0,
    highlights_Y: float = 0.0,
    highlights_C: float = 0.0,
    highlights_H: float = 0.0,
    global_Y: float = 0.0,
    global_C: float = 0.0,
    global_H: float = 0.0,
    shadows_weight: float = 0.25,
    white_fulcrum: float = 1.0,
    highlights_weight: float = 0.25,
    chroma_shadows: float = 0.0,
    chroma_highlights: float = 0.0,
    chroma_global: float = 0.0,
    chroma_midtones: float = 0.0,
    saturation_global: float = 0.0,
    saturation_highlights: float = 0.0,
    saturation_midtones: float = 0.0,
    saturation_shadows: float = 0.0,
    hue_angle: float = 0.0,
    brilliance_global: float = 0.0,
    brilliance_highlights: float = 0.0,
    brilliance_midtones: float = 0.0,
    brilliance_shadows: float = 0.0,
    mask_grey_fulcrum: float = 0.1845,
    vibrance: float = 0.0,
    grey_fulcrum: float = 0.1845,
    contrast: float = 0.0,
    saturation_formula: int = 0,  # DT_COLORBALANCE_SATURATION_JZAZBZ = 0
) -> str:
    """colorbalancergb module params (version 5, 132 bytes).

    32 floats + 1 int32 = 128 + 4 = 132 bytes.
    Field order matches dt_iop_colorbalancergb_params_t in darktable 5.x source.
    """
    data = struct.pack(
        "<ffffffffffffffffffffffffffffffffi",
        shadows_Y, shadows_C, shadows_H,
        midtones_Y, midtones_C, midtones_H,
        highlights_Y, highlights_C, highlights_H,
        global_Y, global_C, global_H,
        shadows_weight, white_fulcrum, highlights_weight,
        chroma_shadows, chroma_highlights, chroma_global, chroma_midtones,
        saturation_global, saturation_highlights, saturation_midtones, saturation_shadows,
        hue_angle,
        brilliance_global, brilliance_highlights, brilliance_midtones, brilliance_shadows,
        mask_grey_fulcrum,
        vibrance, grey_fulcrum, contrast,
        saturation_formula,
    )
    return data.hex()


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


def _encode_filmicrgb(
    grey_point_source: float = 18.0,
    black_point_source: float = -7.0,
    white_point_source: float = 4.0,
    reconstruct_threshold: float = 1.0,
    reconstruct_feather: float = 3.0,
    reconstruct_bloom_vs_details: float = 100.0,
    reconstruct_grey_vs_color: float = 100.0,
    reconstruct_structure_vs_texture: float = 0.0,
    security_factor: float = 0.0,
    grey_point_target: float = 18.45,
    black_point_target: float = 0.0,
    white_point_target: float = 100.0,
    output_power: float = 4.0,
    latitude: float = 33.0,
    contrast: float = 1.0,
    saturation: float = 0.0,
    balance: float = 0.0,
    noise_level: float = 0.2,
    preserve_color: int = 1,       # DT_FILMIC_METHOD_MAX_RGB
    version: int = 4,              # DT_FILMIC_COLORSCIENCE_V5
    auto_hardness: int = 1,
    custom_grey: int = 0,
    high_quality_reconstruction: int = 1,
    noise_distribution: int = 2,   # DT_NOISE_POISSONIAN
    shadows: int = 0,              # DT_FILMIC_CURVE_POLY_4
    highlights: int = 0,
    compensate_icc_black: int = 0,
    spline_version: int = 2,       # DT_FILMIC_SPLINE_VERSION_V3
    enable_highlight_reconstruction: int = 1,
) -> str:
    """filmicrgb module params version 6 (116 bytes).

    struct dt_iop_filmicrgb_params_t: 18 floats + 11 int32s.
    Verified against darktable master src/iop/filmicrgb.c.
    """
    data = struct.pack(
        "<ffffffffffffffffffiiiiiiiiiii",
        grey_point_source, black_point_source, white_point_source,
        reconstruct_threshold, reconstruct_feather,
        reconstruct_bloom_vs_details, reconstruct_grey_vs_color,
        reconstruct_structure_vs_texture, security_factor,
        grey_point_target, black_point_target, white_point_target,
        output_power, latitude, contrast, saturation, balance, noise_level,
        preserve_color, version, auto_hardness, custom_grey,
        high_quality_reconstruction, noise_distribution,
        shadows, highlights, compensate_icc_black, spline_version,
        enable_highlight_reconstruction,
    )
    return data.hex()


def _encode_shadhi(
    shadows: float = 50.0,
    highlights: float = -50.0,
    radius: float = 100.0,
    whitepoint: float = 0.0,
    compress: float = 50.0,
    shadows_ccorrect: float = 100.0,
    highlights_ccorrect: float = 50.0,
    low_approximation: float = 0.01,
) -> str:
    """shadhi (shadows & highlights) module params version 5 (48 bytes).

    struct dt_iop_shadhi_params_t: int32 + 8 floats + uint32 + float + int32.
    Verified against darktable master src/iop/shadhi.c.

    shadows: positive = lift shadows (0–100)
    highlights: negative = compress highlights (-100–0)
    """
    order = 0       # DT_IOP_GAUSSIAN_ZERO
    reserved2 = 0.0
    flags = 0
    shadhi_algo = 1  # DT_SHADHI_ALGO_BILATERAL
    data = struct.pack(
        "<iffffffffIfi",
        order,
        radius,
        shadows,
        whitepoint,
        highlights,
        reserved2,
        compress,
        shadows_ccorrect,
        highlights_ccorrect,
        flags,
        low_approximation,
        shadhi_algo,
    )
    return data.hex()


def _encode_levels(black: float = 0.0, grey: float = 1.0, white: float = 1.0) -> str:
    """levels module params version 3 (12 bytes).

    struct dt_iop_levels_params_t { float levels[3]; }
    black: input black point 0..1
    grey:  input mid-tone gamma 0.1..10
    white: input white point 0..1
    """
    return struct.pack("<fff", black, grey, white).hex()


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
   <darktable:mask_id>
    <rdf:Seq/>
   </darktable:mask_id>
   <darktable:mask_type>
    <rdf:Seq/>
   </darktable:mask_type>
   <darktable:mask_name>
    <rdf:Seq/>
   </darktable:mask_name>
   <darktable:mask_version>
    <rdf:Seq/>
   </darktable:mask_version>
   <darktable:mask>
    <rdf:Seq/>
   </darktable:mask>
   <darktable:mask_nb>
    <rdf:Seq/>
   </darktable:mask_nb>
   <darktable:mask_src>
    <rdf:Seq/>
   </darktable:mask_src>
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


# ---------------------------------------------------------------------------
# Masked ops sidecar (.masked_ops.json)
# ---------------------------------------------------------------------------

def _masked_ops_path(raw_path: Path) -> Path:
    return raw_path.with_name(raw_path.stem + ".masked_ops.json")


def load_masked_ops(raw_path: Path) -> list[dict]:
    """Load the list of pending masked ops for a RAW file."""
    p = _masked_ops_path(raw_path)
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def save_masked_ops(raw_path: Path, ops: list[dict]) -> None:
    p = _masked_ops_path(raw_path)
    with open(p, "w") as f:
        json.dump(ops, f, indent=2)


def clear_masked_ops(raw_path: Path) -> None:
    p = _masked_ops_path(raw_path)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# History item template for blendop v14 (masked modules)
# ---------------------------------------------------------------------------

_HISTORY_ITEM_V14 = """\
     <rdf:li
      darktable:operation="{operation}"
      darktable:enabled="1"
      darktable:modversion="{modversion}"
      darktable:params="{params}"
      darktable:multi_name=""
      darktable:multi_priority="{multi_priority}"
      darktable:blendop_version="14"
      darktable:blendop_params="{blendop}"
      />
"""


# ---------------------------------------------------------------------------
# Combined XMP writer: global adjustments + masked ops
# ---------------------------------------------------------------------------

_SUPPORTED_MODULES = (
    "exposure", "colorbalancergb", "filmicrgb", "shadhi", "levels",
    "sharpen", "temperature", "brightness_contrast", "vignette", "denoiseprofile",
)


def _encode_module(module: str, params: dict) -> tuple[int, str]:
    """Encode a module's params dict into (modversion, hex_params)."""
    if module == "exposure":
        return 6, _encode_exposure(
            params.get("exposure_ev", 0.0),
            params.get("black_level", 0.0),
        )
    if module == "colorbalancergb":
        return 5, _encode_colorbalancergb(**{
            k: v for k, v in params.items()
            if k in _COLORBALANCERGB_FIELDS
        })
    if module == "filmicrgb":
        return 6, _encode_filmicrgb(**{
            k: v for k, v in params.items()
            if k in {
                "grey_point_source", "black_point_source", "white_point_source",
                "reconstruct_threshold", "reconstruct_feather",
                "reconstruct_bloom_vs_details", "reconstruct_grey_vs_color",
                "reconstruct_structure_vs_texture", "security_factor",
                "grey_point_target", "black_point_target", "white_point_target",
                "output_power", "latitude", "contrast", "saturation", "balance",
                "noise_level", "preserve_color", "version", "auto_hardness",
                "custom_grey", "high_quality_reconstruction", "noise_distribution",
                "shadows", "highlights", "compensate_icc_black", "spline_version",
                "enable_highlight_reconstruction",
            }
        })
    if module == "shadhi":
        return 5, _encode_shadhi(
            shadows=params.get("shadows", 50.0),
            highlights=params.get("highlights", -50.0),
            radius=params.get("radius", 100.0),
            whitepoint=params.get("whitepoint", 0.0),
            compress=params.get("compress", 50.0),
            shadows_ccorrect=params.get("shadows_ccorrect", 100.0),
            highlights_ccorrect=params.get("highlights_ccorrect", 50.0),
            low_approximation=params.get("low_approximation", 0.01),
        )
    if module == "levels":
        return 3, _encode_levels(
            black=params.get("black", 0.0),
            grey=params.get("grey", 1.0),
            white=params.get("white", 1.0),
        )
    if module == "sharpen":
        return 3, _encode_sharpen(params.get("amount", 50.0))
    if module == "temperature":
        return 5, _encode_temperature(
            params.get("temperature_kelvin", 5500.0),
            params.get("tint", 0.0),
        )
    if module == "brightness_contrast":
        return 1, _encode_brightness_contrast(
            params.get("brightness", 0.0),
            params.get("contrast", 0.0),
        )
    if module == "vignette":
        return 5, _encode_vignette(params.get("amount", 0.0))
    if module == "denoiseprofile":
        return 10, _encode_denoiseprofile(params.get("strength", 50.0))
    raise ValueError(
        f"Module '{module}' is not supported. "
        f"Supported: {', '.join(_SUPPORTED_MODULES)}"
    )


_COLORBALANCERGB_FIELDS = {
    "shadows_Y", "shadows_C", "shadows_H",
    "midtones_Y", "midtones_C", "midtones_H",
    "highlights_Y", "highlights_C", "highlights_H",
    "global_Y", "global_C", "global_H",
    "shadows_weight", "white_fulcrum", "highlights_weight",
    "chroma_shadows", "chroma_highlights", "chroma_global", "chroma_midtones",
    "saturation_global", "saturation_highlights", "saturation_midtones", "saturation_shadows",
    "hue_angle",
    "brilliance_global", "brilliance_highlights", "brilliance_midtones", "brilliance_shadows",
    "mask_grey_fulcrum", "vibrance", "grey_fulcrum", "contrast", "saturation_formula",
}


def write_full_xmp(edit_state: EditState) -> Path:
    """Write a complete XMP combining global adjustments + any saved masked ops.

    Masked ops (from .masked_ops.json) are appended after global history items.
    Each masked op injects a rasterfile module + a blendop-v14 edit module pair.
    """
    raw_path = edit_state.source_path
    adj = edit_state.adjustments
    global_blendop = _blendop()
    items: list[str] = []

    def _item(op: str, ver: int, params: str) -> str:
        return _HISTORY_ITEM.format(
            operation=op, modversion=ver, params=params, blendop=global_blendop
        )

    # --- Global (unmasked) history items ---
    if adj.exposure_ev != 0.0 or adj.black_level != 0.0:
        items.append(_item("exposure", 6, _encode_exposure(adj.exposure_ev, adj.black_level)))

    if adj.temperature_kelvin is not None:
        items.append(_item("temperature", 5, _encode_temperature(adj.temperature_kelvin, adj.tint)))

    if adj.brightness != 0.0 or adj.contrast != 0.0:
        items.append(_item("brightness_contrast", 1, _encode_brightness_contrast(adj.brightness, adj.contrast)))

    if adj.sharpness > 0:
        items.append(_item("sharpen", 3, _encode_sharpen(adj.sharpness)))

    if adj.noise_reduction > 0:
        items.append(_item("denoiseprofile", 10, _encode_denoiseprofile(adj.noise_reduction)))

    if adj.vignette != 0.0:
        items.append(_item("vignette", 5, _encode_vignette(adj.vignette)))

    if edit_state.crop:
        items.append(_item("crop", 5, _encode_crop(edit_state.crop)))

    # --- Masked ops ---
    masked_ops = load_masked_ops(raw_path)
    for i, op in enumerate(masked_ops):
        pfm_path = Path(op["pfm_path"]) if op.get("pfm_path") else None
        module = op["module"]
        params = op.get("params", {})

        if pfm_path and pfm_path.exists():
            # Inject rasterfile module before the edit
            rf_params = _encode_rasterfile(str(pfm_path.parent), pfm_path.name)
            rf_blendop = _blendop_v14_default()
            items.append(_HISTORY_ITEM_V14.format(
                operation="rasterfile",
                modversion=1,
                params=rf_params,
                multi_priority=i,
                blendop=rf_blendop,
            ))
            edit_blendop = _blendop_v14_raster("rasterfile", instance=i)
        else:
            edit_blendop = _blendop_v14_default()

        modversion, params_hex = _encode_module(module, params)  # raises ValueError for unknown

        items.append(_HISTORY_ITEM_V14.format(
            operation=module,
            modversion=modversion,
            params=params_hex,
            multi_priority=0,
            blendop=edit_blendop,
        ))

    xmp_content = _XMP_TEMPLATE.format(
        history_end=len(items),
        history_items="".join(items),
    )

    xmp_path = raw_path.with_suffix(".xmp")
    xmp_path.write_text(xmp_content, encoding="utf-8")
    return xmp_path
