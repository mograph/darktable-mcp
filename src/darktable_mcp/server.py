"""Darktable MCP Server — photo editing tools for Claude."""
from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP, Image

from .edits import CropState, EditState
from .processor import ALL_EXTENSIONS, RAW_EXTENSIONS, ImageProcessor
from .xmp_writer import (
    write_xmp,
    write_full_xmp,
    load_masked_ops,
    save_masked_ops,
    clear_masked_ops,
)

mcp = FastMCP("DarktableMCP")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT_MACOS_PATHS = [
    "/Applications/darktable.app/Contents/MacOS/darktable-cli",
    "/usr/local/bin/darktable-cli",
]
_DT_WIN_PATH = r"C:\Program Files\darktable\bin\darktable-cli.exe"


def _find_dt_cli() -> Optional[str]:
    found = shutil.which("darktable-cli")
    if found:
        return found
    for p in _DT_MACOS_PATHS:
        if Path(p).exists():
            return p
    if Path(_DT_WIN_PATH).exists():
        return _DT_WIN_PATH
    return None


DARKTABLE_CLI = _find_dt_cli()


def _load_or_new(image_path: str) -> tuple[Path, EditState]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    state = EditState.load(path) or EditState(source_path=path)
    return path, state


def _quick_preview(path: Path, state: EditState, max_size: int) -> Image:
    proc = ImageProcessor(path)
    img = proc.process(state, preview_size=max_size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return img, buf.getvalue()


def _save_and_open_preview(path: Path, img_bytes: bytes) -> str:
    """Save preview to disk next to the source file and open it."""
    import os
    preview_path = path.with_name(path.stem + "__preview.jpg")
    preview_path.write_bytes(img_bytes)
    try:
        import os
        os.startfile(str(preview_path))   # Windows: opens in default image viewer
    except Exception:
        pass
    return str(preview_path)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_images(directory: str) -> list[dict]:
    """List all supported image files (RAW and raster) in *directory*.

    Returns a list of dicts with keys: path, name, type, has_edits, output_name.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return [{"error": f"Directory not found: {directory}"}]

    results = []
    for file in sorted(dir_path.iterdir()):
        if file.suffix.lower() not in ALL_EXTENSIONS:
            continue
        state = EditState.load(file)
        results.append({
            "path": str(file),
            "name": file.name,
            "type": "raw" if file.suffix.lower() in RAW_EXTENSIONS else "image",
            "has_edits": state.has_changes() if state else False,
            "output_name": state.output_name if state else None,
        })
    return results


@mcp.tool()
def get_image_info(image_path: str) -> dict:
    """Return detailed metadata (EXIF, dimensions, camera info) and current edit state for an image."""
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    proc = ImageProcessor(path)
    info = proc.get_info()
    info["edits"] = state.to_dict()
    info["darktable_cli_available"] = DARKTABLE_CLI is not None
    return info


@mcp.tool()
def get_image_preview(image_path: str, max_size: int = 1200) -> list:
    """Render the image with all current edits applied, save a preview file, and open it.

    Saves a __preview.jpg next to the source and opens it in the Windows image
    viewer so the user can see it immediately.  Also returns the image so Claude
    can analyse it and suggest further improvements.
    *max_size* controls the longest edge of the preview in pixels (default 1200).
    """
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return [f"Error: {e}"]

    _img, img_bytes = _quick_preview(path, state, max_size)
    preview_path = _save_and_open_preview(path, img_bytes)

    return [
        f"Preview saved to: {preview_path}\nOpening in your image viewer now...",
        Image(data=img_bytes, format="jpeg"),
    ]


@mcp.tool()
def apply_adjustments(
    image_path: str,
    # Exposure
    exposure_ev: Optional[float] = None,
    black_level: Optional[float] = None,
    highlight_recovery: Optional[float] = None,
    shadow_lift: Optional[float] = None,
    # White balance
    temperature_kelvin: Optional[float] = None,
    tint: Optional[float] = None,
    # Tone
    contrast: Optional[float] = None,
    brightness: Optional[float] = None,
    highlights: Optional[float] = None,
    shadows: Optional[float] = None,
    whites: Optional[float] = None,
    blacks: Optional[float] = None,
    # Colour
    saturation: Optional[float] = None,
    vibrance: Optional[float] = None,
    # Detail
    sharpness: Optional[float] = None,
    noise_reduction: Optional[float] = None,
    # Effects
    vignette: Optional[float] = None,
    clarity: Optional[float] = None,
) -> dict:
    """Apply one or more non-destructive adjustments to an image.

    Only the parameters you supply are changed; the rest keep their current values.
    Call get_image_preview afterwards to see the result.

    Parameters
    ----------
    exposure_ev : float
        Exposure in stops (-5 to +5).  0 = no change.
    black_level : float
        Lift or crush the black point (0.0–0.5).
    highlight_recovery : float
        Recover blown highlights (0.0–1.0).
    shadow_lift : float
        Open up dark shadows (0.0–1.0).
    temperature_kelvin : float
        Colour temperature (2000–12000 K).  ~5500 K is daylight, ~3200 K is tungsten.
    tint : float
        Green (positive) / magenta (negative) tint correction (-100 to +100).
    contrast : float
        Global contrast (-100 to +100).
    brightness : float
        Global brightness (-100 to +100).
    highlights : float
        Recover or boost highlights (-100 to +100).
    shadows : float
        Open shadows (-100 to +100).
    whites : float
        White-point adjustment (-100 to +100).
    blacks : float
        Black-point adjustment (-100 to +100).
    saturation : float
        Global colour saturation (-100 to +100).  0 = original.
    vibrance : float
        Selective saturation boost for muted colours (-100 to +100).
    sharpness : float
        Sharpening strength (0 to 100).
    noise_reduction : float
        Noise reduction strength (0 to 100).
    vignette : float
        Vignette: negative values darken edges, positive lighten (-100 to +100).
    clarity : float
        Local contrast / clarity (-100 to +100).
    """
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    params = {k: v for k, v in {
        "exposure_ev": exposure_ev,
        "black_level": black_level,
        "highlight_recovery": highlight_recovery,
        "shadow_lift": shadow_lift,
        "temperature_kelvin": temperature_kelvin,
        "tint": tint,
        "contrast": contrast,
        "brightness": brightness,
        "highlights": highlights,
        "shadows": shadows,
        "whites": whites,
        "blacks": blacks,
        "saturation": saturation,
        "vibrance": vibrance,
        "sharpness": sharpness,
        "noise_reduction": noise_reduction,
        "vignette": vignette,
        "clarity": clarity,
    }.items() if v is not None}

    state.update(params)
    state.save()

    return {"status": "ok", "applied": list(params.keys()), "edits": state.to_dict()}


@mcp.tool()
def crop_image(
    image_path: str,
    left: Optional[float] = None,
    top: Optional[float] = None,
    right: Optional[float] = None,
    bottom: Optional[float] = None,
    aspect_ratio: Optional[str] = None,
) -> dict:
    """Crop an image.

    Provide either:
    - *left*, *top*, *right*, *bottom* as normalised coordinates (0.0–1.0), or
    - *aspect_ratio* as a string like "16:9", "4:3", "1:1", "3:2" for a centre crop.

    Both methods can be combined (apply aspect ratio first, then fine-tune coordinates).
    """
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    crop = state.crop or CropState()

    if aspect_ratio:
        try:
            w_ratio, h_ratio = (float(x) for x in aspect_ratio.split(":"))
        except ValueError:
            return {"error": f"Invalid aspect_ratio format '{aspect_ratio}'. Use e.g. '16:9'."}

        # Current crop area dimensions (in normalised space)
        cw = crop.right - crop.left
        ch = crop.bottom - crop.top
        centre_x = crop.left + cw / 2
        centre_y = crop.top + ch / 2

        target_ar = w_ratio / h_ratio
        current_ar = cw / ch if ch > 0 else 1.0

        if current_ar > target_ar:
            new_w = ch * target_ar
            new_h = ch
        else:
            new_w = cw
            new_h = cw / target_ar

        crop.left = max(0.0, centre_x - new_w / 2)
        crop.top = max(0.0, centre_y - new_h / 2)
        crop.right = min(1.0, centre_x + new_w / 2)
        crop.bottom = min(1.0, centre_y + new_h / 2)

    if left is not None:
        crop.left = max(0.0, float(left))
    if top is not None:
        crop.top = max(0.0, float(top))
    if right is not None:
        crop.right = min(1.0, float(right))
    if bottom is not None:
        crop.bottom = min(1.0, float(bottom))

    if crop.right <= crop.left or crop.bottom <= crop.top:
        return {"error": "Invalid crop: right must be > left and bottom must be > top."}

    state.crop = crop
    state.save()

    return {"status": "ok", "crop": {
        "left": crop.left, "top": crop.top,
        "right": crop.right, "bottom": crop.bottom,
        "rotation": crop.rotation,
    }}


@mcp.tool()
def rotate_image(image_path: str, degrees: float) -> dict:
    """Rotate / straighten the image by *degrees* (clockwise).

    Positive = clockwise, negative = counter-clockwise.
    Typical use: small corrections like +0.5 or -1.2 to straighten horizons.
    Large rotations (90, 180, 270) are also supported.
    """
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    crop = state.crop or CropState()
    crop.rotation = degrees
    state.crop = crop
    state.save()

    return {"status": "ok", "rotation_degrees": degrees}


@mcp.tool()
def reset_crop(image_path: str) -> dict:
    """Remove all crop and rotation from the image, restoring the full frame."""
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    state.crop = None
    state.save()
    return {"status": "ok", "message": "Crop reset to full frame."}


@mcp.tool()
def rename_output(image_path: str, new_name: str) -> dict:
    """Set the output filename stem (without extension) for the exported image.

    For example: rename_output("/photos/DSC001.NEF", "golden_hour_lake")
    will export as golden_hour_lake.jpg (or whichever format you choose in export_image).
    """
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    # Strip any extension the user may have included
    stem = Path(new_name).stem or new_name
    state.output_name = stem
    state.save()
    return {"status": "ok", "output_name": stem}


@mcp.tool()
def reset_edits(image_path: str) -> dict:
    """Reset all adjustments, crop, and output name — back to the unedited original."""
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    sidecar = path.with_name(path.stem + ".mcp.json")
    if sidecar.exists():
        sidecar.unlink()

    return {"status": "ok", "message": "All edits reset to original."}


@mcp.tool()
def export_image(
    image_path: str,
    output_directory: Optional[str] = None,
    format: str = "jpeg",
    quality: int = 92,
    use_darktable_cli: bool = True,
    write_xmp_sidecar: bool = True,
    max_dimension: Optional[int] = None,
) -> dict:
    """Export the edited image to a file.

    Parameters
    ----------
    image_path : str
        Source image path.
    output_directory : str, optional
        Destination folder.  Defaults to the same folder as the source.
    format : str
        Output format: "jpeg", "png", "tiff" (default "jpeg").
    quality : int
        JPEG quality 1–100 (default 92).  Ignored for PNG/TIFF.
    use_darktable_cli : bool
        Try to use darktable-cli for export when available (better RAW rendering).
    write_xmp_sidecar : bool
        Write a Darktable-compatible XMP sidecar alongside the source file.
    max_dimension : int, optional
        Resize so the longest edge is at most this many pixels.
    """
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    out_dir = Path(output_directory) if output_directory else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = state.output_name or path.stem
    ext_map = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "tiff": ".tif", "tif": ".tif"}
    ext = ext_map.get(format.lower(), ".jpg")
    out_path = out_dir / (stem + ext)

    # Avoid collision
    counter = 1
    while out_path.exists():
        out_path = out_dir / f"{stem}_{counter}{ext}"
        counter += 1

    # Optionally write Darktable XMP sidecar (global + masked ops merged)
    xmp_path = None
    if write_xmp_sidecar:
        try:
            xmp_path = write_full_xmp(state)
        except Exception:
            xmp_path = None  # non-fatal

    # Try darktable-cli first for RAW files
    if use_darktable_cli and DARKTABLE_CLI and path.suffix.lower() in RAW_EXTENSIONS:
        result = _export_via_darktable_cli(path, out_path, xmp_path, quality)
        if result.get("status") == "ok":
            result["xmp_sidecar"] = str(xmp_path) if xmp_path else None
            return result

    # Fall back to rawpy + Pillow
    proc = ImageProcessor(path)
    img = proc.process(state)

    if max_dimension:
        from PIL import Image as PILImage
        img.thumbnail((max_dimension, max_dimension), PILImage.LANCZOS)

    save_kwargs: dict = {}
    if format.lower() in ("jpeg", "jpg"):
        save_kwargs = {"format": "JPEG", "quality": quality, "optimize": True}
    elif format.lower() == "png":
        save_kwargs = {"format": "PNG", "optimize": True}
    elif format.lower() in ("tiff", "tif"):
        save_kwargs = {"format": "TIFF"}

    img.save(out_path, **save_kwargs)

    return {
        "status": "ok",
        "output_path": str(out_path),
        "format": format,
        "size_bytes": out_path.stat().st_size,
        "xmp_sidecar": str(xmp_path) if xmp_path else None,
        "rendered_via": "rawpy+pillow",
    }


def _export_via_darktable_cli(
    src: Path, dst: Path, xmp: Optional[Path], quality: int,
    max_dim: Optional[int] = None,
) -> dict:
    cmd = [DARKTABLE_CLI, str(src)]
    if xmp and xmp.exists():
        cmd.append(str(xmp))
    cmd += [str(dst), "--width", str(max_dim or 0), "--height", str(max_dim or 0),
            "--quality", str(quality)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and dst.exists():
            return {
                "status": "ok",
                "output_path": str(dst),
                "rendered_via": "darktable-cli",
                "size_bytes": dst.stat().st_size,
            }
        return {"status": "error", "stderr": result.stderr[:800]}
    except Exception as exc:
        return {"status": "error", "exception": str(exc)}


@mcp.tool()
def get_histogram(image_path: str) -> dict:
    """Compute a brightness/channel histogram for the image (with current edits).

    Returns per-channel (R, G, B) and luminance histograms as 256-bin arrays.
    Useful for analysing exposure, clipping, and tonal distribution.
    """
    import numpy as np

    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    proc = ImageProcessor(path)
    img = proc.process(state, preview_size=800)
    arr = np.array(img, dtype=np.uint8)

    def _hist(channel: np.ndarray) -> list[int]:
        counts, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))
        return counts.tolist()

    r_hist = _hist(arr[..., 0])
    g_hist = _hist(arr[..., 1])
    b_hist = _hist(arr[..., 2])
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    lum_hist = _hist(lum)

    clipped_highlights = int((arr == 255).all(axis=2).sum())
    clipped_shadows = int((arr == 0).all(axis=2).sum())
    total_pixels = arr.shape[0] * arr.shape[1]

    return {
        "r": r_hist,
        "g": g_hist,
        "b": b_hist,
        "luminance": lum_hist,
        "clipped_highlights_px": clipped_highlights,
        "clipped_shadows_px": clipped_shadows,
        "total_pixels": total_pixels,
        "highlight_clip_pct": round(clipped_highlights / total_pixels * 100, 2),
        "shadow_clip_pct": round(clipped_shadows / total_pixels * 100, 2),
    }


@mcp.tool()
def make_mask(
    image_path: str,
    prompt: Optional[str] = None,
    points: Optional[list] = None,
    labels: Optional[list] = None,
    bbox: Optional[list] = None,
) -> dict:
    """Generate a SAM mask for a region in the image and save it as a PFM file.

    Provide EXACTLY ONE of the following prompt types:
    - prompt (str): Natural-language description, e.g. "the sky", "the person's face".
      Requires SAM 3 model.
    - points + labels: List of [x, y] pixel coordinates with labels 1=include, 0=exclude.
      Works with SAM 2 and SAM 3. Get pixel coords from get_image_preview.
    - bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates.

    Returns the path to the saved PFM mask file, which you can pass to apply_masked_edit.
    """
    from .sam_engine import generate_mask, model_info

    try:
        pfm_path = generate_mask(
            image_path,
            prompt=prompt,
            points=points,
            labels=labels,
            bbox=bbox,
        )
        return {
            "status": "ok",
            "mask_path": str(pfm_path),
            "model": model_info().get("model"),
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def apply_masked_edit(
    image_path: str,
    module: str,
    params: dict,
    mask_path: Optional[str] = None,
) -> dict:
    """Add a non-destructive edit to an image, optionally confined to a mask region.

    Edits are stored in a .masked_ops.json sidecar and applied when you call
    render_photo or export_image.

    Parameters
    ----------
    image_path : str
        Path to the RAW or image file.
    module : str
        Darktable module name. Supported: "exposure", "colorbalancergb", "filmicrgb",
        "shadhi", "levels", "sharpen", "temperature", "brightness_contrast",
        "vignette", "denoiseprofile".
    params : dict
        Module parameters. Examples:
          exposure:            {"exposure_ev": -1.5, "black_level": 0.0}
          colorbalancergb:     {"shadows_H": 0.08, "shadows_C": 0.03, "midtones_C": 0.02}
          filmicrgb:           {"contrast": 1.2, "latitude": 30.0, "saturation": 0.1}
          shadhi:              {"shadows": 40.0, "highlights": -30.0, "radius": 80.0}
          levels:              {"black": 0.02, "grey": 1.1, "white": 0.95}
          sharpen:             {"amount": 60.0}
          temperature:         {"temperature_kelvin": 5200.0, "tint": 0.0}
          brightness_contrast: {"brightness": 10.0, "contrast": 15.0}
          vignette:            {"amount": -30.0}
    mask_path : str, optional
        Path to a PFM file from make_mask. If omitted, the edit applies globally.

    Returns the updated list of pending ops for this image.
    """
    from .xmp_writer import _encode_module, _SUPPORTED_MODULES  # validate early

    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image not found: {image_path}"}

    try:
        _encode_module(module, params)  # validate module + params before queueing
    except ValueError as exc:
        return {"error": str(exc)}

    ops = load_masked_ops(path)
    ops.append({
        "module": module,
        "params": params,
        "pfm_path": mask_path,
    })
    save_masked_ops(path, ops)

    return {
        "status": "ok",
        "pending_ops": len(ops),
        "ops": ops,
        "note": "Call render_photo to produce a JPEG with these edits applied.",
    }


@mcp.tool()
def render_photo(
    image_path: str,
    output_path: Optional[str] = None,
    max_dimension: Optional[int] = None,
    quality: int = 92,
) -> dict:
    """Render a RAW to JPEG applying all global adjustments and masked ops.

    Builds the full XMP sidecar (global + masked ops), then invokes darktable-cli
    for high-quality RAW rendering.

    Parameters
    ----------
    image_path : str
        Source RAW or image file.
    output_path : str, optional
        Full destination path including extension.  Defaults to
        <source_stem>_rendered.jpg next to the source.
    max_dimension : int, optional
        Resize so the longest edge is at most this many pixels.
    quality : int
        JPEG quality 1–100 (default 92).
    """
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    if not DARKTABLE_CLI:
        return {"error": "darktable-cli not found. Install darktable from darktable.org."}

    dst = Path(output_path) if output_path else path.with_name(path.stem + "_rendered.jpg")
    dst.parent.mkdir(parents=True, exist_ok=True)

    xmp_path = None
    try:
        xmp_path = write_full_xmp(state)
    except Exception as exc:
        return {"error": f"XMP write failed: {exc}"}

    result = _export_via_darktable_cli(path, dst, xmp_path, quality, max_dim=max_dimension)
    if result.get("status") == "ok":
        ops = load_masked_ops(path)
        result["masked_ops_applied"] = len(ops)
        result["xmp_sidecar"] = str(xmp_path)
    return result


@mcp.tool()
def clear_masked_edits(image_path: str) -> dict:
    """Remove all pending masked edits for an image (clears the .masked_ops.json sidecar)."""
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image not found: {image_path}"}
    clear_masked_ops(path)
    return {"status": "ok", "message": "All masked edits cleared."}


@mcp.tool()
def list_masked_edits(image_path: str) -> dict:
    """Show the current pending masked edits queued for an image."""
    path = Path(image_path)
    ops = load_masked_ops(path)
    return {"image": image_path, "pending_ops": len(ops), "ops": ops}


@mcp.tool()
def batch_edit(
    folder: str,
    recipe: list,
    output_folder: str,
    max_dimension: int = 2400,
    quality: int = 92,
) -> dict:
    """Apply a recipe of masked/unmasked edits to every RAW in a folder, then export.

    recipe: list of dicts, each with keys:
      - module (str): e.g. "exposure", "colorbalancergb"
      - params (dict): module parameters
      - mask_prompt (str, optional): SAM text prompt to generate a mask for this op
      - mask_points (list, optional): [[x,y], ...] point prompts (alternative to mask_prompt)
      - mask_labels (list, optional): [1, 0, ...] labels for mask_points
      - mask_bbox (list, optional): [x1, y1, x2, y2] (alternative to mask_prompt)

    Returns a summary dict with processed count and any per-file errors.
    """
    from .sam_engine import generate_mask

    folder_path = Path(folder)
    out_path = Path(output_folder)

    if not folder_path.is_dir():
        return {"error": f"Folder not found: {folder}"}

    out_path.mkdir(parents=True, exist_ok=True)
    raw_files = [f for f in sorted(folder_path.iterdir())
                 if f.suffix.lower() in RAW_EXTENSIONS]

    if not raw_files:
        return {"error": f"No RAW files found in {folder}"}

    if not DARKTABLE_CLI:
        return {"error": "darktable-cli not found. Install darktable from darktable.org."}

    processed, errors = 0, []

    for raw in raw_files:
        try:
            _, state = _load_or_new(str(raw))
            ops = []

            for step in recipe:
                module = step.get("module")
                params = step.get("params", {})
                mask_prompt = step.get("mask_prompt")
                mask_points = step.get("mask_points")
                mask_labels = step.get("mask_labels")
                mask_bbox = step.get("mask_bbox")

                pfm_path = None
                if mask_prompt or mask_points or mask_bbox:
                    try:
                        pfm_path = str(generate_mask(
                            str(raw),
                            prompt=mask_prompt,
                            points=mask_points,
                            labels=mask_labels,
                            bbox=mask_bbox,
                        ))
                    except Exception as e:
                        errors.append({"file": raw.name, "step": module, "mask_error": str(e)})
                        pfm_path = None

                ops.append({"module": module, "params": params, "pfm_path": pfm_path})

            save_masked_ops(raw, ops)
            xmp_path = write_full_xmp(state)

            dst = out_path / (raw.stem + ".jpg")
            result = _export_via_darktable_cli(raw, dst, xmp_path, quality, max_dim=max_dimension)

            if result.get("status") == "ok":
                processed += 1
            else:
                errors.append({"file": raw.name, "error": result.get("stderr") or result.get("exception")})

        except Exception as exc:
            errors.append({"file": raw.name, "error": str(exc)})

    return {
        "status": "ok",
        "processed": processed,
        "total": len(raw_files),
        "errors": errors,
        "output_folder": str(out_path),
    }


# ---------------------------------------------------------------------------
# Darktable styles
# ---------------------------------------------------------------------------

_STYLE_DIRS = [
    Path.home() / ".config" / "darktable" / "styles",
    Path("/Applications/darktable.app/Contents/Resources/share/darktable/styles"),
]


def _find_style(style_name: str) -> Optional[Path]:
    """Search all style dirs for a .dtstyle file matching name (case-insensitive stem)."""
    for d in _STYLE_DIRS:
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix.lower() == ".dtstyle" and f.stem.lower() == style_name.lower():
                return f
    return None


@mcp.tool()
def list_styles() -> list[dict]:
    """List all available darktable styles (.dtstyle files) from user and bundle directories.

    Returns a list of dicts with keys: name, path, source ("user" or "bundle").
    Pass the name to apply_style to render a RAW with that style.
    """
    results = []
    for d in _STYLE_DIRS:
        if not d.is_dir():
            continue
        source = "user" if "config" in str(d) else "bundle"
        for f in sorted(d.iterdir()):
            if f.suffix.lower() == ".dtstyle":
                results.append({"name": f.stem, "path": str(f), "source": source})
    return results


@mcp.tool()
def apply_style(
    image_path: str,
    style_name: str,
    output_path: Optional[str] = None,
    max_dimension: Optional[int] = None,
    quality: int = 92,
) -> dict:
    """Render a RAW file with a named darktable style applied.

    Styles are colour-science presets bundled with darktable (e.g. camera input
    profiles) or created in the darktable GUI. Run list_styles() to see all names.

    Parameters
    ----------
    image_path : str
        Source RAW file.
    style_name : str
        Style filename stem, e.g. "darktable_Canon_EOS 5D Mark IV" (no .dtstyle extension).
    output_path : str, optional
        Full destination path including extension.  Defaults to
        <stem>_<style_name>.jpg next to the source.
    max_dimension : int, optional
        Resize so the longest edge is at most this many pixels.
    quality : int
        JPEG quality 1–100 (default 92).
    """
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    if not DARKTABLE_CLI:
        return {"error": "darktable-cli not found. Install darktable from darktable.org."}

    style_path = _find_style(style_name)
    if style_path is None:
        return {
            "error": f"Style '{style_name}' not found. Run list_styles() to see available styles."
        }

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in style_name)
    dst = (
        Path(output_path)
        if output_path
        else path.with_name(f"{path.stem}_{safe_name}.jpg")
    )
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [DARKTABLE_CLI, str(path), str(dst),
           "--style", str(style_path),
           "--width", str(max_dimension or 0),
           "--height", str(max_dimension or 0),
           "--quality", str(quality)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and dst.exists():
            return {
                "status": "ok",
                "output_path": str(dst),
                "style": style_name,
                "size_bytes": dst.stat().st_size,
                "rendered_via": "darktable-cli",
            }
        return {"status": "error", "stderr": result.stderr[:800]}
    except Exception as exc:
        return {"status": "error", "exception": str(exc)}


@mcp.tool()
def export_style(image_path: str, style_name: str) -> dict:
    """Save the current XMP edits for an image as a reusable darktable style.

    The style is written to ~/.config/darktable/styles/<style_name>.dtstyle.
    It can then be loaded in the darktable GUI or applied with apply_style().

    Note: this exports the global adjustments only (not masked ops), since
    darktable styles are module-based presets without geometry/mask bindings.
    """
    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    xmp_path = None
    try:
        xmp_path = write_full_xmp(state)
    except Exception as exc:
        return {"error": f"XMP write failed: {exc}"}

    styles_dir = Path.home() / ".config" / "darktable" / "styles"
    styles_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in " -_." else "_" for c in style_name)
    dst = styles_dir / f"{safe_name}.dtstyle"

    # Copy XMP as .dtstyle — darktable can import XMP-based styles
    import shutil as _shutil
    _shutil.copy2(str(xmp_path), str(dst))

    return {
        "status": "ok",
        "style_path": str(dst),
        "style_name": safe_name,
        "note": "Style saved. You can load it in darktable GUI via Styles panel.",
    }


# ---------------------------------------------------------------------------
# Mask preview overlay
# ---------------------------------------------------------------------------


@mcp.tool()
def preview_masked(
    image_path: str,
    mask_path: str,
    overlay_color: str = "red",
    alpha: float = 0.45,
    max_size: int = 1200,
) -> list:
    """Show the image with a SAM mask highlighted as a coloured overlay.

    Use this immediately after make_mask to verify the mask covers the right
    region before calling apply_masked_edit.

    Parameters
    ----------
    image_path : str
        Source image path.
    mask_path : str
        Path to a .pfm mask file from make_mask.
    overlay_color : str
        "red", "green", "blue", or "yellow" (default "red").
    alpha : float
        Overlay opacity 0.0–1.0 (default 0.45).
    max_size : int
        Longest edge of the returned preview in pixels (default 1200).
    """
    import numpy as np
    from PIL import Image as PILImage

    try:
        path, state = _load_or_new(image_path)
    except FileNotFoundError as e:
        return [f"Error: {e}"]

    mask_file = Path(mask_path)
    if not mask_file.exists():
        return [f"Error: mask file not found: {mask_path}"]

    # Load image preview
    proc = ImageProcessor(path)
    img = proc.process(state, preview_size=max_size)
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    # Read PFM mask
    with open(mask_file, "rb") as f:
        header = f.readline().strip()
        if header not in (b"Pf", b"PF"):
            return [f"Error: not a valid PFM file: {mask_path}"]
        dims = f.readline().decode().split()
        mw, mh = int(dims[0]), int(dims[1])
        scale = float(f.readline().strip())
        raw_bytes = f.read()
    mask_data = np.frombuffer(raw_bytes, dtype=np.float32).reshape(mh, mw)
    if scale < 0:
        mask_data = np.flipud(mask_data)  # PFM is bottom-up when scale < 0

    # Resize mask to match preview dimensions
    H, W = arr.shape[:2]
    from PIL import Image as _PILImage
    mask_img = _PILImage.fromarray((mask_data * 255).clip(0, 255).astype(np.uint8))
    mask_img = mask_img.resize((W, H), _PILImage.BILINEAR)
    mask_resized = np.array(mask_img, dtype=np.float32) / 255.0

    # Colour map
    color_map = {
        "red":    np.array([1.0, 0.0, 0.0]),
        "green":  np.array([0.0, 1.0, 0.0]),
        "blue":   np.array([0.0, 0.0, 1.0]),
        "yellow": np.array([1.0, 1.0, 0.0]),
    }
    color = color_map.get(overlay_color.lower(), color_map["red"])

    # Blend overlay onto image
    m = mask_resized[..., np.newaxis]  # (H, W, 1)
    blended = arr * (1.0 - m * alpha) + color * m * alpha
    blended = (blended * 255).clip(0, 255).astype(np.uint8)

    out_img = PILImage.fromarray(blended)
    buf = io.BytesIO()
    out_img.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()

    preview_out = path.with_name(path.stem + "__mask_preview.jpg")
    preview_out.write_bytes(img_bytes)

    return [
        f"Mask preview saved to: {preview_out}\nMask covers {float(mask_resized.mean()) * 100:.1f}% of the frame.",
        Image(data=img_bytes, format="jpeg"),
    ]


# ---------------------------------------------------------------------------
# Model preload
# ---------------------------------------------------------------------------


@mcp.tool()
def preload_models(include_florence: bool = False) -> dict:
    """Pre-download and warm up SAM 2.1 (and optionally Florence-2) models.

    SAM 2.1 base (~300 MB) and Florence-2 base (~230 MB) download automatically
    on first use from Ultralytics / HuggingFace Hub. Call this once before a
    batch session to avoid first-call latency.

    Parameters
    ----------
    include_florence : bool
        Also warm up Florence-2 (used for text prompts in make_mask).
        Default False — only load SAM.
    """
    from .sam_engine import preload_sam, preload_florence, model_info

    result = preload_sam()
    if include_florence:
        fl = preload_florence()
        result["florence"] = fl

    result.update(model_info())
    return result


# ---------------------------------------------------------------------------
# Copy settings
# ---------------------------------------------------------------------------


@mcp.tool()
def copy_settings(source_image_path: str, target_image_path: str) -> dict:
    """Copy all edit settings (adjustments + crop) from one image to another.

    Useful for batch-editing a set of photos shot under the same conditions.
    The output_name is NOT copied — each image keeps its own name.
    """
    try:
        src_path, src_state = _load_or_new(source_image_path)
        tgt_path, tgt_state = _load_or_new(target_image_path)
    except FileNotFoundError as e:
        return {"error": str(e)}

    from dataclasses import replace, asdict
    tgt_state.adjustments = replace(src_state.adjustments)
    tgt_state.crop = replace(src_state.crop) if src_state.crop else None
    tgt_state.save()

    return {"status": "ok", "copied_to": str(tgt_path), "edits": tgt_state.to_dict()}
