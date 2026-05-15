"""SAM 2.1 + Florence-2 masking engine.

Two-stage text-to-mask pipeline:
  1. Florence-2 (microsoft/Florence-2-base) — open-vocabulary object detection.
     Converts a text prompt ("the sky", "her face") into bounding boxes.
  2. SAM 2.1 (sam2.1_b.pt) — converts bounding boxes (or point/box prompts
     directly) into pixel-precise binary masks.

Masks are saved as PFM files for darktable's rasterfile module.

Device selection: MPS on Apple Silicon, CUDA if available, else CPU.
Models auto-download on first call from Ultralytics / HuggingFace Hub.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

_DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

MASK_DIR = Path.home() / ".darktable-mcp" / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SAM 2.1 model
# ---------------------------------------------------------------------------

_SAM_MODEL_NAME = "sam2.1_b.pt"   # ultralytics SAM 2.1 base — verified filename
_sam_model = None


def _get_sam():
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM
        _sam_model = SAM(_SAM_MODEL_NAME)
    return _sam_model


def _run_sam(image_path: str, **kwargs) -> np.ndarray:
    """Run SAM inference and return the first mask as a float32 HxW array."""
    model = _get_sam()
    results = model(image_path, device=_DEVICE, **kwargs)
    if not results or results[0].masks is None:
        raise ValueError(
            "SAM returned no masks — try a more specific prompt, "
            "different point placement, or a tighter bounding box."
        )
    return results[0].masks.data[0].cpu().numpy().astype(np.float32)


def mask_from_points(image_path: str, points: list[list[int]], labels: list[int]) -> np.ndarray:
    """SAM mask from point prompts.
    points: [[x, y], ...] in pixel coords  labels: 1=include, 0=exclude
    """
    return _run_sam(image_path, points=points, labels=labels)


def mask_from_box(image_path: str, bbox: list[int]) -> np.ndarray:
    """SAM mask from a bounding box [x1, y1, x2, y2] in pixel coords."""
    return _run_sam(image_path, bboxes=[bbox])


# ---------------------------------------------------------------------------
# Florence-2 text → bounding boxes
# ---------------------------------------------------------------------------

_FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
_florence_model = None
_florence_processor = None


def _get_florence():
    global _florence_model, _florence_processor
    if _florence_model is None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        _florence_processor = AutoProcessor.from_pretrained(
            _FLORENCE_MODEL_ID, trust_remote_code=True
        )
        # MPS doesn't support float16 for all ops in Florence-2 — use float32
        dtype = torch.float32
        _florence_model = AutoModelForCausalLM.from_pretrained(
            _FLORENCE_MODEL_ID, trust_remote_code=True, torch_dtype=dtype
        ).to(_DEVICE)
        _florence_model.eval()
    return _florence_model, _florence_processor


def boxes_from_text(image_path: str, text: str) -> list[list[int]]:
    """Detect regions matching *text* in the image using Florence-2.

    Returns a list of [x1, y1, x2, y2] bounding boxes in pixel coords.
    Raises ValueError if nothing is detected.
    """
    from PIL import Image as PILImage

    model, processor = _get_florence()
    img = PILImage.open(image_path).convert("RGB")
    W, H = img.size

    task = "<OPEN_VOCABULARY_DETECTION>"
    prompt = task + text
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=1024, num_beams=3, do_sample=False
        )

    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw_output,
        task=task,
        image_size=(W, H),
    )
    bboxes = parsed.get(task, {}).get("bboxes", [])
    if not bboxes:
        raise ValueError(
            f"Florence-2 found no region matching '{text}'. "
            "Try a more specific description, or use points/bbox instead."
        )
    return [[int(x) for x in b] for b in bboxes]


def mask_from_text(image_path: str, text: str) -> np.ndarray:
    """Generate a mask from a natural-language description.

    Florence-2 detects bounding boxes matching the text, then SAM 2.1
    produces pixel-precise masks. If multiple boxes are found, their
    masks are union-merged.
    """
    bboxes = boxes_from_text(image_path, text)
    masks = [mask_from_box(image_path, b) for b in bboxes]
    if len(masks) == 1:
        return masks[0]
    # Union: take max across all masks so any detected region is included
    return np.max(np.stack(masks, axis=0), axis=0)


# ---------------------------------------------------------------------------
# PFM writer
# ---------------------------------------------------------------------------

def write_pfm(mask: np.ndarray, out_path: str | Path) -> Path:
    """Write a grayscale PFM file in the format darktable's rasterfile module reads.

    PFM spec: "Pf" header, (width height) line, scale=-1.0 (little-endian),
    then raw float32 pixels stored bottom-up.
    """
    out_path = Path(out_path)
    if mask.ndim != 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got shape {mask.shape}")
    h, w = mask.shape
    m = np.flipud(mask.astype(np.float32))
    with open(out_path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{w} {h}\n".encode())
        f.write(b"-1.0\n")
        f.write(m.tobytes())
    return out_path


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def _pfm_path(image_path: str, tag: str) -> Path:
    h = hashlib.md5(f"{image_path}:{tag}".encode()).hexdigest()[:10]
    return MASK_DIR / f"{Path(image_path).stem}_{h}.pfm"


def generate_mask(
    image_path: str,
    prompt: Optional[str] = None,
    points: Optional[list[list[int]]] = None,
    labels: Optional[list[int]] = None,
    bbox: Optional[list[int]] = None,
) -> Path:
    """Generate a SAM mask and save it as a PFM file.

    Provide EXACTLY ONE prompt type:
    - prompt: natural-language text (uses Florence-2 + SAM 2.1)
    - points + labels: pixel coordinates (SAM 2.1 directly)
    - bbox: [x1, y1, x2, y2] bounding box (SAM 2.1 directly)

    Returns the Path to the saved .pfm file.
    """
    n = sum([prompt is not None, points is not None, bbox is not None])
    if n != 1:
        raise ValueError("Provide exactly one of: prompt, points, or bbox")

    if prompt is not None:
        tag = f"text:{prompt}"
        mask = mask_from_text(image_path, prompt)
    elif points is not None:
        if labels is None:
            labels = [1] * len(points)
        tag = f"pts:{points}"
        mask = mask_from_points(image_path, points, labels)
    else:
        tag = f"box:{bbox}"
        mask = mask_from_box(image_path, bbox)

    pfm_path = _pfm_path(image_path, tag)
    write_pfm(mask, pfm_path)
    return pfm_path


# ---------------------------------------------------------------------------
# Info / preload helpers
# ---------------------------------------------------------------------------

def model_info() -> dict:
    return {
        "sam_loaded": _sam_model is not None,
        "sam_model": _SAM_MODEL_NAME,
        "florence_loaded": _florence_model is not None,
        "florence_model": _FLORENCE_MODEL_ID,
        "device": _DEVICE,
        "mask_dir": str(MASK_DIR),
    }


def preload_sam() -> dict:
    """Download and warm up SAM 2.1. Returns info dict."""
    _get_sam()
    return {"sam": _SAM_MODEL_NAME, "device": _DEVICE, "status": "ready"}


def preload_florence() -> dict:
    """Download and warm up Florence-2. Returns info dict."""
    _get_florence()
    return {"florence": _FLORENCE_MODEL_ID, "device": _DEVICE, "status": "ready"}
