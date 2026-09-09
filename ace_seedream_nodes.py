"""ACE Seedream 5.0 Pro nodes (fal.ai): Layerize + Pro Edit.

Drop into ComfyUI custom_nodes/ (single file) and restart.
API key: node input or FAL_KEY environment variable.

Endpoints (schemas per fal llms.txt, 2026-09):
  - bytedance/seedream/v5/pro/layerize
  - bytedance/seedream/v5/pro/edit
"""

import base64
import json
import os
import time
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

FAL_LAYERIZE_URL = "https://fal.run/bytedance/seedream/v5/pro/layerize"
FAL_EDIT_URL = "https://fal.run/bytedance/seedream/v5/pro/edit"
TIMEOUT = 600


def _output_dir() -> str:
    try:
        import folder_paths
        d = folder_paths.get_output_directory()
    except Exception:
        d = os.path.join(os.getcwd(), "output")
    os.makedirs(d, exist_ok=True)
    return d


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    t = x.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _pil_to_tensor_rgb(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _placeholder(size: int = 512) -> torch.Tensor:
    return _pil_to_tensor_rgb(Image.new("RGB", (size, size), (100, 100, 100)))


def _placeholder_mask(size: int = 512) -> torch.Tensor:
    return torch.zeros((1, size, size), dtype=torch.float32)


def _tensor_to_data_uri(x: torch.Tensor) -> str:
    buf = BytesIO()
    _tensor_to_pil(x).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _get_key(api_key: str) -> str:
    key = (api_key or "").strip() or os.getenv("FAL_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "SEEDREAM ERROR: No API key in node input or FAL_KEY environment variable."
        )
    return key


def _fal_post(url: str, key: str, payload: dict) -> dict:
    resp = requests.post(
        url,
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"fal API {resp.status_code}: {resp.text[:2000]}")
    return resp.json()


def _download(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def _save_raw(raw: bytes, name: str, log: List[str]) -> Optional[str]:
    try:
        p = os.path.join(_output_dir(), name)
        with open(p, "wb") as f:
            f.write(raw)
        log.append(f"raw saved -> {p}")
        return p
    except Exception as e:
        log.append(f"raw save failed ({name}): {e}")
        return None


def _stack_rgb(pils: List[Image.Image]) -> torch.Tensor:
    if not pils:
        return _placeholder()
    w = min(p.width for p in pils)
    h = min(p.height for p in pils)
    out = []
    for p in pils:
        if p.size != (w, h):
            p = p.resize((w, h), Image.LANCZOS)
        out.append(_pil_to_tensor_rgb(p))
    return torch.cat(out, 0)


def _stack_alpha(pils: List[Image.Image]) -> torch.Tensor:
    if not pils:
        return _placeholder_mask()
    w = min(p.width for p in pils)
    h = min(p.height for p in pils)
    out = []
    for p in pils:
        if p.size != (w, h):
            p = p.resize((w, h), Image.LANCZOS)
        rgba = p.convert("RGBA")
        a = np.asarray(rgba, dtype=np.float32)[..., 3] / 255.0
        out.append(torch.from_numpy(a)[None, ...])
    return torch.cat(out, 0)


class AceSeedreamLayerize:
    """Split an image into background + transparent layers (2-17, model-decided).

    fal exposes NO layer-count parameter - steer the count/content via the prompt
    (e.g. 'separate the bottle, each flower, and the background').
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "password": True, "tooltip": "fal API key (or set FAL_KEY env)"},
                ),
                "image": ("IMAGE", {"tooltip": "Image to decompose"}),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Which elements to separate. Empty = auto. The only way to influence layer count.",
                    },
                ),
                "image_size": (
                    ["auto", "auto_1K", "auto_1.5K", "auto_2K"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "enhance_prompt_mode": (["standard", "fast"], {"default": "standard"}),
                "save_raw": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Save untouched layer PNGs (with alpha) to the output folder"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("base_image", "layers", "layer_masks", "layer_info", "raw_paths", "operation_log")
    FUNCTION = "run"
    CATEGORY = "Ace_Seedream"
    DESCRIPTION = (
        "Seedream 5.0 Pro Layerize via fal: base image + up to 16 transparent layers. "
        "Layer count is model-decided (no API parameter); steer via prompt."
    )

    def run(
        self,
        api_key: str,
        image: torch.Tensor,
        prompt: str = "",
        image_size: str = "auto",
        enable_safety_checker: bool = True,
        enhance_prompt_mode: str = "standard",
        save_raw: bool = True,
        **kwargs,
    ):
        key = _get_key(api_key)
        log: List[str] = []

        payload = {
            "image_url": _tensor_to_data_uri(image),
            "image_size": image_size,
            "enable_safety_checker": enable_safety_checker,
            "enhance_prompt_mode": enhance_prompt_mode,
        }
        if prompt.strip():
            payload["prompt"] = prompt.strip()

        t0 = time.time()
        data = _fal_post(FAL_LAYERIZE_URL, key, payload)
        log.append(f"API call completed in {time.time() - t0:.1f}s")

        layers = data.get("layers") or []
        if not layers:
            raise RuntimeError(f"No layers returned. Response: {json.dumps(data)[:1500]}")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        base_pil: Optional[Image.Image] = None
        layer_pils: List[Image.Image] = []
        info: List[dict] = []
        raw_paths: List[str] = []

        for idx, layer in enumerate(layers):
            img_meta = layer.get("image") or {}
            url = img_meta.get("url")
            if not url:
                log.append(f"layer {idx}: no url, skipped")
                continue
            raw = _download(url)
            pil = Image.open(BytesIO(raw))
            z = layer.get("z_index", idx)
            name = layer.get("name") or ("base" if z == 0 else f"layer_{z}")
            if save_raw:
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]
                rp = _save_raw(raw, f"seedream_layer_{stamp}_z{z:02d}_{safe}.png", log)
                if rp:
                    raw_paths.append(rp)
            info.append(
                {
                    "z_index": z,
                    "name": layer.get("name"),
                    "description": layer.get("description"),
                    "bounding_box": layer.get("bounding_box"),
                    "url": url,
                }
            )
            if z == 0 and base_pil is None:
                base_pil = pil
            else:
                layer_pils.append(pil)

        log.append(f"{len(info)} layers total ({len(layer_pils)} above base)")

        base_t = _pil_to_tensor_rgb(base_pil) if base_pil is not None else _placeholder()
        layers_t = _stack_rgb(layer_pils) if layer_pils else _placeholder()
        masks_t = _stack_alpha(layer_pils) if layer_pils else _placeholder_mask()

        return (
            base_t,
            layers_t,
            masks_t,
            json.dumps(info, indent=2),
            "\n".join(raw_paths),
            "\n".join(log),
        )


class AceSeedreamProEdit:
    """Region-precise editing with up to 10 reference images."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
            "output_format": (["png", "jpeg"], {"default": "png"}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
            "save_raw": (
                "BOOLEAN",
                {"default": True, "tooltip": "Save untouched result files to the output folder"},
            ),
        }
        for i in range(1, 11):
            opt[f"image_{i}"] = ("IMAGE", {"forceInput": False})
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "password": True, "tooltip": "fal API key (or set FAL_KEY env)"},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    "STRING",
                    {
                        "default": "auto_2K",
                        "tooltip": "Size preset (default auto_2K) or WIDTHxHEIGHT e.g. 2048x1152. Total pixels 1024x1024..2048x2048.",
                    },
                ),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "raw_paths", "operation_log")
    FUNCTION = "run"
    CATEGORY = "Ace_Seedream"
    DESCRIPTION = "Seedream 5.0 Pro Edit via fal: prompt + up to 10 reference images."

    def run(
        self,
        api_key: str,
        prompt: str,
        image_size: str = "auto_2K",
        num_images: int = 1,
        output_format: str = "png",
        enable_safety_checker: bool = True,
        save_raw: bool = True,
        **kwargs,
    ):
        key = _get_key(api_key)
        log: List[str] = []

        image_urls = []
        for i in range(1, 11):
            im = kwargs.get(f"image_{i}")
            if isinstance(im, torch.Tensor):
                image_urls.append(_tensor_to_data_uri(im))
        if not image_urls:
            raise RuntimeError("Connect at least one image (image_1..image_10).")
        if not prompt.strip():
            raise RuntimeError("Prompt is required.")

        size = image_size.strip() or "auto_2K"
        if "x" in size and size.replace("x", "").isdigit():
            w, h = size.lower().split("x")
            size_val = {"width": int(w), "height": int(h)}
        else:
            size_val = size

        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": size_val,
            "num_images": int(num_images),
            "output_format": output_format,
            "enable_safety_checker": enable_safety_checker,
        }

        t0 = time.time()
        data = _fal_post(FAL_EDIT_URL, key, payload)
        log.append(f"API call completed in {time.time() - t0:.1f}s ({len(image_urls)} input image(s))")

        images = data.get("images") or []
        if not images:
            raise RuntimeError(f"No images returned. Response: {json.dumps(data)[:1500]}")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        pils: List[Image.Image] = []
        raw_paths: List[str] = []
        for idx, meta in enumerate(images):
            url = meta.get("url")
            if not url:
                continue
            raw = _download(url)
            pils.append(Image.open(BytesIO(raw)))
            if save_raw:
                ext = "png" if output_format == "png" else "jpg"
                rp = _save_raw(raw, f"seedream_edit_{stamp}_{idx+1:02d}.{ext}", log)
                if rp:
                    raw_paths.append(rp)

        log.append(f"{len(pils)} image(s) generated")
        return (_stack_rgb(pils), "\n".join(raw_paths), "\n".join(log))


NODE_CLASS_MAPPINGS = {
    "AceSeedreamLayerize": AceSeedreamLayerize,
    "AceSeedreamProEdit": AceSeedreamProEdit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AceSeedreamLayerize": "ACE Seedream Layerize",
    "AceSeedreamProEdit": "ACE Seedream Pro Edit",
}
