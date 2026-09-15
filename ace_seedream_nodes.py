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


import threading

_rate_lock = threading.Lock()
_last_call_time = 0.0


def _throttled_call(func, *args, **kwargs):
    """Global 1 call/second throttle across threads."""
    global _last_call_time
    with _rate_lock:
        now = time.time()
        wait = max(0.0, 1.0 - (now - _last_call_time))
        _last_call_time = now + wait
    if wait > 0:
        time.sleep(wait)
    return func(*args, **kwargs)


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


def _tensor_frames_to_data_uris(x: torch.Tensor) -> List[str]:
    """Explode a [B,H,W,C] batch (or single image) into one data URI per frame."""
    t = x.detach().cpu()
    if t.ndim == 3:
        t = t[None, ...]
    return [_tensor_to_data_uri(t[i]) for i in range(t.shape[0])]


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


def _pil_rgb_over_white(pil: Image.Image) -> Image.Image:
    rgba = pil.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def _pil_alpha_mask(pil: Image.Image) -> torch.Tensor:
    a = np.asarray(pil.convert("RGBA"), dtype=np.float32)[..., 3] / 255.0
    return torch.from_numpy(a)[None, ...]


def _bbox_ltrb(bbox, canvas_w: int, canvas_h: int) -> Optional[List[int]]:
    """Parse API bounding_box to [left, top, right, bottom] pixels, or None.
    Official format: {"absolute": [l,t,r,b], "normalized": [0-1000 l,t,r,b]}."""
    if isinstance(bbox, dict):
        if isinstance(bbox.get("absolute"), (list, tuple)) and len(bbox["absolute"]) >= 4:
            return [int(round(v)) for v in bbox["absolute"][:4]]
        if isinstance(bbox.get("normalized"), (list, tuple)) and len(bbox["normalized"]) >= 4:
            nl, nt, nr, nb = bbox["normalized"][:4]
            return [
                int(round(nl / 1000.0 * canvas_w)),
                int(round(nt / 1000.0 * canvas_h)),
                int(round(nr / 1000.0 * canvas_w)),
                int(round(nb / 1000.0 * canvas_h)),
            ]
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [int(round(v)) for v in bbox[:4]]
    return None


def _bbox_place(pil: Image.Image, bbox, canvas_w: int, canvas_h: int, log: List[str], tag: str) -> Image.Image:
    """Place a cropped RGBA layer at its bounding_box position on a full-size transparent canvas."""
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    rgba = pil.convert("RGBA")
    try:
        if rgba.size == (canvas_w, canvas_h):
            # layer already full-canvas: alpha carries the position
            return rgba.copy()
        ltrb = _bbox_ltrb(bbox, canvas_w, canvas_h)
        if ltrb is None:
            raise ValueError(f"unrecognized bbox: {bbox!r}")
        left, top, right, bottom = ltrb
        w, h = right - left, bottom - top
        if w <= 0 or h <= 0:
            raise ValueError(f"degenerate bbox: {ltrb}")
        if (w, h) != rgba.size:
            rgba = rgba.resize((w, h), Image.LANCZOS)
        canvas.paste(rgba, (left, top))  # no mask: empty canvas, plain copy keeps true alpha
    except Exception as e:
        log.append(f"{tag}: bbox placement failed ({e}); layer centered instead.")
        canvas.paste(rgba, ((canvas_w - rgba.width) // 2, (canvas_h - rgba.height) // 2))
    return canvas


N_SOCKETS = 12  # layer sockets per Layerize node (API max is 16; extras go to raw_paths/layer_info)
FABRIC_SLOTS = 8  # Compositor V4 Config has 8 image inputs: base -> image1, layer_1..7 -> image2..8


def _num_layers_steer(prompt: str, num_layers: int, log: List[str]) -> str:
    """Inject layer-count guidance into the prompt. No provider exposes a count parameter."""
    p = (prompt or "").strip()
    if num_layers > 0:
        steer = f"Decompose the image into exactly {num_layers} separate layers."
        p = (p + " " + steer).strip() if p else steer
        log.append(
            f"num_layers={num_layers}: prompt-steered only (no API parameter exists); model may deviate."
        )
    return p


def _fabric_data(
    canvas_w: int,
    canvas_h: int,
    padding: int,
    placed: bool,
    layers_geo: List[Optional[List[int]]],
    layer_sizes: List[Tuple[int, int]],
    log: List[str],
) -> str:
    """Build Compositor V4 fabricData JSON: base in slot 1, layer_1..7 in slots 2..8.
    Positions = padding + bbox left/top. With place_on_canvas ON layers are full-canvas,
    so every slot sits at (padding, padding) at canvas size."""
    def entry(left, top, w, h, sx=1.0, sy=1.0):
        # fabric.js semantics: displayed size = natural size (xwidth/xheight) * scale
        return {
            "left": left, "top": top, "scaleX": round(sx, 6), "scaleY": round(sy, 6),
            "angle": 0, "flipX": False, "flipY": False, "originX": "left", "originY": "top",
            "xwidth": w, "xheight": h, "skewY": 0, "skewX": 0, "opacity": 1,
            "visible": True, "selectable": True, "evented": True,
        }

    transforms = [entry(padding, padding, canvas_w, canvas_h)]  # base, slot 1
    n_layers = min(len(layer_sizes), FABRIC_SLOTS - 1)
    for i in range(n_layers):
        if placed:
            transforms.append(entry(padding, padding, canvas_w, canvas_h))
        else:
            geo = layers_geo[i] if i < len(layers_geo) else None
            w, h = layer_sizes[i]
            if geo is not None and w > 0 and h > 0:
                bw, bh = geo[2] - geo[0], geo[3] - geo[1]
                transforms.append(
                    entry(padding + geo[0], padding + geo[1], w, h,
                          sx=bw / float(w), sy=bh / float(h))
                )
            else:
                transforms.append(entry(padding, padding, w, h))
    while len(transforms) < FABRIC_SLOTS:
        transforms.append(None)
    transforms.append(None)  # 9th trailing slot as observed in Compositor state
    if len(layer_sizes) > FABRIC_SLOTS - 1:
        log.append(
            f"fabric data covers base + {FABRIC_SLOTS - 1} layers (Compositor has {FABRIC_SLOTS} slots); "
            f"{len(layer_sizes) - (FABRIC_SLOTS - 1)} layer(s) not included."
        )
    bboxes = [
        {
            "left": t["left"],
            "top": t["top"],
            "xwidth": round(t["xwidth"] * t["scaleX"], 4),
            "xheight": round(t["xheight"] * t["scaleY"], 4),
        }
        if isinstance(t, dict) else None
        for t in transforms
    ]
    return json.dumps(
        {
            "transforms": transforms,
            "bboxes": bboxes,
            "imageNames": [None] * len(transforms),
            "imagePositions": list(range(len(transforms))),
            "maskStates": [True] * len(transforms),
            "applyMaskInConfig": True,
            "snapEnabled": False,
            "gridSize": 1,
            "width": canvas_w,
            "height": canvas_h,
            "padding": padding,
            "backgroundColor": "rgba(0,0,0,0.2)",
            "foregroundImageName": None,
        }
    )


def _per_layer_outputs(layer_pils: List[Image.Image], log: List[str], n: int = N_SOCKETS):
    """Individual native-size outputs: RGB composited over white + true alpha masks."""
    imgs: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for i in range(n):
        if i < len(layer_pils):
            p = layer_pils[i]
            imgs.append(_pil_to_tensor_rgb(_pil_rgb_over_white(p)))
            masks.append(_pil_alpha_mask(p))
        else:
            imgs.append(_placeholder(64))
            masks.append(_placeholder_mask(64))
    if len(layer_pils) > n:
        log.append(
            f"{len(layer_pils) - n} extra layer(s) beyond {n} sockets - full set in raw_paths/layer_info"
        )
    return imgs, masks


_LAYER_RETURN_TYPES = (
    ("IMAGE",) + ("IMAGE",) * N_SOCKETS + ("MASK",) * N_SOCKETS + ("STRING", "STRING", "STRING", "STRING")
)
_LAYER_RETURN_NAMES = tuple(
    ["base_image"]
    + [f"layer_{i}" for i in range(1, N_SOCKETS + 1)]
    + [f"mask_{i}" for i in range(1, N_SOCKETS + 1)]
    + ["compositor_fabric_data", "layer_info", "raw_paths", "operation_log"]
)


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
                "num_layers": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": N_SOCKETS,
                        "tooltip": "0 = model decides. Otherwise injected into the prompt as guidance - the API has NO layer-count parameter, so this steers but does not guarantee.",
                    },
                ),
                "fabric_padding": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 1024,
                        "tooltip": "Must match the Compositor Config padding. Used only for the compositor_fabric_data output.",
                    },
                ),
                "place_on_canvas": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Place each layer at its bounding_box position on a full-canvas transparent frame (z-ordered). Stack layer_1..N over base_image to reconstruct the original; off = cropped native-size layers for Compositor (wire compositor_fabric_data for positions).",
                    },
                ),
                "compare_modes": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Run BOTH standard and fast once and put both base images side by side in base_image (frame 1 = standard, frame 2 = fast) with timings in the log. Layers/masks come from the mode selected above. Costs two generations.",
                    },
                ),
            },
        }

    RETURN_TYPES = _LAYER_RETURN_TYPES
    RETURN_NAMES = _LAYER_RETURN_NAMES
    FUNCTION = "run"
    CATEGORY = "Ace_Seedream"
    DESCRIPTION = (
        "Seedream 5.0 Pro Layerize via fal: base image + up to 10 individual native-size layers "
        "(RGB over white) with true alpha masks. Layer count is model-decided; steer via prompt."
    )

    def _call(self, key: str, image: torch.Tensor, prompt: str, image_size: str,
              enable_safety_checker: bool, mode: str, log: List[str]) -> Tuple[dict, float]:
        payload = {
            "image_url": _tensor_to_data_uri(image),
            "image_size": image_size,
            "enable_safety_checker": enable_safety_checker,
            "enhance_prompt_mode": mode,
        }
        if prompt.strip():
            payload["prompt"] = prompt.strip()
        t0 = time.time()
        data = _fal_post(FAL_LAYERIZE_URL, key, payload)
        dt = time.time() - t0
        log.append(f"[{mode}] API call completed in {dt:.1f}s")
        return data, dt

    def run(
        self,
        api_key: str,
        image: torch.Tensor,
        prompt: str = "",
        image_size: str = "auto",
        enable_safety_checker: bool = True,
        enhance_prompt_mode: str = "standard",
        save_raw: bool = True,
        compare_modes: bool = False,
        place_on_canvas: bool = True,
        num_layers: int = 0,
        fabric_padding: int = 100,
        **kwargs,
    ):
        key = _get_key(api_key)
        log: List[str] = []
        prompt = _num_layers_steer(prompt, num_layers, log)

        compare_base: Optional[Image.Image] = None
        if compare_modes:
            other = "fast" if enhance_prompt_mode == "standard" else "standard"
            data_a, dt_a = self._call(key, image, prompt, image_size, enable_safety_checker, enhance_prompt_mode, log)
            data_b, dt_b = self._call(key, image, prompt, image_size, enable_safety_checker, other, log)
            log.append(f"compare: {enhance_prompt_mode} {dt_a:.1f}s vs {other} {dt_b:.1f}s")
            data = data_a
            try:
                for layer in (data_b.get("layers") or []):
                    if layer.get("z_index", 1) == 0 and (layer.get("image") or {}).get("url"):
                        compare_base = Image.open(BytesIO(_download(layer["image"]["url"])))
                        break
            except Exception as e:
                log.append(f"compare base fetch failed: {e}")
        else:
            data, _ = self._call(key, image, prompt, image_size, enable_safety_checker, enhance_prompt_mode, log)

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
                layer_pils.append((z, pil, layer.get("bounding_box")))

        layer_pils.sort(key=lambda t: t[0])
        log.append(f"{len(info)} layers total ({len(layer_pils)} above base)")
        cw, ch = base_pil.size if base_pil is not None else (2048, 2048)
        layers_geo = [_bbox_ltrb(bb, cw, ch) for _, _, bb in layer_pils]
        layer_sizes = [p.size for _, p, _ in layer_pils]
        if place_on_canvas and base_pil is not None:
            layer_pils = [
                _bbox_place(p, bb, cw, ch, log, f"layer z{z}") for z, p, bb in layer_pils
            ]
            log.append(f"layers placed on {cw}x{ch} canvas by bounding_box, z-ordered")
        else:
            if place_on_canvas:
                log.append("place_on_canvas: no base image found; layers left cropped")
            layer_pils = [p for _, p, _ in layer_pils]
        fabric = _fabric_data(cw, ch, fabric_padding, place_on_canvas and base_pil is not None,
                              layers_geo, layer_sizes, log)

        if base_pil is not None and compare_base is not None:
            base_t = _stack_rgb([base_pil, compare_base])
            log.append("base_image batch: frame 1 = selected mode, frame 2 = other mode")
        elif base_pil is not None:
            base_t = _pil_to_tensor_rgb(base_pil)
        else:
            base_t = _placeholder()

        imgs, masks = _per_layer_outputs(layer_pils, log, N_SOCKETS)
        return tuple(
            [base_t] + imgs + masks
            + [fabric, json.dumps(info, indent=2), "\n".join(raw_paths), "\n".join(log)]
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
        opt["layers"] = (
            "IMAGE",
            {
                "forceInput": False,
                "tooltip": "Batch input (e.g. Layerize 'layers' output). Every image in the batch is sent as a separate reference.",
            },
        )
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
        layers_in = kwargs.get("layers")
        if isinstance(layers_in, torch.Tensor):
            image_urls.extend(_tensor_frames_to_data_uris(layers_in))
        for i in range(1, 11):
            im = kwargs.get(f"image_{i}")
            if isinstance(im, torch.Tensor):
                image_urls.extend(_tensor_frames_to_data_uris(im))
        if len(image_urls) > 10:
            log.append(
                f"{len(image_urls)} reference images collected; API uses only the LAST 10 - trimming to first 10 instead for predictability."
            )
            image_urls = image_urls[:10]
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


# =====================================================================
# BytePlus ModelArk (Ark) variants
# =====================================================================

ARK_DEFAULT_BASE = "https://ark.ap-southeast.bytepluses.com/api/v3"
ARK_DEFAULT_MODEL = "seedream-5-0-pro"


def _get_ark_key(api_key: str) -> str:
    key = (api_key or "").strip() or os.getenv("ARK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "SEEDREAM ERROR: No API key in node input or ARK_API_KEY environment variable."
        )
    return key


def _ark_post(base_url: str, key: str, payload: dict) -> dict:
    url = base_url.rstrip("/") + "/images/generations"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Ark API {resp.status_code}: {resp.text[:2000]}\n"
            "If the error mentions the model, check the exact model ID in your Ark console Model list."
        )
    return resp.json()


def _ark_item_bytes(item: dict) -> bytes:
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        return _download(item["url"])
    raise RuntimeError(f"Ark item has neither url nor b64_json: {list(item.keys())}")


class AceSeedreamLayerizeArk:
    """Layer decomposition via BytePlus ModelArk (layer_decomposition=true)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "password": True, "tooltip": "Ark API key (or set ARK_API_KEY env)"},
                ),
                "image": ("IMAGE", {"tooltip": "Image to decompose"}),
                "model": (
                    "STRING",
                    {"default": ARK_DEFAULT_MODEL, "tooltip": "Exact model ID from your Ark console Model list"},
                ),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Which elements to separate. Empty = auto."},
                ),
            },
            "optional": {
                "num_layers": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": N_SOCKETS,
                        "tooltip": "0 = model decides. Otherwise injected into the prompt as guidance - the API has NO layer-count parameter, so this steers but does not guarantee.",
                    },
                ),
                "fabric_padding": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 1024,
                        "tooltip": "Must match the Compositor Config padding. Used only for the compositor_fabric_data output.",
                    },
                ),
                "size": ("STRING", {"default": "2K", "tooltip": "e.g. 2K, 4K or 2048x2048"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF, "tooltip": "-1 = omit"}),
                "watermark": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "BytePlus visible AI-generated watermark on outputs"},
                ),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "base_url": ("STRING", {"default": ARK_DEFAULT_BASE}),
                "save_raw": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Save untouched layer files to the output folder"},
                ),
                "place_on_canvas": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Place each layer at its bounding_box position on a full-canvas transparent frame (z-ordered). Stack layer_1..N over base_image to reconstruct the original; off = cropped native-size layers.",
                    },
                ),
            },
        }

    RETURN_TYPES = _LAYER_RETURN_TYPES
    RETURN_NAMES = _LAYER_RETURN_NAMES
    FUNCTION = "run"
    CATEGORY = "Ace_Seedream"
    DESCRIPTION = (
        "Seedream layer decomposition via BytePlus Ark: base + up to 10 individual native-size "
        "layers with alpha masks. Layer count is model-decided; num_layers steers via prompt."
    )

    def run(
        self,
        api_key: str,
        image: torch.Tensor,
        model: str = ARK_DEFAULT_MODEL,
        prompt: str = "",
        num_layers: int = 0,
        size: str = "2K",
        seed: int = -1,
        watermark: bool = False,
        response_format: str = "url",
        base_url: str = ARK_DEFAULT_BASE,
        save_raw: bool = True,
        place_on_canvas: bool = True,
        fabric_padding: int = 100,
        **kwargs,
    ):
        key = _get_ark_key(api_key)
        log: List[str] = []

        p = _num_layers_steer(prompt, num_layers, log)

        payload = {
            "model": model.strip(),
            "image": [_tensor_to_data_uri(image)],
            "layer_decomposition": True,
            "size": size.strip() or "2K",
            "response_format": response_format,
            "watermark": watermark,
        }
        if p:
            payload["prompt"] = p
        if seed >= 0:
            payload["seed"] = seed

        t0 = time.time()
        data = _ark_post(base_url, key, payload)
        log.append(f"Ark call completed in {time.time() - t0:.1f}s")

        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"No data returned. Response: {json.dumps(data)[:1500]}")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        base_pil: Optional[Image.Image] = None
        layer_pils: List[Image.Image] = []
        info: List[dict] = []
        raw_paths: List[str] = []

        for idx, item in enumerate(items):
            raw = _ark_item_bytes(item)
            pil = Image.open(BytesIO(raw))
            z = item.get("z_index", idx)
            name = item.get("name") or ("base" if z == 0 else f"layer_{z}")
            if save_raw:
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]
                rp = _save_raw(raw, f"seedream_ark_layer_{stamp}_z{z:02d}_{safe}.png", log)
                if rp:
                    raw_paths.append(rp)
            info.append(
                {
                    "z_index": z,
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "bounding_box": item.get("bounding_box"),
                    "size": item.get("size"),
                    "url": item.get("url"),
                }
            )
            if z == 0 and base_pil is None:
                base_pil = pil
            else:
                layer_pils.append((z, pil, item.get("bounding_box")))

        layer_pils.sort(key=lambda t: t[0])
        log.append(f"{len(info)} items total ({len(layer_pils)} above base)")
        cw, ch = base_pil.size if base_pil is not None else (2048, 2048)
        layers_geo = [_bbox_ltrb(bb, cw, ch) for _, _, bb in layer_pils]
        layer_sizes = [pp.size for _, pp, _ in layer_pils]
        if place_on_canvas and base_pil is not None:
            layer_pils = [
                _bbox_place(pp, bb, cw, ch, log, f"layer z{z}") for z, pp, bb in layer_pils
            ]
            log.append(f"layers placed on {cw}x{ch} canvas by bounding_box, z-ordered")
        else:
            if place_on_canvas:
                log.append("place_on_canvas: no base image found; layers left cropped")
            layer_pils = [pp for _, pp, _ in layer_pils]
        fabric = _fabric_data(cw, ch, fabric_padding, place_on_canvas and base_pil is not None,
                              layers_geo, layer_sizes, log)
        if data.get("usage"):
            log.append(f"usage: {json.dumps(data['usage'])}")

        base_t = _pil_to_tensor_rgb(base_pil) if base_pil is not None else _placeholder()
        imgs, masks = _per_layer_outputs(layer_pils, log, N_SOCKETS)
        return tuple(
            [base_t] + imgs + masks
            + [fabric, json.dumps(info, indent=2), "\n".join(raw_paths), "\n".join(log)]
        )


class AceSeedreamProEditArk:
    """Ark edit with nano-banana-style parallel slots (5 outputs)."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "size": ("STRING", {"default": "2K", "tooltip": "e.g. 2K, 4K or 2048x2048"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF, "tooltip": "-1 = omit; slot N uses seed+N-1"}),
            "watermark": ("BOOLEAN", {"default": False}),
            "response_format": (["url", "b64_json"], {"default": "url"}),
            "sequential_image_generation": (["disabled", "auto"], {"default": "disabled"}),
            "max_images": (
                "INT",
                {"default": 1, "min": 1, "max": 15, "tooltip": "Only used when sequential_image_generation=auto"},
            ),
            "base_url": ("STRING", {"default": ARK_DEFAULT_BASE}),
            "save_raw": ("BOOLEAN", {"default": True}),
        }
        for i in range(1, 6):
            opt[f"source_image_{i}"] = ("IMAGE", {"forceInput": False})
            opt[f"ref_a_{i}"] = ("IMAGE", {"forceInput": False})
            opt[f"ref_b_{i}"] = ("IMAGE", {"forceInput": False})
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "password": True, "tooltip": "Ark API key (or set ARK_API_KEY env)"},
                ),
                "model": (
                    "STRING",
                    {"default": ARK_DEFAULT_MODEL, "tooltip": "Exact model ID from your Ark console Model list"},
                ),
                "in_parallel": ("INT", {"default": 1, "min": 1, "max": 5}),
                "parallels_share_inputs": ("BOOLEAN", {"default": True}),
                "prompt_1": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 1"}),
                "prompt_2": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 2"}),
                "prompt_3": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 3"}),
                "prompt_4": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 4"}),
                "prompt_5": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 5"}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("result_1", "result_2", "result_3", "result_4", "result_5", "operation_log", "raw_paths")
    FUNCTION = "run"
    CATEGORY = "Ace_Seedream"
    DESCRIPTION = "Seedream editing via BytePlus Ark with up to 5 parallel slots."

    def _slot_call(
        self,
        base_url: str,
        key: str,
        model: str,
        prompt: str,
        images: List[torch.Tensor],
        size: str,
        seed: int,
        watermark: bool,
        response_format: str,
        seq: str,
        max_images: int,
        save_raw: bool,
        slot_id: int,
    ) -> Tuple[torch.Tensor, str, List[str]]:
        log: List[str] = []
        payload = {
            "model": model,
            "prompt": prompt,
            "image": [_tensor_to_data_uri(im) for im in images],
            "size": size,
            "response_format": response_format,
            "watermark": watermark,
            "sequential_image_generation": seq,
        }
        if seq == "auto":
            payload["sequential_image_generation_options"] = {"max_images": max_images}
        if seed >= 0:
            payload["seed"] = seed + slot_id - 1

        t0 = time.time()
        data = _throttled_call(_ark_post, base_url, key, payload)
        log.append(f"Slot {slot_id}: Ark call completed in {time.time() - t0:.1f}s")

        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"Slot {slot_id}: no data. Response: {json.dumps(data)[:1000]}")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        pils: List[Image.Image] = []
        raw_paths: List[str] = []
        for idx, item in enumerate(items):
            raw = _ark_item_bytes(item)
            pils.append(Image.open(BytesIO(raw)))
            if save_raw:
                rp = _save_raw(raw, f"seedream_ark_edit_{stamp}_s{slot_id}_{idx+1:02d}.png", log)
                if rp:
                    raw_paths.append(rp)
        if data.get("usage"):
            log.append(f"Slot {slot_id}: usage {json.dumps(data['usage'])}")
        return _stack_rgb(pils), "\n".join(log), raw_paths

    def run(
        self,
        api_key: str,
        model: str = ARK_DEFAULT_MODEL,
        in_parallel: int = 1,
        parallels_share_inputs: bool = True,
        prompt_1: str = "",
        prompt_2: str = "",
        prompt_3: str = "",
        prompt_4: str = "",
        prompt_5: str = "",
        size: str = "2K",
        seed: int = -1,
        watermark: bool = False,
        response_format: str = "url",
        sequential_image_generation: str = "disabled",
        max_images: int = 1,
        base_url: str = ARK_DEFAULT_BASE,
        save_raw: bool = True,
        **kwargs,
    ):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        key = _get_ark_key(api_key)
        prompts = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5]

        def slot_images(i: int) -> List[torch.Tensor]:
            idx = 1 if parallels_share_inputs else i
            out = []
            for name in (f"source_image_{idx}", f"ref_a_{idx}", f"ref_b_{idx}"):
                im = kwargs.get(name)
                if isinstance(im, torch.Tensor):
                    out.append(im)
            return out

        slots = []
        for i in range(1, in_parallel + 1):
            p = (prompts[i - 1] or "").strip() or (prompt_1 if parallels_share_inputs else "")
            if not p:
                raise RuntimeError(f"Slot {i}: prompt required.")
            imgs = slot_images(i)
            if not imgs:
                raise RuntimeError(f"Slot {i}: connect at least one image (source/ref_a/ref_b).")
            slots.append((i, p, imgs))

        results = {}
        logs: List[str] = []
        all_raw: List[str] = []
        with ThreadPoolExecutor(max_workers=min(in_parallel, 5)) as ex:
            futs = {
                ex.submit(
                    self._slot_call,
                    base_url, key, model.strip(), p, imgs,
                    size.strip() or "2K", seed, watermark, response_format,
                    sequential_image_generation, max_images, save_raw, sid,
                ): sid
                for sid, p, imgs in slots
            }
            failed = []
            for fut in as_completed(futs):
                sid = futs[fut]
                try:
                    results[sid] = fut.result()
                except Exception as e:
                    failed.append((sid, str(e)))
            if failed:
                raise RuntimeError(
                    "Parallel execution failed: " + "; ".join(f"Slot {s}: {m}" for s, m in failed)
                )

        out: List[torch.Tensor] = []
        for i in range(1, in_parallel + 1):
            t, m, rp = results[i]
            out.append(t)
            logs.append(m)
            all_raw.extend(rp)
        while len(out) < 5:
            out.append(_placeholder())

        return tuple(out + ["\n".join(logs), "\n".join(all_raw)])


NODE_CLASS_MAPPINGS = {
    "AceSeedreamLayerize": AceSeedreamLayerize,
    "AceSeedreamProEdit": AceSeedreamProEdit,
    "AceSeedreamLayerizeArk": AceSeedreamLayerizeArk,
    "AceSeedreamProEditArk": AceSeedreamProEditArk,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AceSeedreamLayerize": "ACE Seedream Layerize (fal)",
    "AceSeedreamProEdit": "ACE Seedream Pro Edit (fal)",
    "AceSeedreamLayerizeArk": "ACE Seedream Layerize (BytePlus)",
    "AceSeedreamProEditArk": "ACE Seedream Pro Edit (BytePlus)",
}
