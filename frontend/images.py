"""Images in a request (spec/MODEL_VISION.md, section 2; ADR-0031, items 3 and 7).

An image arrives as a `data:` URL or base64 bytes. Pillow decodes it to 8-bit RGB; the model's processor's rule
(`smart_resize`) picks a size whose sides are multiples of 32, inside 65,536 pixels and the engine's largest image,
and the image is resized to it with bicubic resampling. It is then cut into patches of 16 x 16 pixels in the
encoder's order. The engine normalises the pixels itself.

In the prompt, the chat template writes one `<|image_pad|>` per image; `expand` repeats it once per token of the
image and notes where each image's tokens start. It does the same for a video's frame pairs (frontend/video.py),
whose pad is `<|video_pad|>`.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

PATCH = 16
FACTOR = 32
"""The patch size times the merge size: an image's sides are multiples of it."""
MIN_PIXELS = 65_536
"""`preprocessor_config.json`'s `shortest_edge`, which is a pixel count."""
MAX_SOURCE_PIXELS = 64 * 2**20
"""The largest image decoded, before resizing: a guard against decompression bombs."""
IMAGE_PAD = "<|image_pad|>"


@dataclass(frozen=True, slots=True)
class ImageSpec:
    """Patch geometry and resizing mode for a vision encoder."""

    patch: int
    merge: int
    pad_token: str
    resize: str = "smart"

    @property
    def factor(self) -> int:
        return self.patch * self.merge


QWEN_IMAGE = ImageSpec(patch=PATCH, merge=2, pad_token=IMAGE_PAD, resize="smart")
INKLING2_IMAGE = ImageSpec(patch=20, merge=2, pad_token="<|unused_200054|>", resize="pad")


class ImageError(ValueError):
    """The client's image is unreadable or not allowed."""


@dataclass(frozen=True, slots=True)
class PromptImage:
    """An image, or a video's frame pair, of the prompt: its tokens start at `first`; `pixels` holds grid_h x grid_w
    patches."""

    first: int
    grid_h: int
    grid_w: int
    pixels: bytes = field(repr=False)
    frames: int = 1
    merge: int = 2

    @property
    def tokens(self) -> int:
        return (self.grid_h // self.merge) * (self.grid_w // self.merge)


def from_data_url(url: str) -> bytes:
    """The bytes of a `data:image/...;base64,` URL. Other URLs are refused: fetching is not on (ADR-0031, item 7)."""
    if not url.startswith("data:"):
        raise ImageError("only data: URLs are read; this engine does not fetch images")
    head, sep, data = url.partition(",")
    if not sep or not head.endswith(";base64") or not head[5:].startswith("image/"):
        raise ImageError("an image URL must be data:image/<type>;base64,<data>")
    return from_base64(data)


def from_base64(data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except binascii.Error, ValueError:
        raise ImageError("the image's base64 does not decode") from None


def smart_resize(height: int, width: int, max_pixels: int, min_pixels: int = MIN_PIXELS) -> tuple[int, int]:
    """The processor's size rule (spec/MODEL_VISION.md, section 2.1), with factor 32."""
    if max(height, width) / min(height, width) > 200:
        raise ImageError(f"an image's sides may differ at most 200 times, not {height} x {width}")
    h = max(FACTOR, round(height / FACTOR) * FACTOR)
    w = max(FACTOR, round(width / FACTOR) * FACTOR)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / FACTOR) * FACTOR
        w = math.ceil(width * beta / FACTOR) * FACTOR
    return h, w


def patches(rgb: np.ndarray, patch: int = PATCH, merge: int = 2) -> bytes:
    """An H x W x 3 uint8 image, both sides multiples of `patch * merge`, as patches in merge-window order."""
    height, width, _ = rgb.shape
    gh, gw = height // patch, width // patch
    blocks = rgb.reshape(gh // merge, merge, patch, gw // merge, merge, patch, 3)
    return np.ascontiguousarray(blocks.transpose(0, 3, 1, 4, 6, 2, 5)).tobytes()


def _pad_resize(width: int, height: int, max_tokens: int, factor: int) -> tuple[int, int, int, int]:
    """Scale `(width, height)` down only when padded area exceeds `max_tokens` tokens, then pad up to `factor`."""
    aw = max(factor, -(-width // factor) * factor)
    ah = max(factor, -(-height // factor) * factor)
    max_pixels = max_tokens * factor * factor
    rw, rh = width, height
    if aw * ah > max_pixels:
        scale = math.sqrt(max_pixels / (aw * ah))
        rw = max(1, math.floor(width * scale))
        rh = max(1, math.floor(height * scale))
        aw = max(factor, -(-rw // factor) * factor)
        ah = max(factor, -(-rh // factor) * factor)
        while (aw // factor) * (ah // factor) > max_tokens:
            if rw * ah >= rh * aw and aw > factor:
                aw -= factor
                rw = min(rw, aw)
            elif ah > factor:
                ah -= factor
                rh = min(rh, ah)
            else:
                break
    return rw, rh, aw, ah


def prepare(raw: bytes, max_tokens: int, spec: ImageSpec = QWEN_IMAGE) -> tuple[int, int, bytes]:
    """An image's grid (in patches) and pixels, sized to at most `max_tokens` tokens."""
    try:
        with PILImage.open(io.BytesIO(raw)) as im:
            if im.width * im.height > MAX_SOURCE_PIXELS:
                raise ImageError(f"an image of {im.width} x {im.height} pixels is too large to read")
            rgb = im.convert("RGB")
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError) as exc:
        raise ImageError(f"the image does not decode: {exc}") from None
    if spec.resize == "pad":
        rw, rh, aw, ah = _pad_resize(rgb.width, rgb.height, max_tokens, spec.factor)
        if (rw, rh) != (rgb.width, rgb.height):
            rgb = rgb.resize((rw, rh), PILImage.Resampling.BILINEAR)
        arr = np.zeros((ah, aw, 3), dtype=np.uint8)
        arr[:rh, :rw, :] = np.asarray(rgb, dtype=np.uint8)
        return ah // spec.patch, aw // spec.patch, patches(arr, spec.patch, spec.merge)
    h, w = smart_resize(rgb.height, rgb.width, max_tokens * spec.factor * spec.factor)
    # The minimum can push a small limit over: the longer side gives way.
    while (h // spec.factor) * (w // spec.factor) > max_tokens:
        if h >= w:
            h -= spec.factor
        else:
            w -= spec.factor
    if (rgb.height, rgb.width) != (h, w):
        rgb = rgb.resize((w, h), PILImage.Resampling.BICUBIC)
    return h // spec.patch, w // spec.patch, patches(np.asarray(rgb, dtype=np.uint8), spec.patch, spec.merge)


def expand(
    ids: Sequence[int],
    pad_id: int,
    images: Sequence[tuple[int, int, bytes]],
    video_pad_id: int = -1,
    pairs: Sequence[tuple[int, int, bytes]] = (),
    *,
    image_merge: int = 2,
) -> tuple[list[int], list[PromptImage]]:
    """The prompt with each image's one pad token repeated once per token of the image, and each frame pair's one
    video pad likewise, and where each starts, in prompt order."""
    pads = [i for i, t in enumerate(ids) if t == pad_id]
    if len(pads) != len(images):
        raise ImageError(f"the rendered prompt has {len(pads)} image places for {len(images)} images")
    vpads = [i for i, t in enumerate(ids) if t == video_pad_id] if pairs else []
    if len(vpads) != len(pairs):
        raise ImageError(f"the rendered prompt has {len(vpads)} places for {len(pairs)} frame pairs")
    places = sorted(
        [(at, pad_id, 1, image_merge, im) for at, im in zip(pads, images, strict=True)]
        + [(at, video_pad_id, 2, 2, p) for at, p in zip(vpads, pairs, strict=True)]
    )
    out: list[int] = []
    placed: list[PromptImage] = []
    start = 0
    for at, pad, frames, merge, (gh, gw, pixels) in places:
        out.extend(ids[start:at])
        im = PromptImage(len(out), gh, gw, pixels, frames, merge)
        out.extend([pad] * im.tokens)
        placed.append(im)
        start = at + 1
    out.extend(ids[start:])
    return out, placed
