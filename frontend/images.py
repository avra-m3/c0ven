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


class ImageError(ValueError):
    """The client's image is unreadable or not allowed."""


@dataclass(frozen=True, slots=True)
class PromptImage:
    """An image, or a video's frame pair, of the prompt: its tokens start at `first`; `pixels` holds grid_h x grid_w
    patches of `frames` x 768 bytes."""

    first: int
    grid_h: int
    grid_w: int
    pixels: bytes = field(repr=False)
    frames: int = 1

    @property
    def tokens(self) -> int:
        return (self.grid_h // 2) * (self.grid_w // 2)


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


def patches(rgb: np.ndarray) -> bytes:
    """An H x W x 3 uint8 image, both sides multiples of 32, as patches in the encoder's order: merge windows
    row-major, each window's four patches row-major, each patch channel, row, column (section 2.2)."""
    height, width, _ = rgb.shape
    gh, gw = height // PATCH, width // PATCH
    blocks = rgb.reshape(gh // 2, 2, PATCH, gw // 2, 2, PATCH, 3)  # R, dr, y, C, dc, x, channel
    return np.ascontiguousarray(blocks.transpose(0, 3, 1, 4, 6, 2, 5)).tobytes()


def prepare(raw: bytes, max_tokens: int) -> tuple[int, int, bytes]:
    """An image's grid (in patches) and pixels, sized to at most `max_tokens` tokens."""
    try:
        with PILImage.open(io.BytesIO(raw)) as im:
            if im.width * im.height > MAX_SOURCE_PIXELS:
                raise ImageError(f"an image of {im.width} x {im.height} pixels is too large to read")
            rgb = im.convert("RGB")
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError) as exc:
        raise ImageError(f"the image does not decode: {exc}") from None
    h, w = smart_resize(rgb.height, rgb.width, max_tokens * FACTOR * FACTOR)
    # The minimum can push a small limit over: the longer side gives way.
    while (h // FACTOR) * (w // FACTOR) > max_tokens:
        if h >= w:
            h -= FACTOR
        else:
            w -= FACTOR
    if (rgb.height, rgb.width) != (h, w):
        rgb = rgb.resize((w, h), PILImage.Resampling.BICUBIC)
    return h // PATCH, w // PATCH, patches(np.asarray(rgb, dtype=np.uint8))


def expand(
    ids: Sequence[int],
    pad_id: int,
    images: Sequence[tuple[int, int, bytes]],
    video_pad_id: int = -1,
    pairs: Sequence[tuple[int, int, bytes]] = (),
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
        [(at, pad_id, 1, im) for at, im in zip(pads, images, strict=True)]
        + [(at, video_pad_id, 2, p) for at, p in zip(vpads, pairs, strict=True)]
    )
    out: list[int] = []
    placed: list[PromptImage] = []
    start = 0
    for at, pad, frames, (gh, gw, pixels) in places:
        out.extend(ids[start:at])
        im = PromptImage(len(out), gh, gw, pixels, frames)
        out.extend([pad] * im.tokens)
        placed.append(im)
        start = at + 1
    out.extend(ids[start:])
    return out, placed
