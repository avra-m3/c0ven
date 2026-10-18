"""Videos in a request (spec/MODEL_VISION.md, section 6; ADR-0033).

A video arrives as a `data:video/...` URL or base64 bytes. PyAV decodes it; frames are sampled at 2 a second, at
least 4 and at most 768, as the model's video processor samples them. One size, picked by the processor's rule with
its per-frame cap, serves every frame, which Pillow resizes with bicubic resampling. Frames go to the encoder in
pairs, the last repeated when their count is odd; each pair is one block of tokens, cut into patches of two frames.

In the prompt the chat template writes `<|vision_start|><|video_pad|><|vision_end|>` per video. `placeholder` is the
text that replaces it: per pair, the pair's time as `<t seconds>`, then the pair's own three tokens.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import av
import numpy as np
from av.error import FFmpegError
from PIL import Image as PILImage

from frontend.images import FACTOR, MAX_SOURCE_PIXELS, PATCH, ImageError, from_base64

FPS = 2.0
"""Frames sampled per second of video (`Qwen3VLVideoProcessor.fps`)."""
MIN_FRAMES = 4
MAX_FRAMES = 768
MIN_PIXELS = 4_096
"""`video_preprocessor_config.json`'s `shortest_edge`: pixels of the whole video, all sampled frames together."""
MAX_PIXELS = 25_165_824
"""Its `longest_edge`, likewise."""
FRAME_TOKENS = 768
"""The per-frame cap: at most this many tokens' worth of pixels a frame (`max_video_tokens`)."""
DEFAULT_FPS = 24.0
"""The frame rate assumed when the file gives none, as transformers assumes."""
VIDEO_PAD = "<|video_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
TEMPLATE_ITEM = VISION_START + VIDEO_PAD + VISION_END
"""What the chat template writes for a video."""


@dataclass(frozen=True, slots=True)
class FramePair:
    """Two consecutive sampled frames: their time in seconds, their grid in patches and their pixels, grid_h x grid_w
    patches of two frames' 768 bytes each, in the encoder's order."""

    seconds: float
    grid_h: int
    grid_w: int
    pixels: bytes = field(repr=False)


def from_data_url(url: str) -> bytes:
    """The bytes of a `data:video/...;base64,` URL. Other URLs are refused: this engine does not fetch."""
    if not url.startswith("data:"):
        raise ImageError("only data: URLs are read; this engine does not fetch videos")
    head, sep, data = url.partition(",")
    if not sep or not head.endswith(";base64") or not head[5:].startswith("video/"):
        raise ImageError("a video URL must be data:video/<type>;base64,<data>")
    return from_base64(data)


def sample(total: int, fps: float) -> list[int]:
    """The indices of the frames sampled from `total` frames at `fps` (`Qwen3VLVideoProcessor.sample_frames`)."""
    n = int(total / fps * FPS)
    n = min(max(n, MIN_FRAMES), MAX_FRAMES, total)
    return [int(i) for i in np.linspace(0, total - 1, n).round().astype(int)]


def smart_resize(frames: int, height: int, width: int, max_tokens: int) -> tuple[int, int]:
    """The size every frame takes: the video processor's rule with its per-frame cap, factor 32 and temporal factor 2
    (spec/MODEL_VISION.md, section 6), then shrunk until a pair has at most `max_tokens` tokens."""
    if frames < 2:
        raise ImageError("a video needs at least 2 frames")
    per_frame = max(min(FRAME_TOKENS * FACTOR * FACTOR, MAX_PIXELS // frames), int(MIN_PIXELS * 1.05))
    max_pixels = per_frame * frames
    if height < FACTOR or width < FACTOR:
        scale = max(FACTOR / height, FACTOR / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ImageError(f"a video's sides may differ at most 200 times, not {height} x {width}")
    h = round(height / FACTOR) * FACTOR
    w = round(width / FACTOR) * FACTOR
    t = round(frames / 2) * 2
    if t * h * w > max_pixels:
        beta = math.sqrt(frames * height * width / max_pixels)
        h = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif t * h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (frames * height * width))
        h = math.ceil(height * beta / FACTOR) * FACTOR
        w = math.ceil(width * beta / FACTOR) * FACTOR
    while (h // FACTOR) * (w // FACTOR) > max_tokens:
        if h >= w:
            h -= FACTOR
        else:
            w -= FACTOR
    return h, w


def timestamps(indices: list[int], fps: float) -> list[float]:
    """Each pair's time: the mean of its two frames' times, the last frame repeated for an odd count
    (`Qwen3VLProcessor._calculate_timestamps`)."""
    padded = indices + indices[-1:] * (len(indices) % 2)
    times = [i / fps for i in padded]
    return [(times[i] + times[i + 1]) / 2 for i in range(0, len(times), 2)]


def pair_patches(first: np.ndarray, second: np.ndarray) -> bytes:
    """Two H x W x 3 uint8 frames, both sides multiples of 32, as patches in the encoder's order, each patch the first
    frame's 768 bytes and then the second's, each frame channel, row, column (spec/MODEL_VISION.md, section 6)."""
    height, width, _ = first.shape
    gh, gw = height // PATCH, width // PATCH
    both = np.stack([first, second]).reshape(2, gh // 2, 2, PATCH, gw // 2, 2, PATCH, 3)  # f, R, dr, y, C, dc, x, ch
    return np.ascontiguousarray(both.transpose(1, 4, 2, 5, 0, 7, 3, 6)).tobytes()


def _decode(raw: bytes) -> tuple[list[np.ndarray], list[int], float]:
    """The sampled frames as RGB arrays, their indices and the video's frame rate."""
    try:
        with av.open(io.BytesIO(raw), mode="r") as container:
            if not container.streams.video:
                raise ImageError("the video has no video stream")
            stream = container.streams.video[0]
            if stream.width * stream.height > MAX_SOURCE_PIXELS:
                raise ImageError(f"a video of {stream.width} x {stream.height} pixels is too large to read")
            fps = float(stream.average_rate or 0) or DEFAULT_FPS
            total = stream.frames or sum(1 for p in container.demux(stream) if p.size)
            if total < 1:
                raise ImageError("the video has no frames")
            indices = sample(total, fps)
            wanted = set(indices)
            container.seek(0)
            frames: dict[int, np.ndarray] = {}
            for i, frame in enumerate(container.decode(stream)):
                if i > indices[-1]:
                    break
                if i in wanted:
                    frames[i] = frame.to_ndarray(format="rgb24")
    except FFmpegError as exc:
        raise ImageError(f"the video does not decode: {exc}") from None
    if not frames:
        raise ImageError("the video has no frames")
    # A file whose header promised more frames than it holds: the missing ones are the last one decoded.
    last = frames[max(frames)]
    kept = [i for i in indices if i <= max(frames)]
    return [frames.get(i, last) for i in kept], kept, fps


def prepare(raw: bytes, max_tokens: int) -> list[FramePair]:
    """A video's frame pairs, each of at most `max_tokens` tokens."""
    frames, indices, fps = _decode(raw)
    height, width, _ = frames[0].shape
    if any(f.shape != frames[0].shape for f in frames):
        raise ImageError("the video's frames change size")
    h, w = smart_resize(len(frames), height, width, max_tokens)
    if (height, width) != (h, w):
        frames = [
            np.asarray(PILImage.fromarray(f).resize((w, h), PILImage.Resampling.BICUBIC), dtype=np.uint8)
            for f in frames
        ]
    if len(frames) % 2:
        frames.append(frames[-1])
    return [
        FramePair(t, h // PATCH, w // PATCH, pair_patches(frames[2 * k], frames[2 * k + 1]))
        for k, t in enumerate(timestamps(indices, fps))
    ]


def placeholder(pairs: list[FramePair]) -> str:
    """The prompt text that replaces a video's `TEMPLATE_ITEM`: per pair, its time, then one pad between the vision
    marks (`Qwen3VLProcessor.replace_video_token` at transformers v4.57.0)."""
    return "".join(f"<{p.seconds:.1f} seconds>{TEMPLATE_ITEM}" for p in pairs)
