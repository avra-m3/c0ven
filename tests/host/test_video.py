"""Videos through the control plane (ADR-0033): the front end's sampling, sizing, timestamps and frame pairs, the
prompt's text and expansion, the IMG line with two frames, and a request with a video from the HTTP API down to the
core's calls."""

# PyAV's stubs leave add_stream's keywords and encode's packets partly untyped; only mp4() calls them.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from av.video.stream import VideoStream
from starlette.testclient import TestClient
from tokenizers import Tokenizer

import protocol
from engine.api.interface import IMAGE_PATCH_BYTES, Image, prefix_key
from engine.process import read_image
from frontend import images, video
from frontend.app import create_app
from frontend.model import Model
from tests.helpers import FakeEngine
from tests.host.test_native import expected, make, prompt_of


def mp4(width: int, height: int, frames: int, fps: int = 4) -> bytes:
    """A video whose frame i is one flat grey of level 20 i, so a decoded frame names its index."""
    out = io.BytesIO()
    with av.open(out, mode="w", format="mp4") as container:
        stream: VideoStream = container.add_stream("mpeg4", rate=Fraction(fps))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(frames):
            rgb = np.full((height, width, 3), 20 * i, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out.getvalue()


def test_frames_are_sampled_at_two_a_second():
    assert video.sample(10, 4.0) == [0, 2, 4, 7, 9]
    assert video.sample(3, 30.0) == [0, 1, 2], "fewer frames than the minimum: all of them"
    assert len(video.sample(100_000, 24.0)) == video.MAX_FRAMES


def test_a_pairs_time_is_the_mean_of_its_frames():
    assert video.timestamps([0, 2, 4, 7, 9], 4.0) == [0.25, 1.375, 2.25]
    assert video.timestamps([0, 30], 30.0) == [0.5]


@pytest.mark.parametrize(
    ("frames", "hw", "max_tokens", "want"),
    [
        (5, (64, 96), 4096, (64, 96)),  # multiples of 32 inside the limits: kept
        (100, (1080, 1920), 4096, (352, 640)),  # the per-frame share of the video's pixels
        (8, (1080, 1920), 4096, (640, 1152)),  # the per-frame cap of 768 tokens
        (8, (1080, 1920), 64, (256, 256)),  # the engine's limit: the longer side gives way
        (4, (10, 20), 4096, (32, 64)),  # a side under 32 pixels: scaled up to 32
        (2, (40, 40), 4096, (64, 64)),  # below the video's 4,096 pixels: scaled up
    ],
)
def test_smart_resize(frames: int, hw: tuple[int, int], max_tokens: int, want: tuple[int, int]):
    h, w = video.smart_resize(frames, *hw, max_tokens)
    assert (h, w) == want
    assert (h // 32) * (w // 32) <= min(max_tokens, video.FRAME_TOKENS)


def test_a_one_frame_video_is_refused():
    with pytest.raises(images.ImageError, match="2 frames"):
        video.smart_resize(1, 64, 64, 4096)


def test_a_pairs_patches_hold_both_frames_in_the_encoders_order():
    rng = np.random.default_rng(2)
    first, second = (rng.integers(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(2))
    data = video.pair_patches(first, second)
    gh, gw = 4, 6
    assert len(data) == gh * gw * 2 * IMAGE_PATCH_BYTES
    one, two = images.patches(first), images.patches(second)
    for n in range(gh * gw):
        patch = data[n * 1536 : (n + 1) * 1536]
        assert patch[:768] == one[n * 768 : (n + 1) * 768]
        assert patch[768:] == two[n * 768 : (n + 1) * 768]


def test_prepare_pairs_the_sampled_frames():
    pairs = video.prepare(mp4(96, 64, 10), 4096)
    assert [round(p.seconds, 3) for p in pairs] == [0.25, 1.375, 2.25]
    assert all((p.grid_h, p.grid_w) == (4, 6) for p in pairs)
    # Sampled frames 0, 2, 4, 7, 9 and 9 again: each half of a patch is one flat grey.
    levels = [
        [int(np.frombuffer(p.pixels[f * 768 : (f + 1) * 768], dtype=np.uint8).mean().round()) for f in (0, 1)]
        for p in pairs
    ]
    want = [[0, 40], [80, 140], [180, 180]]
    assert np.abs(np.array(levels) - np.array(want)).max() <= 6, levels  # a lossy codec


def test_a_video_that_does_not_decode_is_refused():
    with pytest.raises(images.ImageError):
        video.prepare(b"not a video", 4096)
    with pytest.raises(images.ImageError, match="data:video"):
        video.from_data_url("data:image/png;base64,AAAA")


def test_placeholder_writes_each_pairs_time_and_block():
    pairs = [video.FramePair(0.25, 2, 2, b""), video.FramePair(1.375, 2, 2, b"")]
    assert video.placeholder(pairs) == (
        "<0.2 seconds><|vision_start|><|video_pad|><|vision_end|>"
        "<1.4 seconds><|vision_start|><|video_pad|><|vision_end|>"
    )


def test_expand_places_images_and_frame_pairs_in_prompt_order():
    ids, placed = images.expand(
        [5, 8, 6, 9, 7, 8], 9, [(2, 2, b"i" * 4 * 768)], 8, [(2, 4, b"a" * 16 * 768), (2, 4, b"b" * 16 * 768)]
    )
    assert ids == [5, 8, 8, 6, 9, 7, 8, 8]
    assert [(p.first, p.tokens, p.frames) for p in placed] == [(1, 2, 2), (4, 1, 1), (6, 2, 2)]


def test_img_carries_the_frames():
    line = protocol.Img(3, 17, 28, 36, "/proc/1/fd/9", 2)
    assert line.encode() == "IMG 3 17 28 36 2 /proc/1/fd/9"
    assert protocol.parse_command(line.encode()) == line


def test_the_engine_reads_a_frame_pairs_file(tmp_path: Path):
    f = tmp_path / "pair"
    f.write_bytes(b"\x07" * 4 * 1536)
    im = read_image(protocol.Img(1, 3, 2, 2, str(f), 2))
    assert (im.first, im.tokens, im.frames, len(im.pixels)) == (3, 1, 2, 4 * 1536)
    with pytest.raises(ValueError, match="bytes"):
        read_image(protocol.Img(1, 3, 2, 2, str(f), 1))
    with pytest.raises(ValueError, match="frame"):
        read_image(protocol.Img(1, 3, 2, 2, str(f), 3))


def test_prefix_keys_tell_a_still_image_from_a_frame_pair():
    frame = b"\x01" * 4 * 768
    still, pair = Image(2, 2, 2, frame), Image(2, 2, 2, frame + frame, 2)
    prompt = (1, 2, 9, 3)
    assert prefix_key(prompt, (still,)) != prefix_key(prompt, (pair,))


def test_a_video_from_the_openai_api_reaches_the_engine(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer, image_tokens_max=64)
    url = "data:video/mp4;base64," + base64.b64encode(mp4(96, 64, 10)).decode()
    content = [{"type": "video_url", "video_url": {"url": url}}, {"type": "text", "text": "What happens?"}]
    with TestClient(create_app(model, engine)) as client:
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": content}]})
    assert r.status_code == 200, r.text
    (sent,) = engine.images
    assert [(im.grid_h, im.grid_w, im.frames) for im in sent] == [(4, 6, 2)] * 3
    prompt = engine.prompts[0]
    pad, start, end = (tokenizer.token_to_id(t) for t in ("<|video_pad|>", "<|vision_start|>", "<|vision_end|>"))
    for im in sent:
        assert prompt[im.first - 1] == start
        assert prompt[im.first : im.first + im.tokens] == [pad] * im.tokens
        assert prompt[im.first + im.tokens] == end
    text = tokenizer.decode(prompt, skip_special_tokens=False)
    assert "<0.2 seconds><|vision_start|>" in text
    assert "<1.4 seconds>" in text
    assert "<2.2 seconds>" in text
    assert text.count("<|vision_start|>") == 3, "the template's own marks are replaced, not doubled"


def test_a_video_from_the_anthropic_api_reaches_the_engine(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer, image_tokens_max=64)
    source = {"type": "base64", "media_type": "video/mp4", "data": base64.b64encode(mp4(96, 64, 6)).decode()}
    body = {
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [{"type": "video", "source": source}, {"type": "text", "text": "?"}]}],
    }
    with TestClient(create_app(model, engine)) as client:
        r = client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    (sent,) = engine.images
    assert len(sent) == 2
    assert all(im.frames == 2 for im in sent)


def test_videos_are_refused_by_an_engine_without_images(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer)
    url = "data:video/mp4;base64," + base64.b64encode(mp4(64, 64, 4)).decode()
    content = [{"type": "video_url", "video_url": {"url": url}}]
    with TestClient(create_app(model, engine)) as client:
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": content}]})
    assert r.status_code == 400


def test_frame_pairs_go_to_the_core_with_their_frames(monkeypatch: pytest.MonkeyPatch):
    run = make()
    seen: list[list[Image]] = []
    admit = run.core.admit_images

    def record(slot: int, ims: Sequence[Image]) -> None:
        seen.append(list(ims))
        admit(slot, ims)

    monkeypatch.setattr(run.core, "admit_images", record)
    head = prompt_of(10, 5)
    pairs = (Image(10, 4, 4, b"\x01" * 16 * 1536, 2), Image(16, 4, 4, b"\x02" * 16 * 1536, 2))
    prompt = [*head, *[9] * 4, 1, 2, *[9] * 4, *prompt_of(5, 7)]
    run.add(1, prompt, 3, pairs)
    run.until_idle()
    assert run.tokens(1) == expected(prompt, 3)
    (got,) = seen
    assert [(im.first, im.frames) for im in got] == [(10, 2), (16, 2)]
