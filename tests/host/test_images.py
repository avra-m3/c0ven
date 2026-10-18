"""Images through the control plane (ADR-0031): the IMG line, the front end's sizing and patches, the prompt's
expansion, the prefix keys, and a request with images from the HTTP API down to the core's calls."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage
from starlette.testclient import TestClient
from tokenizers import Tokenizer

import protocol
from engine.api.interface import IMAGE_PATCH_BYTES, Image, prefix_key
from engine.process import read_image
from frontend import images
from frontend.app import create_app
from frontend.model import Model
from tests.helpers import FakeEngine
from tests.host.test_native import expected, make, prompt_of


def png(width: int, height: int, seed: int = 0) -> bytes:
    rgb = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    out = io.BytesIO()
    PILImage.fromarray(rgb).save(out, "PNG")
    return out.getvalue()


def test_img_round_trips():
    line = protocol.Img(3, 17, 28, 36, "/proc/1/fd/9")
    assert protocol.parse_command(line.encode()) == line
    with pytest.raises(protocol.ProtocolError):
        protocol.Img(3, 0, 2, 2, "a b").encode()


@pytest.mark.parametrize(
    ("hw", "max_pixels", "want"),
    [
        ((448, 576), 4096 * 1024, (448, 576)),  # multiples of 32 inside the limits: kept
        ((100, 100), 4096 * 1024, (256, 256)),  # below 65,536 pixels: scaled up
        ((488, 640), 4096 * 1024, (480, 640)),  # rounded to 32
        ((4000, 3000), 1024 * 1024, (1152, 864)),  # above the maximum: scaled down, floored to 32
    ],
)
def test_smart_resize(hw: tuple[int, int], max_pixels: int, want: tuple[int, int]):
    h, w = images.smart_resize(*hw, max_pixels)
    assert (h, w) == want
    assert h % 32 == 0
    assert w % 32 == 0
    assert h * w <= max(max_pixels, 65_536 * 2)


def test_patches_are_in_the_encoders_order():
    rgb = np.random.default_rng(1).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    data = images.patches(rgb)
    gh, gw = 4, 6
    assert len(data) == gh * gw * IMAGE_PATCH_BYTES
    n = 0
    for big_r in range(gh // 2):
        for big_c in range(gw // 2):
            for k in range(4):
                r, c = 2 * big_r + k // 2, 2 * big_c + k % 2
                patch = np.frombuffer(data[n * 768 : (n + 1) * 768], dtype=np.uint8).reshape(3, 16, 16)
                want = rgb[16 * r : 16 * r + 16, 16 * c : 16 * c + 16, :].transpose(2, 0, 1)
                assert (patch == want).all(), (r, c)
                n += 1


def test_prepare_sizes_an_image_to_the_engines_limit():
    gh, gw, pixels = images.prepare(png(640, 488), 64)
    assert gh // 2 * (gw // 2) <= 64
    assert len(pixels) == gh * gw * 768


def test_expand_repeats_each_pad_once_per_token():
    ids, placed = images.expand([5, 9, 6, 9, 7], 9, [(2, 4, b"a" * 8 * 768), (4, 2, b"b" * 8 * 768)])
    assert ids == [5, 9, 9, 6, 9, 9, 7]
    assert [(p.first, p.tokens) for p in placed] == [(1, 2), (4, 2)]
    with pytest.raises(images.ImageError):
        images.expand([5, 9], 9, [])


def test_prefix_keys_differ_by_pixels_only_where_the_image_is():
    a = Image(2, 2, 4, b"\x01" * 8 * 768)
    b = Image(2, 2, 4, b"\x02" * 8 * 768)
    prompt = (1, 2, 9, 9, 3)
    ka, kb = prefix_key(prompt, (a,)), prefix_key(prompt, (b,))
    assert ka[:2] == kb[:2] == (1, 2)
    assert ka[2:4] != kb[2:4]
    assert all(t < 0 for t in ka[2:4])
    assert ka[4:] == kb[4:]
    assert prefix_key(prompt, ()) == prompt


def test_the_engine_reads_an_images_file(tmp_path: Path):
    f = tmp_path / "img"
    f.write_bytes(b"\x07" * 4 * 768)
    im = read_image(protocol.Img(1, 3, 2, 2, str(f)))
    assert (im.first, im.grid_h, im.grid_w, im.tokens) == (3, 2, 2, 1)
    with pytest.raises(ValueError, match="bytes"):
        read_image(protocol.Img(1, 3, 2, 4, str(f)))


def test_an_image_from_the_openai_api_reaches_the_engine(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer, image_tokens_max=64)
    url = "data:image/png;base64," + base64.b64encode(png(256, 256)).decode()
    content = [{"type": "text", "text": "What is it?"}, {"type": "image_url", "image_url": {"url": url}}]
    with TestClient(create_app(model, engine)) as client:
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": content}]})
    assert r.status_code == 200, r.text
    (sent,) = engine.images
    (im,) = sent
    pad = tokenizer.token_to_id("<|image_pad|>")
    prompt = engine.prompts[0]
    assert (im.grid_h, im.grid_w, im.tokens) == (16, 16, 64)
    assert prompt[im.first - 1] == tokenizer.token_to_id("<|vision_start|>")
    assert prompt[im.first : im.first + 64] == [pad] * 64
    assert prompt[im.first + 64] == tokenizer.token_to_id("<|vision_end|>")


def test_an_image_from_the_anthropic_api_reaches_the_engine(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer, image_tokens_max=64)
    source = {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png(300, 200)).decode()}
    body = {
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [{"type": "image", "source": source}, {"type": "text", "text": "?"}]}],
    }
    with TestClient(create_app(model, engine)) as client:
        r = client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    (sent,) = engine.images
    assert len(sent) == 1
    assert sent[0].tokens <= 64


def test_images_are_refused_by_an_engine_without_them(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer)
    url = "data:image/png;base64," + base64.b64encode(png(64, 64)).decode()
    content = [{"type": "image_url", "image_url": {"url": url}}]
    with TestClient(create_app(model, engine)) as client:
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": content}]})
        assert r.status_code == 400
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x"}}]}]},
        )
        assert r.status_code == 400


def test_images_go_to_the_core_and_key_the_shared_prefix():
    def at_64(prompt: tuple[int, ...]) -> list[tuple[int, bool]]:
        return [(64, True)] if len(prompt) > 64 else []

    run = make(rule=at_64)
    head = prompt_of(10, 5)
    block = [9] * 16
    prefix = [*head, *block, *prompt_of(64 - 26, 6)]
    one = Image(10, 8, 8, b"\x01" * 64 * 768)
    other = Image(10, 8, 8, b"\x02" * 64 * 768)
    a, b, c = [*prefix, *prompt_of(5, 7)], [*prefix, *prompt_of(6, 8)], [*prefix, *prompt_of(7, 9)]
    run.add(1, a, 3, (one,))
    run.until_idle()
    assert run.tokens(1) == expected(a, 3)
    run.add(2, b, 3, (other,))
    run.until_idle()
    assert run.done(2).reused == 0, "a different image shares no prefix past it"
    run.add(3, c, 3, (one,))
    run.until_idle()
    assert run.done(3).reused == 64
    assert not [e for e in run.events if isinstance(e, protocol.Error)]


def test_the_scheduler_refuses_images_out_of_place():
    run = make()
    prompt = prompt_of(20)
    run.add(1, prompt, 3, (Image(18, 4, 4, b"\x00" * 16 * 768),))
    (err,) = [e for e in run.events if isinstance(e, protocol.Error)]
    assert "past the prompt" in err.message
