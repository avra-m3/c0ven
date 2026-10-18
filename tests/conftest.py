from __future__ import annotations

import pytest
from tokenizers import Tokenizer

from frontend.model import Model
from tests.helpers import tiny_model, tiny_tokenizer


@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    return tiny_tokenizer()


@pytest.fixture(scope="session")
def model(tokenizer: Tokenizer) -> Model:
    return tiny_model(tokenizer)
