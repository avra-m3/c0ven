from __future__ import annotations

from engine.models.inkling2_loop import tensors
from engine.models.inkling2_loop.turns import KEEP_SYSTEM, PERIODIC, ChatTokens, checkpoint_rule


def test_checkpoint_marks_model_turns_long_system_and_periodic():
    chat = ChatTokens.of(tensors.from_tsv())
    rule = checkpoint_rule(chat)

    system = (chat.message_system, *([42] * (KEEP_SYSTEM + 10)), chat.end_message)
    user = (chat.message_user, 10, 11, chat.end_message)
    model = (chat.message_model, 20, chat.end_message)
    long_tail = (chat.message_user, *([99] * (PERIODIC + 100)), chat.end_message)
    prompt = (*system, *user, *model, *long_tail)

    marks = rule(prompt)
    assert (len(system), True) in marks
    assert (len(system) + len(user), False) in marks
    assert (PERIODIC, False) in marks
