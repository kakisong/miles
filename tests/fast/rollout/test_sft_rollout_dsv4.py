"""Offline unit tests for the DeepSeek-V4 SFT loss masks.

V4 ships its chat template as ``encoding_dsv4`` (a Python module, not a jinja
template) and there is no V4 tokenizer available offline, so these tests inject a
trivial fake ``encoding_dsv4`` plus a char-level tokenizer. That makes char offsets
== token offsets, so the char span each message renders to maps 1:1 onto token
positions and the expected mask can be derived independently of the implementation.

Covered:
  * ``DeepSeekV4LossMaskGenerator``            — assistant turns trained, rest 0.
  * ``DeepSeekV4AllButSystemLossMaskGenerator`` — every non-system turn trained,
    only ``system`` spans (and the leading bos) stay 0.
  * ``step_loss_mask=0`` zeroes a trained turn under the all-but-system mask.
  * the two ``generate_rollout*`` entry points pick the matching generator and the
    derived ``response_length`` / sliced ``loss_mask`` are consistent.
"""

import json
import types
from argparse import Namespace

from tests.ci.ci_register import register_cpu_ci

import miles.rollout.sft_rollout_dsv4 as mod
from miles.rollout.sft_rollout_dsv4 import (
    DeepSeekV4AllButSystemLossMaskGenerator,
    DeepSeekV4LossMaskGenerator,
)
from miles.utils.types import Sample

register_cpu_ci(est_time=30, suite="stage-a-cpu")

BOS = "<bos>"


def _fake_enc():
    """Minimal stand-in for ``encoding_dsv4`` with deterministic per-message renders.

    Agent-aware but backward-compatible: a plain ``{role, content}`` message still
    renders to ``<role>content``. system ``tools`` render a ``<functions>`` block (so
    we can assert the schema lands in the system span); assistant ``reasoning_content``
    / ``tool_calls`` render ``<think>`` / ``<tool_call>`` blocks.
    """

    def render_message(i, messages, thinking_mode="chat", drop_thinking=False):
        msg = messages[i]
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            out = f"<{role}>{content}"
            if msg.get("tools"):
                out += "<functions>" + json.dumps(msg["tools"], sort_keys=True) + "</functions>"
            return out
        if role == "assistant":
            out = f"<{role}>"
            if msg.get("reasoning_content"):
                out += "<think>" + msg["reasoning_content"] + "</think>"
            out += content
            if msg.get("tool_calls"):
                out += "<tool_call>" + json.dumps(msg["tool_calls"], sort_keys=True) + "</tool_call>"
            return out
        return f"<{role}>{content}"

    return types.SimpleNamespace(
        bos_token=BOS,
        render_message=render_message,
        merge_tool_messages=lambda msgs: list(msgs),
        sort_tool_results_by_call_order=lambda msgs: list(msgs),
        _drop_thinking_messages=lambda msgs: list(msgs),
    )


class CharTokenizer:
    """Char-level tokenizer: one token per character, offsets == char positions."""

    name_or_path = ""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _make_gen(cls):
    gen = cls(CharTokenizer())
    gen._dsv4_encoding = _fake_enc()  # short-circuit _load_dsv4_encoding
    return gen


def _reference_spans(messages):
    """Independently reproduce (text, [(start, end, role)]) the generator builds."""
    text = BOS
    spans = []
    for msg in messages:
        rendered = f"<{msg['role']}>{msg['content']}"
        spans.append((len(text), len(text) + len(rendered), msg["role"]))
        text += rendered
    return text, spans


def _expected_mask(messages, trained_roles):
    text, spans = _reference_spans(messages)
    mask = [0] * len(text)
    for start, end, role in spans:
        if role in trained_roles:
            for k in range(start, end):
                mask[k] = 1
    return text, mask


CONVERSATION = [
    {"role": "system", "content": "SYSTEM PROMPT"},
    {"role": "user", "content": "USER ASKS"},
    {"role": "assistant", "content": "ASSISTANT REPLIES"},
    {"role": "tool", "content": "TOOL RESULT"},
    {"role": "assistant", "content": "FINAL ANSWER"},
]


def test_assistant_only_mask():
    gen = _make_gen(DeepSeekV4LossMaskGenerator)
    token_ids, loss_mask = gen.get_loss_mask(CONVERSATION)

    text, expected = _expected_mask(CONVERSATION, trained_roles={"assistant"})
    assert gen.tokenizer.decode(token_ids) == text
    assert len(token_ids) == len(loss_mask) == len(text)
    assert loss_mask == expected

    # Exactly the two assistant turns are trained; system / user / tool / bos are 0.
    trained = gen.get_text_from_loss_mask(token_ids, loss_mask)
    assert trained == ["<assistant>ASSISTANT REPLIES", "<assistant>FINAL ANSWER"]


def test_all_but_system_mask():
    gen = _make_gen(DeepSeekV4AllButSystemLossMaskGenerator)
    token_ids, loss_mask = gen.get_loss_mask(CONVERSATION)

    text, expected = _expected_mask(
        CONVERSATION, trained_roles={"user", "assistant", "tool"}
    )
    assert gen.tokenizer.decode(token_ids) == text
    assert len(token_ids) == len(loss_mask) == len(text)
    assert loss_mask == expected

    # Only the leading bos and the system span are masked out (0); everything else 1.
    sys_start, sys_end = 0, len(BOS) + len("<system>SYSTEM PROMPT")
    assert loss_mask[sys_start:sys_end] == [0] * sys_end
    assert all(m == 1 for m in loss_mask[sys_end:])

    # The whole non-system tail is contiguous, so it decodes back as one trained run.
    trained = gen.get_text_from_loss_mask(token_ids, loss_mask)
    assert trained == [text[sys_end:]]


def test_all_but_system_step_loss_mask_override():
    convo = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "KEEP ME", "step_loss_mask": 0},
        {"role": "assistant", "content": "TRAIN ME"},
    ]
    gen = _make_gen(DeepSeekV4AllButSystemLossMaskGenerator)
    token_ids, loss_mask = gen.get_loss_mask(convo)

    text, spans = _reference_spans(convo)
    by_role = {role: (s, e) for s, e, role in spans}
    # user turn is opted out via step_loss_mask=0 -> zeroed despite being non-system.
    u_s, u_e = by_role["user"]
    assert loss_mask[u_s:u_e] == [0] * (u_e - u_s)
    # assistant turn is still trained.
    a_s, a_e = by_role["assistant"]
    assert loss_mask[a_s:a_e] == [1] * (a_e - a_s)
    # system turn stays out.
    s_s, s_e = by_role["system"]
    assert loss_mask[s_s:s_e] == [0] * (s_e - s_s)


AGENT_CONVO = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "List the files."},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should call ls.",
        "tool_calls": [{"function": {"name": "ls", "arguments": "{}"}, "id": "c0", "type": "function"}],
    },
    {"role": "tool", "content": "a.txt b.txt"},
    {"role": "assistant", "content": "There are two files: a.txt and b.txt."},
]
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {"name": "ls", "description": "list files", "parameters": {"type": "object", "properties": {}}},
    }
]


def test_all_but_system_with_agent_tools():
    """Agent data: tools (passed separately, like metadata['tools']) get injected into the
    system message, so the <functions> schema is masked out while reasoning / tool_calls /
    tool-result turns are all trained."""
    gen = _make_gen(DeepSeekV4AllButSystemLossMaskGenerator)
    token_ids, loss_mask = gen.get_loss_mask(AGENT_CONVO, tools=AGENT_TOOLS)
    text = gen.tokenizer.decode(token_ids)
    assert len(token_ids) == len(loss_mask) == len(text)

    # The tool schema was injected and rendered (would be entirely absent before the fix).
    assert "<functions>" in text
    assert '"ls"' in text

    # The first non-system turn starts at "<user>"; everything before it (bos + the whole
    # system turn, INCLUDING the injected <functions> schema) is masked out, everything
    # after is trained.
    sys_end = text.index("<user>")
    assert "<functions>" in text[:sys_end]  # schema sits inside the masked system span
    assert all(m == 0 for m in loss_mask[:sys_end])
    assert all(m == 1 for m in loss_mask[sys_end:])

    # Reasoning, tool_calls and the tool-result turn are all in the trained region.
    trained_text = text[sys_end:]
    for marker in ("<think>", "<tool_call>", "<tool>a.txt b.txt", "There are two files"):
        assert marker in trained_text
    # ...and the tool schema is NOT in what actually gets trained.
    assert "<functions>" not in "".join(gen.get_text_from_loss_mask(token_ids, loss_mask))


def test_inject_tools_is_noop_without_tools():
    gen = _make_gen(DeepSeekV4AllButSystemLossMaskGenerator)
    msgs = [{"role": "user", "content": "hi"}]
    assert gen._inject_tools(msgs, None) is msgs
    assert gen._inject_tools(msgs, []) is msgs


def _capture_warnings(fn):
    """Run *fn*, returning the list of WARNING+ messages emitted by the module logger."""
    import logging

    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    mod.logger.addHandler(handler)
    old_level = mod.logger.level
    mod.logger.setLevel(logging.WARNING)
    try:
        fn()
    finally:
        mod.logger.removeHandler(handler)
        mod.logger.setLevel(old_level)
    return records


def test_render_equivalence_guard_silent_when_canonical_matches():
    gen = _make_gen(DeepSeekV4AllButSystemLossMaskGenerator)
    enc = gen._dsv4_encoding
    # encode_messages that reproduces bos + Σ render_message exactly -> no warning.
    enc.encode_messages = lambda messages, **kw: BOS + "".join(
        enc.render_message(i, messages) for i in range(len(messages))
    )
    warnings = _capture_warnings(lambda: gen.get_loss_mask(CONVERSATION))
    assert not any("diverged" in m for m in warnings), warnings


def test_render_equivalence_guard_warns_on_mismatch_without_raising():
    gen = _make_gen(DeepSeekV4AllButSystemLossMaskGenerator)
    enc = gen._dsv4_encoding
    enc.encode_messages = lambda messages, **kw: "COMPLETELY DIFFERENT RENDER"
    # Must NOT raise (warn-only), and must emit the divergence warning.
    warnings = _capture_warnings(lambda: gen.get_loss_mask(CONVERSATION))
    assert any("diverged from encode_messages" in m for m in warnings), warnings


def test_render_equivalence_guard_skips_when_encoder_call_raises():
    gen = _make_gen(DeepSeekV4AllButSystemLossMaskGenerator)
    enc = gen._dsv4_encoding

    def _boom(messages, **kw):
        raise RuntimeError("bad kwargs")

    enc.encode_messages = _boom
    # The guard swallows the encoder error (cannot-run != mismatch) and never raises.
    warnings = _capture_warnings(lambda: gen.get_loss_mask(CONVERSATION))
    assert any("could not cross-check" in m for m in warnings), warnings
    assert not any("diverged" in m for m in warnings), warnings


def _run_entry(entry_fn, messages, monkeypatch):
    """Drive a generate_rollout* entry point with fakes; return the mutated Sample."""
    monkeypatch.setattr(mod, "load_tokenizer", lambda *a, **k: CharTokenizer())
    monkeypatch.setattr(mod, "load_processor", lambda *a, **k: object())
    monkeypatch.setattr(
        mod.DeepSeekV4LossMaskGenerator,
        "_load_dsv4_encoding",
        lambda self: _fake_enc(),
    )
    # Reset module-level caches so each entry point starts clean.
    mod.TOKENIZER = None
    mod.PROCESSOR = None
    mod._MASK_GENERATORS.clear()
    mod._SAMPLE_PRINTED.clear()

    sample = Sample(prompt=messages, metadata={})

    class _Buf:
        def get_samples(self, n):
            return [(sample,)]

    args = Namespace(
        rollout_global_dataset=True,
        hf_checkpoint="dummy",
        chat_template_path=None,
        rollout_batch_size=1,
    )
    entry_fn(args, 0, _Buf())
    return sample


def test_entry_points_select_matching_generator(monkeypatch):
    text, _ = _reference_spans(CONVERSATION)
    sys_len = len(BOS) + len("<system>SYSTEM PROMPT")

    asst = _run_entry(mod.generate_rollout, CONVERSATION, monkeypatch)
    allbut = _run_entry(mod.generate_rollout_all_but_system, CONVERSATION, monkeypatch)

    # sample.tokens is always the full sequence; loss_mask is its trained tail.
    assert asst.tokens == [ord(c) for c in text]
    assert allbut.tokens == [ord(c) for c in text]

    # response_length = first trained token .. end. The all-but-system mask starts
    # training at the first user token (right after bos + system span); the
    # assistant-only mask starts later, at the first assistant token.
    assert allbut.response_length == len(text) - sys_len
    assert allbut.response_length > asst.response_length
    assert len(allbut.loss_mask) == allbut.response_length
    assert len(asst.loss_mask) == asst.response_length

    # All-but-system trains strictly more tokens than assistant-only.
    assert sum(allbut.loss_mask) > sum(asst.loss_mask)
    # Its trained tail is the entire non-system remainder (no zeros once it starts).
    assert allbut.loss_mask == [1] * allbut.response_length


if __name__ == "__main__":
    test_assistant_only_mask()
    test_all_but_system_mask()
    test_all_but_system_step_loss_mask_override()
    print("ok")
