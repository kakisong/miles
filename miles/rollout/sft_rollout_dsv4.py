"""DeepSeek-V4 SFT rollout — self-contained, plugged in via ``--rollout-function-path``.

Why this lives here instead of in ``mask_utils.py`` / ``arguments.py``:
  miles' convention for custom behavior is a ``--*-path`` hook resolved by
  ``load_function`` (``--rollout-function-path``, ``--custom-rm-path``, ...). SFT is
  already selected exactly this way::

      --rollout-function-path miles.rollout.sft_rollout.generate_rollout

  Routing V4 through its own rollout keeps the V4-specific loss-mask logic off
  trunk's shared ``MultiTurnLossMaskGenerator`` and the ``--loss-mask-type`` enum,
  so the upstream-sync branch carries one additive file instead of edits to hot
  shared files (fewer rebase conflicts).

Launch (replace the rollout-function-path in your SFT run.sh)::

      --rollout-function-path miles.rollout.sft_rollout_dsv4.generate_rollout
      --debug-train-only

  ``--loss-mask-type`` is ignored by this module; it always builds the V4 mask.

Env knobs (same names/semantics as before)::

      MILES_DSV4_THINKING_MODE = "chat" (default) | "thinking"
      MILES_DSV4_DROP_THINKING = "0" (default) | "1"
"""

import logging

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_processor, load_tokenizer

__all__ = ["generate_rollout", "DeepSeekV4LossMaskGenerator"]

logger = logging.getLogger(__name__)


class DeepSeekV4LossMaskGenerator(MultiTurnLossMaskGenerator):
    """Build the DeepSeek-V4 multi-turn SFT loss mask from per-message render spans.

    V4 ships its chat template as a Python module (``encoding_dsv4.py``) rather than a
    jinja ``chat_template``, so the base class' probe (``apply_chat_template`` on a dummy
    conversation) would raise. We skip that probe and instead render each message through
    ``encoding_dsv4`` to recover its char span, then map spans to token spans via the
    tokenizer's offset mapping.
    """

    # Low-level symbols the V4 mask builder needs from ``encoding_dsv4``. Used to validate
    # the sglang-bundled module before trusting it, so a vendored copy with a different
    # surface transparently falls back to the checkpoint's reference module.
    _DSV4_REQUIRED = (
        "bos_token",
        "render_message",
        "merge_tool_messages",
        "sort_tool_results_by_call_order",
        "_drop_thinking_messages",
    )

    def __init__(self, tokenizer):
        # Intentionally do NOT call super().__init__: the base ctor probes the tokenizer
        # via apply_chat_template, which V4 has no jinja template for. We set the same
        # attributes the base would, minus the probe.
        self.tokenizer = tokenizer
        self.tokenizer_type = "deepseek_v4"
        self.system_message_length = 0
        self.gen_token_length = 0

    def _load_dsv4_encoding(self):
        """Return the DeepSeek-V4 ``encoding_dsv4`` module (cached on the instance).

        Source order:
          1. ``sglang.srt.entrypoints.openai.encoding_dsv4`` — the image-bundled module
             that the ``apply_chat_template`` bridge (``chat_template_utils/deepseek_v4.py``)
             and the DeepSeek-V4 TITO tokenizer already use, so the SFT mask renders from
             the same encoder as rollout/serving (consistency == correctness).
          2. fallback: the checkpoint's ``encoding/encoding_dsv4.py`` (for images that do
             not vendor it, or to pin the exact published tokenization).

        Both expose the same low-level API (``render_message`` / ``merge_tool_messages`` /
        ``sort_tool_results_by_call_order`` / ``_drop_thinking_messages`` / ``bos_token``).
        Only the bridge's ``render_messages()`` wrapper is unusable here, because it returns
        a single joined prompt string with no per-message spans.
        """
        enc = getattr(self, "_dsv4_encoding", None)
        if enc is not None:
            return enc
        try:
            from sglang.srt.entrypoints.openai import encoding_dsv4 as sg_enc
        except Exception as exc:  # noqa: BLE001 — any import failure falls back to the checkpoint
            logger.info("sglang encoding_dsv4 unavailable (%s); loading from checkpoint", exc)
            enc = self._load_dsv4_encoding_from_checkpoint()
        else:
            missing = [name for name in self._DSV4_REQUIRED if not hasattr(sg_enc, name)]
            if missing:
                logger.info("sglang encoding_dsv4 missing %s; loading from checkpoint", missing)
                enc = self._load_dsv4_encoding_from_checkpoint()
            else:
                enc = sg_enc
        self._dsv4_encoding = enc
        return enc

    def _load_dsv4_encoding_from_checkpoint(self):
        """Load ``encoding_dsv4`` directly from the checkpoint dir (fallback path)."""
        import importlib.util
        import os
        import sys

        tk_path = getattr(self.tokenizer, "name_or_path", None) or ""
        enc_path = os.path.join(tk_path, "encoding", "encoding_dsv4.py")
        if not os.path.isfile(enc_path):
            raise FileNotFoundError(
                f"DeepSeek-V4 chat-template module not found at {enc_path}. "
                "Install an sglang that vendors encoding_dsv4, or pass --hf-checkpoint "
                "pointing to a V4 HF dir that contains encoding/encoding_dsv4.py."
            )
        spec = importlib.util.spec_from_file_location("encoding_dsv4", enc_path)
        enc = importlib.util.module_from_spec(spec)
        sys.modules["encoding_dsv4"] = enc
        spec.loader.exec_module(enc)
        return enc

    def gen_multi_turn_loss_mask_deepseek_v4(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        """Mask=1 over each assistant render span (reasoning + content + tool_calls + eos),
        mask=0 elsewhere (incl. transition tokens like ``<｜Assistant｜>`` / ``<think>``)."""
        import os

        enc = self._load_dsv4_encoding()

        thinking_mode = os.environ.get("MILES_DSV4_THINKING_MODE", "chat")
        drop_thinking = os.environ.get("MILES_DSV4_DROP_THINKING", "0") == "1"
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"MILES_DSV4_THINKING_MODE must be 'chat' or 'thinking', got {thinking_mode!r}")

        # Pre-process messages the same way encode_messages does, so render_message
        # sees the same context.
        full_messages = enc.merge_tool_messages(list(messages))
        full_messages = enc.sort_tool_results_by_call_order(full_messages)
        if any(m.get("tools") for m in full_messages):
            drop_thinking = False
        if thinking_mode == "thinking" and drop_thinking:
            full_messages = enc._drop_thinking_messages(full_messages)

        # Render piece-by-piece so we know each message's char span.
        bos = enc.bos_token
        pieces: list[tuple[int, int, str]] = []  # (start, end, role)
        text = bos
        for i in range(len(full_messages)):
            rendered = enc.render_message(
                i, full_messages, thinking_mode=thinking_mode, drop_thinking=drop_thinking
            )
            pieces.append((len(text), len(text) + len(rendered), full_messages[i].get("role", "")))
            text += rendered

        # Tokenize whole conversation with offsets, so we can map char spans -> token spans.
        enc_out = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = enc_out["input_ids"]
        offsets = enc_out["offset_mapping"]
        loss_mask = [0] * len(token_ids)

        for start, end, role in pieces:
            if role != "assistant":
                continue
            # Mark tokens whose offset is inside [start, end). offsets[k] = (s, e).
            for k, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue  # special tokens with no offset
                if s >= start and e <= end:
                    loss_mask[k] = 1

        # Per-message override (consistent with other branches).
        # If a single assistant turn has step_loss_mask=0, zero out that span.
        for (start, end, role), msg in zip(pieces, full_messages):
            if role != "assistant":
                continue
            if msg.get("step_loss_mask", 1) == 1:
                continue
            for k, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue
                if s >= start and e <= end:
                    loss_mask[k] = 0

        return token_ids, loss_mask

    def get_loss_mask(self, messages: list[dict], tools: list[dict] = None) -> tuple[list[int], list[int]]:
        # V4-only generator: always build the V4 mask. Overriding get_loss_mask (rather
        # than just exposing the method) also keeps get_loss_mask_with_multimodal_alignment
        # working, since it calls self.get_loss_mask internally.
        return self.gen_multi_turn_loss_mask_deepseek_v4(messages, tools)


TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """DeepSeek-V4 SFT rollout.

    Mirror of ``miles.rollout.sft_rollout.generate_rollout``, but always builds the
    loss mask with :class:`DeepSeekV4LossMaskGenerator` (ignores ``--loss-mask-type``).

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[Sample]: a list of samples generated by the rollout
    """
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    if MASK_GENERATOR is None:
        MASK_GENERATOR = DeepSeekV4LossMaskGenerator(TOKENIZER)

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample in enumerate(samples):
        (sample,) = sample
        messages = sample.prompt
        tools = sample.metadata.get("tools", None)

        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)

        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                f"sft_rollout_dsv4::generate_rollout example data: {sample=} (raw){messages=} (raw){token_ids=} (raw){loss_mask=} {response_length=}"
            )
            SAMPLE_PRINTED = True

    return samples
