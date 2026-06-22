"""DeepSeek-V4 SFT rollouts — self-contained, plugged in via ``--rollout-function-path``.

Why this lives here instead of in ``mask_utils.py`` / ``arguments.py``:
  miles' convention for custom behavior is a ``--*-path`` hook resolved by
  ``load_function`` (``--rollout-function-path``, ``--custom-rm-path``, ...). SFT is
  already selected exactly this way::

      --rollout-function-path miles.rollout.sft_rollout.generate_rollout

  Routing V4 through its own rollout keeps the V4-specific loss-mask logic off
  trunk's shared ``MultiTurnLossMaskGenerator`` and the ``--loss-mask-type`` enum,
  so the upstream-sync branch carries one additive file instead of edits to hot
  shared files (fewer rebase conflicts).

Two loss masks are exposed as separate rollout entry points (pick one via
``--rollout-function-path``). Both ignore ``--loss-mask-type`` and build the V4 mask
from per-message render spans; they differ only in *which roles enter the loss*:

  * ``...sft_rollout_dsv4.generate_rollout`` — train **assistant turns only**
    (reasoning + content + tool_calls + eos). The original / default V4 SFT mask.

  * ``...sft_rollout_dsv4.generate_rollout_all_but_system`` — train **everything
    except ``system``**. Every non-system message (user + assistant + tool, including
    its role markers / reasoning / eos) is mask=1; only ``system`` spans (and the
    leading bos) stay mask=0. Use when the model should fit the whole conversation
    and you merely want the system prompt dropped from the loss. For agent data this
    means: the tool-*schema* (the ``<functions>`` block, injected into the system
    message) is mask=0, while assistant reasoning + tool_calls and ``tool`` (tool-
    *result*) turns are all mask=1.

Launch (replace the rollout-function-path in your SFT run.sh), e.g.::

      --rollout-function-path miles.rollout.sft_rollout_dsv4.generate_rollout_all_but_system
      --debug-train-only

Env knobs (same names/semantics as before; they only change how turns are *rendered*,
not which turns are trained, so they apply to both masks)::

      MILES_DSV4_THINKING_MODE = "chat" (default) | "thinking"
      MILES_DSV4_DROP_THINKING = "0" (default) | "1"
"""

import logging

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_processor, load_tokenizer

__all__ = [
    "generate_rollout",
    "generate_rollout_all_but_system",
    "DeepSeekV4LossMaskGenerator",
    "DeepSeekV4AllButSystemLossMaskGenerator",
]

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

    def _should_train_piece(self, role: str) -> bool:
        """Whether tokens of a message with this ``role`` enter the loss (mask=1).

        Base V4 mask trains assistant turns only. Subclasses widen/narrow the trained
        roles — see :class:`DeepSeekV4AllButSystemLossMaskGenerator`."""
        return role == "assistant"

    @staticmethod
    def _inject_tools(messages: list[dict], tools: list[dict] | None) -> list[dict]:
        """Splice *tools* into the system message, matching the served render path.

        SFT data delivers tools as a separate ``metadata["tools"]`` list, but the V4
        encoder reads them from ``messages[0]["tools"]`` (the ``<functions>`` schema is
        part of the *system* turn). Without this, the rendered text — and therefore
        ``sample.tokens`` — omits the entire tool-schema block the model is served with
        (a train/serve tokenization mismatch). Mirror serving by delegating to the
        canonical ``_inject_tools_into_system`` (round-trips each tool through sglang's
        ``Tool`` so token ids match exactly); fall back to a raw injection when sglang is
        unavailable (checkpoint-encoding path). Returns *messages* unchanged when there
        are no tools. The injected schema lives in the system message, so the masks
        correctly leave it at mask=0.
        """
        if not tools:
            return messages
        try:
            from miles.utils.chat_template_utils.deepseek_v4 import _inject_tools_into_system

            return _inject_tools_into_system(messages, tools)
        except Exception as exc:  # noqa: BLE001 — sglang absent (checkpoint fallback) etc.
            logger.warning(
                "canonical V4 tool injection unavailable (%s); using best-effort raw "
                "injection — tool-schema token ids may drift from what sglang serves",
                exc,
            )
            import copy

            out = copy.deepcopy(messages)
            if not out or out[0].get("role") != "system":
                out.insert(0, {"role": "system", "content": ""})
            out[0]["tools"] = list(tools)
            return out

    def _maybe_check_render_equivalence(self, enc, text, messages, thinking_mode, drop_thinking):
        """Warn (once) if ``bos + Σ render_message`` drifts from ``enc.encode_messages``.

        The loss mask assumes the per-message render concatenation reproduces the canonical
        single-shot ``encode_messages`` (the served path) — but this file is the only caller
        of the low-level ``render_message``, and the equivalence cannot be proven offline
        (``encoding_dsv4`` ships with sglang). So cross-check it at runtime on the first
        sample, where the real encoder exists. Warn-only by design: a guard that cannot run
        must never block training, and a genuine mismatch is surfaced loudly for the operator
        to investigate rather than silently mis-masking. ``messages`` is the (tool-injected)
        list BEFORE merge/sort — ``encode_messages`` does that preprocessing itself.
        """
        if getattr(self, "_render_checked", False):
            return
        self._render_checked = True
        encode_messages = getattr(enc, "encode_messages", None)
        if encode_messages is None:
            return
        import copy

        try:
            canonical = encode_messages(
                copy.deepcopy(messages),
                thinking_mode=thinking_mode,
                drop_thinking=drop_thinking,
                add_default_bos_token=True,
            )
        except Exception as exc:  # noqa: BLE001 — a guard that can't run must not break training
            logger.warning("dsv4 mask: could not cross-check render against encode_messages (%s); skipping", exc)
            return
        if canonical != text:
            logger.warning(
                "dsv4 mask: per-message render diverged from encode_messages (assembled %d chars vs "
                "canonical %d). The loss mask may be misaligned with served tokens — verify the V4 "
                "encoder/config. assembled[:160]=%r canonical[:160]=%r",
                len(text),
                len(canonical),
                text[:160],
                canonical[:160],
            )

    def gen_multi_turn_loss_mask_deepseek_v4(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        """Build the loss mask from per-message render spans.

        Mask=1 over the render span of every message whose role passes
        :meth:`_should_train_piece` (base: assistant only — reasoning + content +
        tool_calls + eos), mask=0 elsewhere (the leading bos, untrained roles, and
        special tokens that carry no char offset). ``tools`` are first injected into the
        system message (matching serving), so the ``<functions>`` schema is part of the
        system span and stays mask=0."""
        import os

        enc = self._load_dsv4_encoding()

        thinking_mode = os.environ.get("MILES_DSV4_THINKING_MODE", "chat")
        drop_thinking = os.environ.get("MILES_DSV4_DROP_THINKING", "0") == "1"
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"MILES_DSV4_THINKING_MODE must be 'chat' or 'thinking', got {thinking_mode!r}")

        # Splice tools into the system message exactly like the served render path
        # (chat_template_utils.deepseek_v4.render_messages -> _inject_tools_into_system),
        # BEFORE merge/sort/render. SFT data passes tools as a separate metadata list, so
        # without this the rendered text (== sample.tokens) would omit the <functions>
        # tool-schema block the model is served with. No-op when there are no tools.
        messages = self._inject_tools(messages, tools)

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

        # Safety net: on the first sample, confirm this per-message assembly matches the
        # canonical encode_messages (warn-only; see method docstring).
        self._maybe_check_render_equivalence(enc, text, messages, thinking_mode, drop_thinking)

        # Tokenize whole conversation with offsets, so we can map char spans -> token spans.
        enc_out = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = enc_out["input_ids"]
        offsets = enc_out["offset_mapping"]
        loss_mask = [0] * len(token_ids)

        for start, end, role in pieces:
            if not self._should_train_piece(role):
                continue
            # Mark tokens whose offset is inside [start, end). offsets[k] = (s, e).
            for k, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue  # special tokens with no offset
                if s >= start and e <= end:
                    loss_mask[k] = 1

        # Per-message override (consistent with other branches).
        # If a single trained turn has step_loss_mask=0, zero out that span.
        for (start, end, role), msg in zip(pieces, full_messages, strict=True):
            if not self._should_train_piece(role):
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


class DeepSeekV4AllButSystemLossMaskGenerator(DeepSeekV4LossMaskGenerator):
    """V4 SFT mask that trains on *everything except* ``system`` turns.

    Reuses the base class' render-span machinery (encoding source, message
    pre-processing, char->token offset mapping); only the trained-role policy differs:
    every non-system message (user + assistant + tool, including its role markers /
    reasoning / eos) is mask=1, while ``system`` spans and the leading bos stay mask=0.
    ``step_loss_mask=0`` on any trained turn still zeroes that turn's span.

    Note: ``tool`` (tool-result) turns are trained too, per the "all but system"
    contract; the tool-*schema* (``<functions>``) is injected into the system message
    by the base class, so it correctly stays mask=0. Use
    :class:`DeepSeekV4LossMaskGenerator` instead if you only want assistant turns in
    the loss.
    """

    def _should_train_piece(self, role: str) -> bool:
        return role != "system"


TOKENIZER = None
PROCESSOR = None
# One cached generator per mask class; one "sample printed" guard per entry point.
# (A single training run uses one --rollout-function-path, but keying by class keeps
# the two entry points independent if both are ever exercised in the same process.)
_MASK_GENERATORS: dict = {}
_SAMPLE_PRINTED: set = set()


def _run_sft_rollout(args, data_buffer, mask_generator_cls, log_prefix):
    """Shared SFT rollout body for the V4 masks.

    Identical to ``miles.rollout.sft_rollout.generate_rollout`` except the loss mask is
    always built by ``mask_generator_cls`` (a :class:`DeepSeekV4LossMaskGenerator`
    subclass), so ``--loss-mask-type`` is ignored; the class selects which V4 mask
    (assistant-only vs all-but-system) to apply.
    """
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    mask_generator = _MASK_GENERATORS.get(mask_generator_cls)
    if mask_generator is None:
        mask_generator = mask_generator_cls(TOKENIZER)
        _MASK_GENERATORS[mask_generator_cls] = mask_generator

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample in enumerate(samples):
        (sample,) = sample
        messages = sample.prompt
        tools = sample.metadata.get("tools", None)

        token_ids, loss_mask = mask_generator.get_loss_mask(messages, tools=tools)

        response_length = mask_generator.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]

        if i == 0 and log_prefix not in _SAMPLE_PRINTED:
            logger.info(
                f"{log_prefix}::generate_rollout example data: {sample=} (raw){messages=} (raw){token_ids=} (raw){loss_mask=} {response_length=}"
            )
            _SAMPLE_PRINTED.add(log_prefix)

    return samples


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """DeepSeek-V4 SFT rollout — assistant-only loss mask.

    Mirror of ``miles.rollout.sft_rollout.generate_rollout``, but always builds the
    loss mask with :class:`DeepSeekV4LossMaskGenerator` (mask=1 over assistant turns
    only; ignores ``--loss-mask-type``).

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[Sample]: a list of samples generated by the rollout
    """
    assert not evaluation
    return _run_sft_rollout(args, data_buffer, DeepSeekV4LossMaskGenerator, "sft_rollout_dsv4")


def generate_rollout_all_but_system(args, rollout_id, data_buffer, evaluation=False):
    """DeepSeek-V4 SFT rollout — all-but-system loss mask.

    Same as :func:`generate_rollout` but builds the mask with
    :class:`DeepSeekV4AllButSystemLossMaskGenerator` (mask=1 over every non-system
    turn — user + assistant + tool; only ``system`` spans stay 0).

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[Sample]: a list of samples generated by the rollout
    """
    assert not evaluation
    return _run_sft_rollout(
        args, data_buffer, DeepSeekV4AllButSystemLossMaskGenerator, "sft_rollout_dsv4_all_but_system"
    )
