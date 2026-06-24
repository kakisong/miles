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

Invalid-sample handling (both rollouts; a masked sample is *invalid* when its loss mask
is all-zero — nothing to train on — or its token sequence is over-length)::

      MILES_DSV4_SFT_SKIP_INVALID = "0" (default) | "1"
          0 -> raise a clear error naming the first invalid sample
          1 -> drop invalid samples AND refill from the data buffer so each rollout still
               returns exactly ``rollout_batch_size`` valid samples (fixed global batch
               size — no dynamic gbs, no downstream trim); drops are logged.
      MILES_DSV4_SFT_MAX_TOKENS   = "0" (default: length check off) | <int>
          when > 0, a sample with len(tokens) > N counts as over-length
      MILES_DSV4_SFT_REFILL_MAX_FACTOR = "10" (default) | <float>
          refill safety cap: pull at most N * rollout_batch_size groups before giving up
          (guards a dataset where almost everything is invalid).
"""

import logging

from miles.rollout.base_types import RolloutFnTrainOutput
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
# Running per-entry-point tally of dropped invalid samples (by reason), accumulated
# across every rollout batch in this process. The rollout runs in a single long-lived
# RolloutManager actor, so this is the whole run's total; the warning below echoes it
# each time a drop happens, so the last `cumulative dropped=` line is the run total
# (monotonic) — no need to sum the per-batch warnings.
_SKIP_TOTALS: dict = {}


def _invalid_sample_reason(token_ids: list[int], response_length: int, max_tokens: int) -> str | None:
    """Return a reason string if a masked SFT sample is invalid, else ``None``.

    Two failure modes that crash training downstream:
      * empty loss mask — nothing to train on (``response_length == 0``, i.e. no token
        has mask=1). Such a sample has zero trainable tokens; it also trips the
        ``len(loss_mask) == response_length`` check in train_data_conversion because
        ``loss_mask[-0:]`` slices the *whole* mask rather than an empty one.
      * over-length — ``len(token_ids) > max_tokens`` (only when ``max_tokens > 0``);
        the packed sequence overflows the training token budget.
    """
    if response_length == 0:
        return "empty-loss-mask"
    if max_tokens > 0 and len(token_ids) > max_tokens:
        return f"over-length ({len(token_ids)} > {max_tokens} tokens)"
    return None


def _process_group(group, mask_generator, max_tokens, skip_invalid, log_prefix, dropped):
    """Mask one sample group; return the (mutated) group if valid, else ``None``.

    Valid: writes ``tokens / response_length / reward / loss_mask`` onto the sample and
    returns the group. Invalid (see :func:`_invalid_sample_reason`): when ``skip_invalid``
    raise a clear error naming the offender; otherwise tally the reason into ``dropped`` and
    return ``None`` so the caller can pull a replacement.
    """
    (sample,) = group
    messages = sample.prompt
    tools = sample.metadata.get("tools", None)

    token_ids, loss_mask = mask_generator.get_loss_mask(messages, tools=tools)
    response_length = mask_generator.get_response_lengths([loss_mask])[0]

    reason = _invalid_sample_reason(token_ids, response_length, max_tokens)
    if reason is not None:
        if not skip_invalid:
            hint = "." if max_tokens else "; set MILES_DSV4_SFT_MAX_TOKENS=<N> to also filter over-length."
            raise ValueError(
                f"{log_prefix}: invalid SFT sample ({reason}). "
                f"Set MILES_DSV4_SFT_SKIP_INVALID=1 to drop such samples{hint}"
            )
        kind = reason.split(" ", 1)[0]  # "empty-loss-mask" | "over-length"
        dropped[kind] = dropped.get(kind, 0) + 1
        return None

    sample.tokens = token_ids
    sample.response_length = response_length
    sample.reward = 0
    sample.loss_mask = loss_mask[-response_length:]
    return group


def _run_sft_rollout(args, data_buffer, mask_generator_cls, log_prefix):
    """Shared SFT rollout body for the V4 masks.

    Mirrors ``miles.rollout.sft_rollout.generate_rollout`` except the loss mask is always
    built by ``mask_generator_cls`` (a :class:`DeepSeekV4LossMaskGenerator` subclass), so
    ``--loss-mask-type`` is ignored; the class selects which V4 mask (assistant-only vs
    all-but-system) to apply.

    Invalid samples (empty loss mask / over-length — see :func:`_invalid_sample_reason`)
    are controlled by env vars (same ``MILES_DSV4_*`` convention as the thinking knobs):
      * ``MILES_DSV4_SFT_SKIP_INVALID=1`` drops them, then **refills** from the data buffer
        so the rollout returns exactly ``rollout_batch_size`` valid samples — the global
        batch size stays fixed (no dynamic gbs, no downstream trim). The default ``0`` raises
        a clear error naming the first offender.
      * ``MILES_DSV4_SFT_MAX_TOKENS=<N>`` enables the over-length check (``0`` = off).
      * ``MILES_DSV4_SFT_REFILL_MAX_FACTOR=<F>`` caps refilling at ``F * rollout_batch_size``
        groups pulled (default 10); exceeding it raises rather than looping forever on a
        mostly-invalid dataset.

    Refilling is safe and resumable: ``data_buffer.get_samples`` advances the dataset offset
    deterministically (wrapping epochs) and persists it via save/load, so pulling extra just
    consumes a little more data.
    """
    import os

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

    skip_invalid = os.environ.get("MILES_DSV4_SFT_SKIP_INVALID", "0") == "1"
    raw_max = os.environ.get("MILES_DSV4_SFT_MAX_TOKENS", "0") or "0"
    try:
        max_tokens = int(raw_max)
    except ValueError as exc:
        raise ValueError(f"MILES_DSV4_SFT_MAX_TOKENS must be an integer, got {raw_max!r}") from exc

    raw_factor = os.environ.get("MILES_DSV4_SFT_REFILL_MAX_FACTOR", "10") or "10"
    try:
        refill_factor = float(raw_factor)
    except ValueError as exc:
        raise ValueError(f"MILES_DSV4_SFT_REFILL_MAX_FACTOR must be a number, got {raw_factor!r}") from exc

    target = args.rollout_batch_size
    # Refill safety cap: never pull more than refill_factor * target groups in total.
    max_pull = max(target, int(target * refill_factor))

    kept: list = []
    dropped: dict[str, int] = {}
    pulled = 0
    # Pull -> filter -> refill the shortfall until we have `target` valid groups. When
    # skip_invalid is off, _process_group raises on the first invalid sample so this never
    # iterates twice; when all samples are valid, the first pull already fills the batch.
    while len(kept) < target:
        groups = data_buffer.get_samples(target - len(kept))
        if not groups:
            break
        pulled += len(groups)
        for group in groups:
            out_group = _process_group(group, mask_generator, max_tokens, skip_invalid, log_prefix, dropped)
            if out_group is None:
                continue
            kept.append(out_group)
            if len(kept) == 1 and log_prefix not in _SAMPLE_PRINTED:
                (sample,) = out_group
                logger.info(
                    f"{log_prefix}::generate_rollout example data: {sample=} (raw){sample.prompt=} (raw){sample.tokens=} (raw){sample.loss_mask=} {sample.response_length=}"
                )
                _SAMPLE_PRINTED.add(log_prefix)
            if len(kept) == target:
                break
        if pulled >= max_pull and len(kept) < target:
            raise ValueError(
                f"{log_prefix}: could only collect {len(kept)}/{target} valid SFT samples after "
                f"pulling {pulled} (cap {max_pull} = {refill_factor}x rollout_batch_size); the data "
                f"is mostly invalid or MILES_DSV4_SFT_MAX_TOKENS is too strict — raise "
                f"MILES_DSV4_SFT_REFILL_MAX_FACTOR to pull more."
            )

    if dropped:
        totals = _SKIP_TOTALS.setdefault(log_prefix, {})
        for kind, count in dropped.items():
            totals[kind] = totals.get(kind, 0) + count
        logger.warning(
            "%s: dropped %d invalid SFT samples this rollout (%s); refilled to %d valid "
            "(pulled %d total). cumulative dropped=%d (%s)",
            log_prefix,
            sum(dropped.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())),
            len(kept),
            pulled,
            sum(totals.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
        )

    # Surface the skip tally to wandb's `rollout/` section (one panel per key). Both keys are
    # cumulative over the run (the rollout lives in one RolloutManager actor, so _SKIP_TOTALS is
    # the whole run's tally): the running count of empty-mask and over-length samples dropped so
    # far. Emitting them every rollout (even at 0) keeps the curves continuous. Returning a
    # RolloutFnTrainOutput rather than a bare list is the supported way for a rollout fn to
    # contribute metrics: call_rollout_fn passes it through and _get_rollout_data forwards
    # `.metrics` to log_rollout_data -> tracking_utils.log.
    totals = _SKIP_TOTALS.get(log_prefix, {})
    metrics = {
        "rollout/sft_skipped_empty_mask": totals.get("empty-loss-mask", 0),
        "rollout/sft_skipped_over_length": totals.get("over-length", 0),
    }
    return RolloutFnTrainOutput(samples=kept, metrics=metrics)


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
