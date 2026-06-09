from transformers import AutoTokenizer


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # return the lengths starting from the first occurrence of 1 to the end of each loss mask
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen"):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        # Skip Qwen-style probe for V4: V4 has no jinja chat_template (uses
        # encoding_dsv4.py instead), so apply_chat_template would raise.
        if tokenizer_type == "deepseek_v4":
            self.system_message_length = 0
            self.gen_token_length = 0
        else:
            self.system_message_length, self.gen_token_length = self.get_system_message_length()

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return get_response_lengths(loss_masks)

    def find_all_sublist_indices(self, main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            if i == 0:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True, tools=tools)
            else:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True)

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
        prefix_token_ids = self.tokenizer.apply_chat_template([prefix_message], tokenize=True)

        for i, message in enumerate(messages):
            if i == 0:
                tailed_message_ids = self.tokenizer.apply_chat_template(
                    [message, prefix_message], tokenize=True, tools=tools
                )
                message_ids = tailed_message_ids[: -len(prefix_token_ids)]
            else:
                prefixed_message_ids = self.tokenizer.apply_chat_template([prefix_message, message], tokenize=True)
                message_ids = prefixed_message_ids[len(prefix_token_ids) :]

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_distill_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, tools=tools
        )
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask

    def gen_multi_turn_loss_mask_deepseek_v4(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        """V4 ships chat template as a Python module (encoding_dsv4.py) instead of jinja.

        Why:
          tokenizer_config.json's chat_template is empty for DeepSeek-V4 — the
          official path is `encoding/encoding_dsv4.py` in the model repo.
        How to apply:
          loaded lazily from `tokenizer.name_or_path/encoding/encoding_dsv4.py`.
          Mask=1 for entire assistant render span (reasoning + content + tool_calls + eos),
          mask=0 elsewhere (incl. transition tokens like `<｜Assistant｜>` / `<think>`).

        Configurable via env vars:
          MILES_DSV4_THINKING_MODE = "chat" (default) | "thinking"
          MILES_DSV4_DROP_THINKING = "0" (default) | "1"
        """
        import os
        import sys
        import importlib.util

        enc = getattr(self, "_dsv4_encoding", None)
        if enc is None:
            tk_path = getattr(self.tokenizer, "name_or_path", None) or ""
            enc_path = os.path.join(tk_path, "encoding", "encoding_dsv4.py")
            if not os.path.isfile(enc_path):
                raise FileNotFoundError(
                    f"DeepSeek-V4 chat-template module not found at {enc_path}. "
                    "Pass --hf-checkpoint pointing to a V4 HF dir that contains encoding/encoding_dsv4.py."
                )
            spec = importlib.util.spec_from_file_location("encoding_dsv4", enc_path)
            enc = importlib.util.module_from_spec(spec)
            sys.modules["encoding_dsv4"] = enc
            spec.loader.exec_module(enc)
            self._dsv4_encoding = enc

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

        # Tokenize whole conversation with offsets, so we can map char spans → token spans.
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
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)

            return self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        elif self.tokenizer_type == "deepseek_v4":
            return self.gen_multi_turn_loss_mask_deepseek_v4(messages, tools)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_loss_mask_with_multimodal_alignment(
        self, messages: list[dict], input_ids: list[int], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        text = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text.append({"role": msg["role"], "content": " ".join(text_parts)})
            else:
                text.append(msg)

        _, loss_mask_text = self.get_loss_mask(text, tools=tools)

        diff = len(input_ids) - len(loss_mask_text)
        assert diff >= 0, (
            f"input_ids (length={len(input_ids)}) is shorter than text loss_mask (length={len(loss_mask_text)}) "
            f"Please check if processor and tokenizer tokenization are consistent."
        )
        loss_mask = [0] * diff + loss_mask_text

        return input_ids, loss_mask

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts
