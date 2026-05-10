"""
Stage A 阻塞点验证:V4 chat template + loss mask 是否正确。

不上 GPU,不依赖 Megatron。在 head 节点 / 任何机器上都能跑。

输出每条样本的 (token_id, mask, decoded_text) 表格,人工核对:
  - role tag (<|user|>, <|assistant|>, <|system|>) 应 mask=0
  - assistant 正文(以及可选的 thinking)应 mask=1
  - 用户/系统 / tool 内容应 mask=0

如果发现错误,按 README §1.4 给 miles/utils/mask_utils.py 加
gen_multi_turn_loss_mask_deepseek_v4 分支,然后 --loss-mask-type deepseek_v4 重跑。

用法:
    python examples/deepseek_v4_sft/verify_chat_template.py \\
        --hf-checkpoint        $MODELS/DeepSeek-V4-Flash-bf16 \\
        --chat-template-path   $REPO/templates/deepseek_v4.jinja \\
        --sample-data          $DATA/openhermes_v4.parquet \\
        --num-samples 5 \\
        --loss-mask-type qwen3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from miles.utils.mask_utils import MultiTurnLossMaskGenerator


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", required=True, help="HF 模型目录,需含 tokenizer 文件")
    p.add_argument(
        "--chat-template-path",
        default=None,
        help="V4 官方 jinja 模板路径。 不传则用 tokenizer 内置(可能不对)",
    )
    p.add_argument("--sample-data", required=True, help="parquet/jsonl,含 messages 字段")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--loss-mask-type", default="qwen3", choices=["qwen", "qwen3", "distill_qwen", "deepseek_v4"])
    p.add_argument("--max-print-tokens", type=int, default=200, help="每条样本只打印前 N 个 token,避免刷屏")
    return p.parse_args()


def load_samples(path: str, n: int) -> list[list[dict]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        rows = df.head(n).to_dict(orient="records")
    elif suffix in (".jsonl", ".json"):
        rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                rows.append(json.loads(line))
    else:
        raise ValueError(f"unsupported sample format: {suffix}")

    out = []
    for r in rows:
        msgs = r.get("messages")
        if msgs is None:
            msgs = r.get("conversations")
        if msgs is None:
            raise ValueError(f"sample missing messages/conversations field: keys={list(r)}")
        out.append([dict(m) for m in msgs])
    return out


def patch_chat_template(tok: AutoTokenizer, jinja_path: str | None, mask_type: str) -> None:
    if mask_type == "deepseek_v4":
        # V4 不走 jinja,而是用 model 仓里的 encoding_dsv4.py(由 mask_utils 自动加载)。
        print(f"{GREEN}[ok]{RESET} V4: 使用模型自带 encoding_dsv4.py(无需 jinja)")
        return
    if jinja_path is None:
        print(f"{YELLOW}[warn]{RESET} 未指定 --chat-template-path,使用 tokenizer 内置模板。"
              " 如 mask 不对,先尝试用 V4 官方 jinja。")
        return
    with open(jinja_path) as f:
        tok.chat_template = f.read()
    print(f"{GREEN}[ok]{RESET} chat_template 已替换为 {jinja_path}")


def color_for_mask(m: int) -> str:
    return GREEN if m == 1 else DIM


def render(token_ids: list[int], mask: list[int], tok: AutoTokenizer, max_print: int) -> None:
    assert len(token_ids) == len(mask), f"len mismatch: {len(token_ids)} vs {len(mask)}"
    n_train = sum(mask)
    n_total = len(mask)
    pct = 100.0 * n_train / max(n_total, 1)
    print(f"  total tokens: {n_total}, masked-in (train): {n_train} ({pct:.1f}%)")
    head = f"  {'IDX':>4} | {'TOK_ID':>7} | {'MASK':>4} | DECODED"
    print(head)
    print("  " + "-" * (len(head) - 2))
    n_show = min(len(token_ids), max_print)
    for i in range(n_show):
        tid = token_ids[i]
        m = mask[i]
        try:
            txt = tok.decode([tid])
        except Exception:
            txt = "<decode-err>"
        txt = txt.replace("\n", "\\n")
        c = color_for_mask(m)
        print(f"  {c}{i:>4} | {tid:>7} | {m:>4} | {txt!r}{RESET}")
    if len(token_ids) > max_print:
        print(f"  ... ({len(token_ids) - max_print} more tokens, 用 --max-print-tokens 调大)")


def sanity_check(messages: list[dict], token_ids: list[int], mask: list[int]) -> list[str]:
    """Heuristic 静态检查,catch 最常见的错配。"""
    warnings = []
    if sum(mask) == 0:
        warnings.append("mask 全 0 — 训练时不会学到任何东西!")
    if all(mask):
        warnings.append("mask 全 1 — 用户/system token 也被算 loss,严重错!")
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if has_assistant and sum(mask) == 0:
        warnings.append("有 assistant 但 mask 全 0 — chat template 对 assistant 起手 tag 切错")
    return warnings


def main() -> int:
    args = parse_args()
    print(f"{DIM}[loading tokenizer from {args.hf_checkpoint}]{RESET}")
    tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    patch_chat_template(tok, args.chat_template_path, args.loss_mask_type)

    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type=args.loss_mask_type)
    print(f"{DIM}[mask generator: type={args.loss_mask_type}, system_message_length={gen.system_message_length}, "
          f"gen_token_length={gen.gen_token_length}]{RESET}")

    samples = load_samples(args.sample_data, args.num_samples)

    failed = 0
    for idx, messages in enumerate(samples):
        print(f"\n{YELLOW}=== sample {idx} (turns={len(messages)}) ==={RESET}")
        try:
            token_ids, mask = gen.get_loss_mask(messages)
        except Exception as e:
            print(f"  {RED}[error]{RESET} get_loss_mask raised: {e!r}")
            failed += 1
            continue
        render(token_ids, mask, tok, args.max_print_tokens)
        warns = sanity_check(messages, token_ids, mask)
        if warns:
            failed += 1
            for w in warns:
                print(f"  {RED}[warn]{RESET} {w}")
        else:
            print(f"  {GREEN}[ok]{RESET} 静态检查通过(仍需人工核对 token-by-token)")

    print()
    if failed:
        print(f"{RED}{failed}/{len(samples)} 个样本未通过自动检查 — 不要进入 Stage A 训练。{RESET}")
        return 1
    print(f"{GREEN}全部样本静态检查通过。 仍请人工核对前几条的 role-tag 切分是否正确。{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
