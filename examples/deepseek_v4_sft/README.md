# DeepSeek-V4-Flash SFT on 8×8 H20 — 操作手册

本目录用于在 **8 节点 × 8 张 H20(96GB)= 64 卡** 上验证 **DeepSeek-V4-Flash (284B)** 的 SFT 后训练。

> **状态提醒**:V4 模型代码尚未合入主分支,本手册基于 [PR #1045](https://github.com/radixark/miles/pull/1045) + [PR #1065](https://github.com/radixark/miles/pull/1065)。 SFT 通路在 PR review 中存在已知阻塞点(chat template / loss mask),所以**必须先做 Stage A 烟测**。

---

## 0. 文件清单

| 文件 | 用途 | 何时跑 |
|---|---|---|
| `prepare_data.py` | 把任意 SFT 数据转为 Miles 期望的 `messages` parquet | 一次 |
| `verify_chat_template.py` | **必读 / 必跑**。 验证 V4 chat template + loss mask 是否正确 | Stage A 之前 |
| `prepare_megatron_ckpt.sh` | FP8→BF16 + HF→Megatron torch_dist | 一次(每变体一次) |
| `run_stage_a_smoke.sh` | 单节点 8 卡,4-layer,通路烟测 | Stage A |
| `run_stage_a2_correctness.md` | 4-layer 收敛 + 确定性回归(命令在 §4.5,无独立脚本) | Stage A2 |
| `run_stage_b0_dryrun.sh` | 8 节点 64 卡,真规模 50 步空跑 | Stage B0 |
| `run_stage_b_full.sh` | 8 节点 64 卡,V4-Flash 284B 全量 SFT | Stage B |
| `run_stage_b_eval.md` | 训练完调 `examples/eval/` 跑指标(命令在 §7,无独立脚本) | Stage B-eval |

## 0.1 验证 ladder

通路点亮 → 数值正确 → 真规模可起 → 全量训练 → 效果验收,五级闸门:

```
 Stage A  ──> Stage A2  ──> Stage B0  ──> Stage B  ──> Stage B-eval
 烟测           确定性          dry-run       全量SFT       后训评测
 1 节点        1 节点          8 节点         8 节点        推理集群
 4-layer       4-layer         284B          284B         284B
 30 min        2-4 h           1-2 h         数天          数小时
```

每一级失败都不要往下走;Stage A2/B0 看似多花时间,但能在便宜阶段就 catch 算子/显存/IB 问题。

---

## 1. 前置准备

### 1.1 代码 / 分支

```bash
cd /root/miles
git fetch origin pull/1045/head:v4-rl
git checkout v4-rl

# 如需要 TITO 一并(可选,只在用到 TITO 卡型/特性时需要)
git fetch origin pull/1065/head:v4-tito
git cherry-pick v4-tito        # 解冲突后继续
```

PR #1045 落地后会出现以下文件,本手册的脚本依赖它们:

```
scripts/models/deepseek-v4-flash.sh             # 284B MODEL_ARGS
scripts/models/deepseek-v4-flash-4layer.sh      # 4 层调试变体
miles_plugins/mbridge/deepseekv4.py             # mbridge 注册
miles_plugins/models/deepseek_v4/               # TileLang sparse-MLA / indexer
miles/backends/megatron_utils/megatron_to_hf/deepseekv4.py
```

`pip install -e . --no-deps` 后这些 plugin 会被 `import miles_plugins.mbridge` 自动注册。

### 1.2 容器

```bash
docker pull radixark/miles:latest
docker run -it --gpus all --network=host --shm-size=200g \
    -v /data:/data -v /root/miles:/root/miles \
    radixark/miles:latest bash
```

### 1.3 环境变量约定(所有脚本都吃这套)

```bash
# 推荐写到一份 setup_env.sh,每个节点 source 一次

export BASE_FOLDER=/data/miles                    # 共享根
export MODELS=$BASE_FOLDER/models
export DATA=$BASE_FOLDER/datasets
export OUT=$BASE_FOLDER/outputs/v4-flash-sft
export REPO=/root/miles
export MEGATRON_PATH=/root/Megatron-LM

export MASTER_ADDR=<head-node-ip>                 # head 节点 IP
export NUM_NODES=8                                # Stage B 用 8;Stage A 用 1
export NUM_GPUS_PER_NODE=8

mkdir -p $MODELS $DATA $OUT
```

> H20 集群通常起 NCCL 调优:`NCCL_IB_DISABLE=0`、`NCCL_IB_GID_INDEX=3`(按集群文档调),脚本里不强写,留给运维 export。

### 1.4 已知阻塞点(开 SFT 前必读)

#### A. `loss_mask_type` 不含 deepseek_v4

`miles/utils/mask_utils.py` 当前只支持 `qwen / qwen3 / distill_qwen`。 V4 chat template 与 Qwen 不一致,沿用 qwen 类型大概率把 `<|user|>` / `<|assistant|>` / `<think>` 段错切。 **这是必修项**。

修复方式:在 `MultiTurnLossMaskGenerator` 加一个 `gen_multi_turn_loss_mask_deepseek_v4`,逻辑参考 `gen_multi_turn_loss_mask_qwen3`,但用 V4 真实的 BOS / EOS / role token。 `verify_chat_template.py` 会把当前实现与人工标注的 ground truth 做对照,直接告诉你哪几个 token 应该被 mask 没被 mask(或反之)。

#### B. PR #1045 的 chat-template TODO

PR review:
> "TODO: fix apply chat template, use their py instead of hf tokenizer"

意思是 V4 官方推荐用模型仓里自带的 `chat_template.py`(或 jinja),HF tokenizer 默认 `apply_chat_template` 行为不一致。 本手册脚本默认用 `--chat-template-path`,把 V4 官方 jinja 拷到 `$REPO/templates/deepseek_v4.jinja` 下使用,而不是依赖 tokenizer 内置模板。

#### C. `reasoning_content` 在多轮里要保留

V4 多轮里 assistant 的 `reasoning_content` 必须传回(参考 [BerriAI/litellm#26395](https://github.com/BerriAI/litellm/issues/26395))。 SFT 数据如果含 think+answer:
- 想训 thinking:`reasoning_content` 段 mask=1
- 只想训 answer:`reasoning_content` 段 mask=0,answer 段 mask=1

`prepare_data.py` 的 `--mask-mode` 控制这一点。

---

## 2. 数据准备

```bash
# 示例:OpenHermes-2.5(纯对话,无 thinking)
python examples/deepseek_v4_sft/prepare_data.py \
    --source teknium/OpenHermes-2.5 \
    --output  $DATA/openhermes_v4.parquet \
    --mask-mode answer-only

# 示例:含 thinking 的自定义数据
python examples/deepseek_v4_sft/prepare_data.py \
    --source /path/to/your_sft.jsonl \
    --output  $DATA/your_v4_sft.parquet \
    --mask-mode include-thinking
```

输出 schema(`messages` 字段就是 V4 的 OpenAI 风格对话):

```python
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant",
     "reasoning_content": "...",   # 可选
     "content": "..."}
  ],
  # 可选:"step_loss_mask" 整轮关 mask
}
```

---

## 3. Ckpt 准备(单次)

### 3.1 下载 + FP8→BF16

```bash
hf download deepseek-ai/DeepSeek-V4-Flash --local-dir $MODELS/DeepSeek-V4-Flash

python $REPO/tools/fp8_cast_bf16.py \
    --input-fp8-hf-path $MODELS/DeepSeek-V4-Flash \
    --output-bf16-hf-path $MODELS/DeepSeek-V4-Flash-bf16
```

### 3.2 HF → Megatron torch_dist

**Stage A(4-layer,1 节点)**:

```bash
bash examples/deepseek_v4_sft/prepare_megatron_ckpt.sh 4layer
# 产物:$MODELS/DeepSeek-V4-Flash-4layer_torch_dist
```

**Stage B(全量 284B,需 1 节点临时跑转换,EP=16/PP=4 切)**:

```bash
bash examples/deepseek_v4_sft/prepare_megatron_ckpt.sh full
# 产物:$MODELS/DeepSeek-V4-Flash_torch_dist
```

转完后 rsync 到每个节点的本地 SSD,避免 64 卡同时打共享存储:

```bash
for IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
  ssh $IP "mkdir -p /root/local_data && \
           rsync -a $MODELS/DeepSeek-V4-Flash-bf16/         /root/local_data/ && \
           rsync -a $MODELS/DeepSeek-V4-Flash_torch_dist/   /root/local_data/V4-Flash_torch_dist/"
done
```

---

## 4. Stage A — 单节点 4-layer 烟测

**目标**:90 分钟内验证 chat template + loss mask + 前向反向 + ckpt save/load 全通。

### 4.1 离线验证 chat template(不上 GPU)

```bash
python examples/deepseek_v4_sft/verify_chat_template.py \
    --hf-checkpoint $MODELS/DeepSeek-V4-Flash-bf16 \
    --chat-template-path $REPO/templates/deepseek_v4.jinja \
    --sample-data       $DATA/openhermes_v4.parquet \
    --num-samples 5 \
    --loss-mask-type    qwen3        # 先试 qwen3,看 mask 是否正确
```

输出长这样(节选):

```
[sample 0]
TOKEN  | MASK | DECODED
1234   |  0   | <|user|>
5678   |  0   | Hello
...
9012   |  1   | <|assistant|>     <-- ❌ 这个不该是 1,assistant 起手 tag 应该 mask=0
3456   |  1   | Sure
```

**判定**:
- 全部样本 `<|assistant|>` / `<think>` / `</think>` 等 role-tag 是 mask=0
- 仅 assistant 正文(以及 thinking,看你 `--mask-mode`)是 mask=1
- `<|end|>` / `<|eos|>` 一般 mask=1(参与训练结束信号)

如果有错,说明现有 `qwen3` 分支不通用——按手册第 1.4 节的修复方式给 `mask_utils.py` 加 `deepseek_v4` 分支,然后切 `--loss-mask-type deepseek_v4` 重跑直到全绿。

### 4.2 启动烟测训练(单节点 8 卡)

```bash
export NUM_NODES=1
bash examples/deepseek_v4_sft/run_stage_a_smoke.sh
```

预期:
- 启动 < 5 分钟(无 SGLang)
- 单步 < 30s,`grad_norm` 在合理范围(0.5–5)
- loss 单调下降(50 步内能看到变化)
- 跑 100 步触发 `--save-interval 50` 至少落两次盘
- `kill -9` 后从最近 ckpt resume,loss 不跳

只要 4.1 + 4.2 全绿,就可以进 Stage A2。

### 4.5 Stage A2 — 确定性 + 微收敛回归(可选但强烈建议)

**目的**:catch 非确定性 bug、算子精度回归。 用 4-layer 这个便宜模型,跑两次完全相同配置,要求 loss 序列 bit-wise 一致;再跑一次更长(500 步)的微收敛,要求 loss 单调下降趋势与 Qwen3-4B 同数据 baseline 形状一致。

**机制**(参考 `examples/reproducibility/README.md`):
```bash
pip uninstall flash_attn_3 -y     # FA3 在某些 path 上非确定
```

在 `run_stage_a_smoke.sh` 启动前 export 几个变量,临时打开 deterministic mode:

```bash
export EXTRA_TRAIN_ARGS="--deterministic-mode"
export EXTRA_ENV_VARS_NCCL_ALGO=Ring
export EXTRA_ENV_VARS_NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export EXTRA_ENV_VARS_CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

> 当前 `run_stage_a_smoke.sh` 还没消费 `EXTRA_TRAIN_ARGS`/`EXTRA_ENV_VARS_*`,需要时手动在脚本最后的 `python3 train_async.py` 后面追加 `--deterministic-mode`,并在 `RUNTIME_ENV_JSON` 的 `env_vars` 里加上 NCCL/NVTE/CUBLAS 三项。 README 里这样描述是为了避免现在污染主路径的 smoke 脚本。

**跑法**:
```bash
# 1) 第一次确定性烟测
NUM_NODES=1 RUN_TAG=det1 bash examples/deepseek_v4_sft/run_stage_a_smoke.sh

# 2) 第二次相同配置
NUM_NODES=1 RUN_TAG=det2 bash examples/deepseek_v4_sft/run_stage_a_smoke.sh

# 3) bit-wise 比较两次的 loss 序列
diff <(jq '.loss' $OUT/det1*/dump_details/iter_*.jsonl) \
     <(jq '.loss' $OUT/det2*/dump_details/iter_*.jsonl)
```

**判定**:diff 必须全空。 不一致就是有非确定算子(常见是 sparse-MLA 的 atomic add、moe permute,或 NCCL 算法分歧)。

**微收敛**: 把 `--num-rollout 500` 跑完,导出 loss 曲线,与同数据下 `scripts/run-qwen3-4B.sh` 跑 500 步的曲线形状对比。 不要求绝对值相同(模型不一样),要求**形状相似**:warmup 后下降、稳态接近水平、grad_norm 不发散。

---

## 5. Stage B0 — 64 卡真规模 dry-run

**目的**:在不烧几天训练的前提下,验证 64 卡上 V4-Flash 284B 的并行/显存/IB/ckpt 全部能起。 跑 50 步(约 1-2 h)。

```bash
NUM_NODES=8 bash examples/deepseek_v4_sft/run_stage_b0_dryrun.sh
```

**通过门槛**(全部满足才能进 Stage B):

| 项 | 阈值 | 失败处理 |
|---|---|---|
| 启动到第 1 步 | < 15 min | 慢则 NCCL/IB 没起,查 `NCCL_DEBUG=INFO` |
| 第 1 步 wall-time | < 90s | 超过则 EP/PP 切错或重计算未生效 |
| 稳态 step 时长 | 30-50s | > 60s 见 README §6 |
| 单卡显存峰值 | < 92 GB | 超过则 `--max-tokens-per-gpu` 4096→3072 |
| save@iter25/50 | 两次都成功落盘 | 失败查共享存储 / `--save-retain-interval` |
| reload | 与 save 前一致 | 不一致看 ckpt 切分参数与 PERF_ARGS 是否对齐 |
| grad_norm | 无 spike(< 50)| spike 看 `--accumulate-allreduce-grads-in-fp32` 是否开 |
| NaN | 无 | 有则查 sparse-MLA softmax/mscale fp32 是否开 |

**为什么必须 dry-run 而不是直接上 Stage B**:OOM、IB 配置错、ckpt save fail 这类错误只有真规模才会暴露。 直接上全量训练撞上去就是浪费几小时 + 64 卡机时;dry-run 失败只丢 1-2 h。

---

## 6. Stage B — 8 节点 64 卡全量 SFT

### 5.1 启动

在 head 节点(MASTER_ADDR 那台):

```bash
export NUM_NODES=8
bash examples/deepseek_v4_sft/run_stage_b_full.sh
```

脚本里 ray head 和所有 worker 通过 `/root/mpi_rack_hostfile` 自动拉起,与 `run-qwen3-235B-A22B-sft.sh` 一致。

### 5.2 推荐并行(脚本默认值)

```
TP=4  PP=4  CP=2  EP=16  ETP=1   →   TP×PP×CP×DP = 4×4×2×2 = 64
```

为什么是这套:
- **TP=4**:V4 attention 重(MLA + indexer + sparse),拆 TP 降单卡激活;
- **PP=4**:61 层 → ~16 层/stage,首尾用 `--decoder-{first,last}-pipeline-num-layers 13` 微调;
- **EP=16**:256 experts / 16 = 每卡 16 个 expert,匹配 Roadmap 默认;
- **CP=2**:SFT 序列普遍长(含 thinking),H20 显存吃紧,必须切;
- **DP=2**:留给 EP 维度的全局通信。

global-batch 必须能被 DP=2 整除。 默认 256/2=128 micro-batches。

### 5.3 H20 vs H200 调整(已在脚本里)

| 项 | H200 默认 | H20 调整 | 原因 |
|---|---|---|---|
| `max-tokens-per-gpu` | 9216 | **4096** 起步 | H20 BF16 算力 ~1/6 |
| `--recompute-num-layers` | 1 | 1(保持) | 显存够,但收益大 |
| `--optimizer-cpu-offload` | 可选 | **必开** | 96GB 不够装 fp32 副本 |
| `--use-precision-aware-optimizer` | 推荐 | **必开** | 同上 |
| `--moe-enable-deepep` | 开 | **暂不开** | DeepEP 在 H20 上 path 未充分验证,SFT 不需要 |
| `--attention-backend flash` | 不要(MLA) | 不要 | V4 是 sparse-MLA,走 PR #1045 的 TileLang 路径 |

### 5.4 单步预算

64 张 H20,V4-Flash BF16 训练 + FP8 权重存,`max-tokens-per-gpu=4096`:
- 第 1 步 warmup 较慢可达 60-90s
- 稳态 30–50s/step
- 显存峰值预期 80–90GB(留 5–10GB 余量,撑不住先减 `max-tokens-per-gpu` 至 3072)

如果稳态 > 90s/step:
1. 看 `nvidia-smi` 是否有卡空跑 → EP/PP 切错
2. 看 NCCL 带宽日志是否下跌 → 跨节点 IB / GPUDirect 没起
3. CPU offload 上锁 → 减小 `--max-tokens-per-gpu` 或关 offload 看显存

---

## 7. SFT 关键开关速查

| 开关 | 作用 | 必填? |
|---|---|---|
| `--rollout-function-path miles.rollout.sft_rollout.generate_rollout` | 把 rollout 换成"从文件读" | ✅ |
| `--loss-type sft_loss` | 用 SFT loss 而非 PPO | ✅ |
| `--calculate-per-token-loss` | 在所有 unmasked token 上取均值 | ✅ |
| `--disable-compute-advantages-and-returns` | SFT 无 advantage,跳过 | ✅ |
| `--debug-train-only` | 不启动 SGLang(SFT 无需推理) | ✅ |
| `--rollout-batch-size == --global-batch-size` | 读一批训一批,不切 micro | 推荐 |
| `--num-epoch N` | epoch 数(替代 `--num-rollout`) | ✅ |
| 不要写 `--n-samples-per-prompt` | SFT 一对一 | ✅ |
| `--chat-template-path` | 用 V4 jinja 而非 tokenizer 内置 | ✅(因 PR TODO) |
| `--loss-mask-type deepseek_v4` | 修完 mask_utils 后切到 v4 | ✅(修完后) |

---

## 8. 验证标准(逐级 gate)

**Stage A.1(verify_chat_template)**: 5 个样本 mask 100% 正确。

**Stage A.2(单节点训练)**: 
- 100 步内 loss 下降 ≥ 30%(以小批数据计)
- save → kill → resume,resume 后第一步 loss 不超过 kill 前的 1.05×
- 单步 < 30s

**Stage B.1(64 卡第 1 步)**:
- 启动 < 15 分钟
- 第 1 步 token throughput ≥ 64 卡 × 4096 / 90s ≈ 2.9k tok/s/GPU(再低就排查)
- 显存峰值 < 92GB

**Stage B.2(整跑)**:
- eval 集 token-level PPL 在 1 个 epoch 内单调下降
- `grad_norm` 不出现持续 spike(> 50)
- 没有 NaN(loss 一旦 NaN,先看 sparse-MLA 内 mscale / softmax 是否走 fp32)

**Stage B.3(回归对照)**:
同数据用现成的 `scripts/run-qwen3-235B-A22B-sft.sh` 跑 100 步,曲线形状(loss/grad_norm 走势)应和 V4 一致——不一致说明问题大概率在 V4 路径而不是数据/超参。

---

## 9. 故障速查

| 症状 | 大概率原因 | 解 |
|---|---|---|
| `KeyError: deepseek_v4` 在 mbridge | PR #1045 没装 / `pip install -e .` 没跑 | 重装 plugin |
| 训练第一步 OOM | `max-tokens-per-gpu` 太大 | 4096→3072→2048 |
| 卡在 ckpt load | torch_dist 切分与 PERF_ARGS 不匹配 | 重新跑 `prepare_megatron_ckpt.sh full`,EP/PP 与训练侧一致 |
| `NaN` loss | sparse-MLA softmax/mscale 精度溢出 | 确保 `--attention-softmax-in-fp32` `--accumulate-allreduce-grads-in-fp32` 都开 |
| chat template 缺 token / 多 token | 用了 HF tokenizer 内置模板 | 切 `--chat-template-path $REPO/templates/deepseek_v4.jinja` |
| Loss 不降 | mask 全 0 或全 1 | 跑 `verify_chat_template.py` 复核 |
| `--apply-chat-template` 与 `--chat-template-path` 冲突 | Miles 二选一 | 只留 `--chat-template-path` |

---

## 10. Stage B-eval — 后训练效果回归

Stage B 跑完(`$OUT/$RUN_ID/iter_<final>`),你需要比 base V4-Flash 在 SFT 之后到底好了/坏了多少。

**做法**:转回 HF 格式,用 `examples/eval/` 跑标准评测集。

```bash
# 1) Megatron torch_dist → HF
PYTHONPATH=$MEGATRON_PATH python $REPO/tools/convert_torch_dist_to_hf.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint $MODELS/DeepSeek-V4-Flash-bf16 \
    --load $OUT/$RUN_ID \
    --save $OUT/$RUN_ID/hf

# 2) eval(参考 examples/eval/scripts/run-qwen3-32B.sh,改 model 路径即可)
#    nemo_skills 套件覆盖 GSM8K / MATH / HumanEval / MMLU
bash examples/eval/scripts/run-qwen3-32B.sh   # 复制改成 v4 版本
```

**回归基线**: 用同一份 eval 套件先评一下 `$MODELS/DeepSeek-V4-Flash-bf16` 自身,得到 baseline。 SFT 后:
- 通用集(MMLU / HellaSwag): SFT 之后允许 ≤ 1 pt 退化(SFT 调对话风格容易动)
- 任务集(你 SFT 数据对应的 domain,如 math/code): 应有显著提升,否则 SFT 数据/超参有问题
- 灾难性遗忘信号: 通用集退化 > 5pt 就停下重看 LR / epoch / mask

**确定性回归**: Stage A2 通过 ≠ Stage B 也确定。 如果你在 Stage B 也想要 bit-wise 复现,启动时也开 `--deterministic-mode`,代价是约 10-15% 速度损失。 一般只在 debug 数值异常时开。

---

## 11. 后续

- Stage B 完整跑通后,`$OUT/checkpoints/iter_*` 即 SFT 后训练产物,可用 `tools/convert_torch_dist_to_hf.py` 转回 HF 格式部署。
- 如果想接 RL,直接基于该 ckpt 切 `run_deepseek_v4.py`(PR #1045 自带)。
- 推荐到 [Issue #1046](https://github.com/radixark/miles/issues/1046) 反馈 SFT 通路结果,推动官方把 `loss_mask_type=deepseek_v4` 合入主分支。

---

## 参考

- [LMSYS Blog — DeepSeek-V4 on Day 0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [PR #1045 — DeepSeek V4 RL support](https://github.com/radixark/miles/pull/1045)
- [Issue #1046 — DeepSeek V4 RL Roadmap](https://github.com/radixark/miles/issues/1046)
- [docs/en/examples/qwen3-4b-base-openhermes.md](https://github.com/radixark/miles/blob/main/docs/en/examples/qwen3-4b-base-openhermes.md)
- [scripts/run-qwen3-235B-A22B-sft.sh](https://github.com/radixark/miles/blob/main/scripts/run-qwen3-235B-A22B-sft.sh)
