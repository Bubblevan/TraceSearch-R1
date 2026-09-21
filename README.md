# TraceSearch-R1

TraceSearch-R1 是一个用于研究多轮 Web Search Agent 的实验脚手架，当前关注两个问题：搜索过程中的失败如何影响后续轨迹，以及不同搜索步骤应当如何分配贡献度。

项目中的基本交互链路是：

```text
问题 → 思考 → search(query) → 证据 → visit(url) → 证据 → 最终回答
```

这里的搜索、访问和回答都被当作 Agent 轨迹的一部分保存下来。后续实验可以据此分析：Agent 在哪一步发起了搜索、拿到了什么结果、是否继续访问页面，以及最后答案是否带有引用。

## 当前范围

- 研究代码保持为轻量的 Python 实现，不复制某个训练框架。
- 搜索和页面访问通过小型工具接口接入，默认使用离线的确定性 fixture，测试不需要账号和网络。
- 现有指标覆盖答案正确性、引用情况、搜索轮次、工具失败和成本等维度。
- 当前提交不包含已验证的真实模型训练实验，也不对强化学习效果作出结论。
- M1 已提供 learned-policy、严格文本 action protocol、HTTP retriever、TrainingTrace 和 vanilla GRPO objective 的 Level-A 边界；真实模型 rollout 与 GRPO 更新仍需外部 model gateway、retriever、rLLM/verl 和可记录的实验资源。

## 目录结构

```text
src/tracesearch/
  agent/          多轮 Agent 控制循环和轨迹类型
  environment/    搜索、访问工具协议和离线工具
  rewards/        失败步骤与贡献度分配辅助函数
  evaluation/     轨迹级指标
  training/       M1 token trace、reward、group rollout 和 GRPO objective
tests/            离线单元测试
docs/             设计说明和实验计划
```

## 快速开始

```bash
# Linux/macOS
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
uv run --python .venv/bin/python pytest

# Windows PowerShell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
uv run --python .venv/Scripts/python.exe pytest
```

根目录 `.venv` 只负责离线单元测试和轻量 CLI。真实 M1-C
rLLM/veRL/vLLM 后端必须使用 Linux/WSL2 下单独的
`.venvs/tracesearch-m1c`；不要把后端依赖安装进根目录环境，也不要用
conda 环境代替它。

## 远程 Linux GPU 从零复现 M1-C/C2

远程服务器建议使用原生 Linux 文件系统（例如 `/home/<user>` 或
`/workspace`），不要把仓库放在 Windows `/mnt/c`、`/mnt/d` 挂载盘上。下面的
流程适用于单张 NVIDIA L40 48 GiB；当前 L40 配置是新硬件上的未验证 profile，
不能把它与历史 4090/16 GiB WSL 结果混写。

### 1. 克隆仓库并准备 uv

```bash
cd /workspace
git clone https://github.com/Bubblevan/TraceSearch-R1.git
cd TraceSearch-R1
git switch main

command -v uv
uv --version
nvidia-smi

# 根环境：只跑测试/轻量 CLI
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
uv run --python .venv/bin/python pytest
```

### 2. 安装固定的 M1-C uv 环境

不要手工拼接一长串 `pip install` 命令。安装脚本会创建
`.venvs/tracesearch-m1c`，使用 Python 3.12，安装锁定的 Torch 2.11、vLLM
0.22.1、rLLM/veRL Git commit、Ray、Hydra/OmegaConf 以及记录过哈希的
FlashAttention 社区 wheel，并执行导入检查：

```bash
bash scripts/bootstrap_m1c_uv.sh
source .venvs/tracesearch-m1c/bin/activate
python scripts/verify_m1c_env.py
```

依赖输入是 `configs/m1/m1c-requirements-linux.txt`，解析时使用
`configs/m1/m1c-uv-overrides.txt`，提交后的 Linux 安装锁是
`configs/m1/m1c-requirements-linux.lock.txt`。上游 veRL/rLLM 的
`numpy<2` 元数据冲突由这个显式 override 处理；不要删除它，也不要用
`uv pip install -e ".[m1]"` 重新解析后端。完整的 commit、版本、wheel
SHA256 和运行契约见 `configs/m1/m1c-backend-lock.json`。

这个 FlashAttention wheel 的前提是 Linux x86_64、CPython 3.12、CUDA 13、
Torch 2.11 和 CXX11 ABI。若远程机器的 `nvidia-smi`/CUDA 不满足这些条件，
先停下来为该机器建立新的经过验证的 lock；不要强行安装后把它报告为同一
环境。vLLM 是否实际使用 FlashAttention 也必须以 manifest/profile 为准，
不能仅凭“安装成功”宣称加速。

### 3. 运行十批 vanilla-GRPO stability smoke

先把模型放在远程 Linux 本地磁盘，并把路径替换为实际路径；不会由脚本自动
下载模型：

```bash
MODEL=/data/models/Qwen2.5-3B-Instruct
RUN_ID="m1-c2-l40-$(date +%Y%m%d-%H%M%S)"
mkdir -p runs

set -o pipefail
timeout 3600s python -u -m tracesearch.cli.run_m1c_backend \
  --phase c2 \
  --config configs/m1/runtime/l40-48g-linux.yaml \
  --model "$MODEL" \
  --dataset data/m1/c2_smoke.jsonl \
  --corpus data/m0/corpus.jsonl \
  --output "runs/$RUN_ID" \
  --task-count 10 \
  --group-size 4 \
  --max-turns 4 \
  --max-tokens 96 \
  --max-prompt-length 896 \
  --top-k 5 \
  --seed 42 \
  2>&1 | tee "runs/$RUN_ID/console.log"
```

原生 Linux 默认使用 CUDA IPC 权重传输，不要加
`--force-shm-weight-transfer`。这个 flag 是针对 WSL2 CUDA IPC 失效的显式
诊断/兼容开关；只有确认自己仍在 WSL2 且遇到对应的 IPC 错误时才使用。

通过的 C2 运行必须有完整的 10 行 `c2_steps.jsonl`、40 个 rollout slot、
每批 mask/advantage parity、`global_step_10` checkpoint、非零参数 delta 和
独立进程 reload 证据。零方差 batch 仍然是正常 vanilla-GRPO 循环的一部分，
不是删除样本或提前停止的理由。C2 只证明这个固定 fixture 上的数值与基础
设施稳定性，不是 benchmark、收敛或能力提升结论。

### 4. SGLang 只能另建环境

SGLang 的锁带有自己的 FlashAttention-4/Transformers 依赖，不能装进上面的
M1-C vLLM 环境：

```bash
uv venv --python 3.12 .venvs/tracesearch-sglang
uv pip sync --python .venvs/tracesearch-sglang/bin/python \
  configs/m1/sglang-requirements-linux.txt
```

除非实验明确要求 SGLang 对照，否则远程 L40 首次复现先使用 vLLM 路径；两种
环境的版本和结果必须分别记录。

### 常见环境错误

- `No module named hydra`：当前 Python 不是 `.venvs/tracesearch-m1c/bin/python`，
  或没有执行 bootstrap。
- `Primary config module 'rllm.trainer.config' not found`：rLLM 没装在当前
  解释器，先运行 `python scripts/verify_m1c_env.py`，不要只安装 Hydra。
- `EADDRINUSE`：上一次 Ray/训练进程仍占用 rendezvous port。确认没有其他
  实验后执行 `ray stop --force`，再用新的输出目录重跑。
- WSL 中出现 `CUDA error: invalid argument` 于 CUDA IPC handle 重建：这是
  WSL 运行时路径问题，使用 `--force-shm-weight-transfer`；原生 Linux 不应
  默认复制这个 workaround。
- FlashAttention import 失败：先检查 Python ABI、Torch/CUDA、wheel SHA256
  和 `python -c 'import flash_attn; print(flash_attn.__file__)'`；仓库中的
  `tracesearch.compat.flash_attn` 不是第三方包的替代安装。

远程 Codex 在任何训练前都应先记录：`git rev-parse HEAD`、
`git status --short`、`python -c 'import sys; print(sys.executable)'`、
`nvidia-smi`、模型绝对路径和完整 CLI。运行器会把 dirty 状态和源码 fingerprint
写入 manifest；不要把未提交代码的运行结果写成 clean checkout 证据。

## 研究计划

| 阶段 | 问题 | 产物 |
| --- | --- | --- |
| 0 | 多轮 Agent 能否正确使用搜索证据？ | 确定性的离线基线和轨迹记录 |
| 1 | SFT/RL 是否能够改善搜索策略？ | Search-R1 风格的训练适配层和固定评测集 |
| 2 | 失败感知的掩码能否减少有害更新？ | Fatal-aware 消融实验 |
| 3 | 哪些搜索轮次真正影响最终答案？ | 基于贡献度的 GRPO 消融实验 |

阶段 0 负责固定实验边界和记录方式。阶段 1 的 Level-A 适配层已经加入；真实模型和训练后端接入前，不把代码能力写成实验收益。所有比较仍需使用同一数据划分、搜索后端和预算条件。

## 复现约定

实验报告需要记录模型、提示词或模板版本、搜索后端、数据划分、随机种子、工具预算、token 预算、奖励公式以及 API 或模型成本。README 和简历中的实验结论，应当能够回到一份已提交的实验报告。

## 致谢与引用

本项目的研究方向参考了 Search-R1、DeepResearcher、OpenSearch-VL 和 CW-GRPO。后续如复用代码、数据集、模型或实验结果，会在对应报告中标明来源。

## 当前状态

当前提交是初始研究脚手架。默认工具使用离线 fixture，因此现有测试可以在没有凭证和网络的环境中运行。后续扩展应继续保留可复现的轨迹记录和明确的实验边界。
