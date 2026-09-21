# TraceSearch-R1 Agent Notes

这份文档记录进入 M1-C1.1 期间已经实际遇到并解决的结构性问题、验证方式和排查结论。它是仓库协作约定的一部分，不是新的训练方案。

## 当前边界

- 当前工作范围是 M0/M1-C1.1 的 rollout、后端证据和单次 vanilla-GRPO proof。
- 不要在没有新的 TRD 或明确授权的情况下开始 C2、训练 sweep、reward 改造、token mask 改造或自动重试实验。
- 当前仓库直接在 `main` 上工作。实验结果必须记录 commit、dirty 状态、运行时配置、模型路径和完整命令。
- C1.1 的安全停止条件优先于“把程序跑完”：如果 group 没有 reward variance，必须报告 `no_learning_signal`，跳过 actor update 和权重同步，不得靠换 seed、换 task 或扩大采样偷偷制造学习信号。

## 已解决的调试经验

### 1. rollout 身份不能只用 task_id

同一个 task 可以有多个并发 rollout。任何按 `task_id` 建字典的实现都会覆盖样本，进而破坏缺槽、排序和分母语义。必须保留 `rollout_id`、`sample_index`，并用固定的 expected group width 构造槽位。

缺失样本要占据自己的槽位，不能把后面的样本压到前面；同时应拒绝重复或越界的 `sample_index`。这条规则同时适用于 evaluator、live observer 和最终 metrics。

### 2. rLLM 的 step identity 以 backend 返回值为准

live rollout 的 `step_ids` 是后端对 response rows 的规范身份。不能把 `uid`、本地 trajectory id 或“第几个返回行”当成等价物。padding row 要由 `is_pad_step` 和 attention mask 明确排除，不能仅按长度或位置猜测。

### 3. live mask 行可能在 rollout 内重排

实际后端返回的 token rows 不保证和 observer 看到的 occurrence 顺序一致。对齐时应使用 exact unpadded token ids 加 attention mask 进行匹配，并验证每一行只匹配一次；不能假设“同一个 rollout 的第 N 行就是 observer 的第 N 行”。

### 4. parity 的数值契约必须完全一致

live advantage parity 使用后端同一套归一化和 epsilon。曾经出现过 `1e-6` 与 `1e-8` 的微小差异，结果是数学上等价但机器比较失败。比较前要记录 epsilon、输入 reward、mask、advantage，以及 max absolute error；不要只看最终布尔值。

### 5. 结构化 termination 不能被统一吞成 INTERNAL

`TerminationEvent.reason` 必须保留。长度上限、超时等有界的 backend termination 是一个已经完成的 reward-zero 样本；真正的 transport/backend exception 才应该作为 proof failure。把前者统一包装成 generic backend error，会错误触发 fail-fast 并丢失有效的 group 分母。

### 6. 输出目录和状态读取顺序很重要

不能因为旧输出目录已经存在就直接复用派生 artifact。运行开始时要拒绝 stale output，或明确创建新的 run directory。读取顺序应是：runtime observer/rollout evidence → 结构化状态 → derived metrics/status；否则会出现日志里已经有终止原因，但最终 `run_status.json` 仍是旧状态的漂移。

### 7. 参数更新必须有真实证据

“调用过 update”不是参数更新证据。优先使用 pre/post checkpoint 的参数 delta；worker monkey patch 或调用计数只能作为诊断信号。还要记录 optimizer step、actor update call、reload 结果和 delta 的非零参数数量。

当 reward variance 为零时，GRPO group normalization 得到全零 advantage；这时跳过 optimizer step 和 batch-end weight sync 是正确行为，不应把它伪装成训练成功。

### 8. logprob 诊断要保存分布，不只保存均值

为了判断 policy 是否真的产生了学习信号，需要保留 logprob 的 count、min/max、mean、std 和有限性检查。单个平均值无法区分全零、相同常数、有效分布或 NaN/Inf。

### 9. 真实运行环境本身是实验变量

clean WSL ext4 checkout、固定 uv 环境、明确的 vLLM/SGLang backend、`git_dirty`、源码 fingerprint、模型路径和 runtime profile 都应进入证据链。Windows 挂载盘上的 repo、WSL ext4 repo 和不同虚拟环境不能混称为同一次运行。

## 最终 C1.1 proof 的解释

`runs/m1-c1.1-proof-r5` 的关键事实是：4 个 rollout 都进入了分母，reward 全为 0，`group_exact_match_variance=0`，live mask parity 和 live advantage parity 均通过，但 actor update 次数为 0、optimizer step 为 0、checkpoint 前后没有 delta，最终状态是 `no_learning_signal`。

这不是“更新器偷偷失败后没有日志”，而是 C1.1 gate 按设计停止了更新。当前没有足够证据把问题归因到 FSDP、LoRA、optimizer 或权重同步。

## 对“是不是 max_tokens 太少”的判断

不是主要原因，至少不是这次最终 proof 的直接原因。

最终 proof 的运行参数是每 turn `--max-tokens 96`、`--max-turns 4`。代码把 rLLM 的 `max_response_length` 设为 `max(96 * 8, 512)=768`，而 `rllm.data.max_prompt_length` 和 vLLM `prompt_length` 是 512，vLLM `max_model_len` 是 1024。

最终 proof 的 4 条 rollout 中有 3 条明确以 `max_prompt_length_exceeded` 结束，只有 1 条正常回答；没有 `max_response_length_exceeded`。正常结束的回答仍然答错了（把目标答案答成了 Mount Ilex，而 gold 是 Sora Vale）。因此证据更支持：

1. 搜索结果和历史上下文让 prompt budget 先耗尽；
2. 全部 reward 为零后，advantage 必然全零，update 被安全 gate 跳过；
3. `max_tokens=96` 可能限制单 turn 的思考/最终回答，但把它盲目调大可能让历史更快膨胀，不能作为这次 proof 的首要解释。

下一次受控实验若获授权，应先分别记录每一 turn 的 prompt token 数、observation token 数和终止原因，再决定是增加 prompt budget、减少 observation、减少搜索结果，还是调整 turn/token 上限。不要只改 `max_tokens` 后把结果归因给它。

## 另一个重要的可复现性风险

当前 policy 的采样 seed 会由 `base_seed + task_id + rollout_id + sample_index + step_index` 经过 BLAKE2 派生。这个设计能隔离并发 rollout，但如果 `rollout_id` 含有每次运行新生成的 rLLM UUID，那么同一条命令、同一个 `--seed` 在不同 run 之间并不一定得到同一组实际采样流。再叠加 Ray/vLLM 并发调度和 GPU kernel 的非完全确定性，run-to-run 的 reward 组合可能变化。

所以 r4 出现 mixed reward、r5 又出现全零 reward 时，不能简单判断为“模型或更新器随机坏了”。应把稳定的 run/group identity 纳入复现设计，或在实验记录中明确区分 nominal seed 与实际派生 seed。这个方向属于后续受控修复，不应在 C1.1 proof 期间通过自动换 seed 绕过零信号 gate。

## 每次 proof 前后的最小检查表

- `git rev-parse HEAD`、`git status --short`、源码 fingerprint、模型路径和环境 Python 可执行文件。
- exact CLI、task id、group size、sample index、rollout id、base seed 和每一步派生 seed。
- 每个 slot 是否完成、是否 missing/padded、termination reason、prompt/response token 数。
- reward 列表、group variance、mask parity、advantage parity 及误差上界。
- actor update call、optimizer step、pre/post checkpoint delta、reload evidence。
- 若为 `no_learning_signal`，保留原始证据并停止，不用重采样替换失败或零 reward 样本。

