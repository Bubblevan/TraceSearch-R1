# TraceSearch-R1：Search Agent / Agentic RL 通用八股

本文与 TraceSearch-R1 的实现状态解耦，目标是准备大模型算法、Agent 算法、LLM 后训练和相关 Agent 开发岗位的通用追问。遇到项目绑定问题，回到 [`project.md`](project.md)；不能把本页出现的论文、算法或系统组件自动写成 TraceSearch-R1 已实现能力。

## 先用这套答题模板

每道题按以下顺序回答：

1. **短答**：20～30 秒给定义和核心判断。
2. **原理 / 公式**：说明信号在哪里产生、进入哪个变量或 objective。
3. **trade-off**：讨论 bias、variance、成本、稳定性、可复现性和 reward hacking。
4. **项目映射**：指出 TraceSearch-R1 当前源码能证明什么，哪些仍是 `[PLANNED]`。
5. **常见追问**：给出面试官最可能继续挑战的点。

## A. Search Agent 与运行时

### 面试官：Agent 和 Workflow 有什么区别？

**短答**：Workflow 的主要控制流由代码预先决定；Agent 在 state、tool 和 policy 约束下，让模型或策略选择下一步 action。两者都可能调用 LLM，“用了 LLM”不等于“是 Agent”。

**原理**：固定 `retrieve → generate` 是 workflow；Search Agent 通常执行 `state → policy → search/visit/answer → observation → state`，直到 answer、预算耗尽或失败终止。

**trade-off**：Agent 能适应未知路径，但会引入成本、非确定性、循环、工具错误和更复杂的 credit assignment；Workflow 更易测试，却只能覆盖预编码路径。

**项目映射**：M0 有显式 action loop 和 runtime budget，但默认是 `OracleFixturePolicy`，没有 learned policy。

**常见追问**：模型选择到什么程度才算 Agent？最终终止权属于 policy 还是 runtime？

### 面试官：Search Agent 和普通 RAG 有什么区别？

**短答**：RAG 通常描述“检索外部知识后生成”；Search Agent 强调多轮决策，策略会根据前一轮 observation 决定下一条 query、是否 visit、是否继续或回答。

**原理**：一次性 RAG 的 retrieval graph 大多固定；Search Agent 的 action sequence 是 trajectory，搜索深度、query reformulation、证据选择和 stop decision 都是 policy 的一部分。

**trade-off**：多轮搜索提高探索能力，但也放大延迟、错误传播、上下文污染和训练难度。先验证任务确实需要多步搜索，避免用复杂 Agent 解决 single-hop shortcut。

**项目映射**：TraceSearch-R1 当前 BM25 只是离线 environment；项目尚未证明多轮 learned policy 优于 retrieve-once baseline。

### 面试官：Search、Visit 和 Answer 为什么要拆成不同 Action？

**短答**：Search 决定候选集合，Visit 决定展开哪条证据，Answer 决定停止并输出。三个动作的成本、失败类型和贡献不同，合并后无法分析 policy 行为或分配 credit。

**原理**：Search observation 通常包含 rank/score/snippet；Visit observation 包含完整页面或文档；Answer 不产生 tool result，而是 terminal action。

**trade-off**：动作拆得越细越可观测，但 horizon 更长、action space 更大、训练更难；动作过粗则丢失 attribution。

**项目映射**：`ActionKind` 当前只有 `search/visit/answer`，是 M0 的最小 action vocabulary。

### 面试官：State、Observation、Step、Trajectory 怎么区分？

**短答**：State 是 policy 决策时可用的执行状态；Observation 是 environment 对 action 的返回；Step 把一次 decision、action 和 observation 绑定；Trajectory 是同一 task 从开始到终止的完整 step sequence。

**原理**：训练时 token/step/trajectory reward 的作用域不同，必须先有稳定层级。Observation 应来自 environment；policy reasoning 不应被当作环境事实。

**trade-off**：记录越完整越利于 replay 和诊断，但存储、隐私和序列长度成本更高；自由文本日志便宜，却难以重算指标。

**项目映射**：M0 用 typed dataclass 和 JSONL 固定这几层；当前没有 token-level rollout log。

### 面试官：为什么需要异步工具接口？

**短答**：Search/Visit 是 I/O-bound 操作，训练或评测时会有大量并发 trajectory；同步等待会让 rollout throughput 被最慢工具调用拖住。

**原理**：asyncio 让单进程在等待 I/O 时切换其他任务；大规模系统还需要 semaphore、queue、timeout、cancellation、rate limit 和 backpressure。

**trade-off**：异步提高利用率，但增加并发竞态、取消传播、顺序复现和错误收口难度。并发度不是越大越好，2048 个 tool call 可能先压垮连接池或后端限流。

**项目映射**：M0 的单 trajectory loop 支持 async policy/tool，但 experiment runner 当前逐 task 执行，不代表已经实现并行 rollout pool。

## B. Environment、数据与评测

### 面试官：什么是 environment alignment？

**短答**：任务、可用工具和可访问证据必须匹配；若 gold evidence 不在 corpus 或工具协议无法完成任务，policy 无论多强都得不到正确 reward。

**原理**：alignment 至少检查 solvability、gold existence、ID/version、访问权限、答案唯一性和 verifier 可用性。否则 environment error 会被错误归因到 policy。

**trade-off**：高度 aligned 的 synthetic environment 易复现但可能过于简单；live environment 真实却会漂移。需要分别报告 mechanism validation 与 real-world robustness。

**项目映射**：M0 fixture 明确对齐 task gold ID 与 local corpus；这不是公开 benchmark。

### 面试官：live Web 和 offline simulator 怎么选？

**短答**：offline simulator 适合 deterministic regression 和 clean ablation；live Web 适合测真实分布、动态内容和网络故障。研究路线通常先 offline，再做版本化的 sim-to-real。

**原理**：offline 固定 corpus hash、ranking 与 failure schedule；live Web 需要记录时间、provider、query、cache、页面快照和访问状态，否则不可比较。

**trade-off**：offline 有 simulator bias；live Web 有 drift、成本、隐私、合规和不可复现问题。

**项目映射**：当前只有 local BM25，live adapter 是 `[PLANNED]`。

### 面试官：execution failure、semantic failure 和 policy failure 有什么区别？

**短答**：Execution failure 是工具没正常执行；semantic failure 是工具成功但内容无关或误导；policy failure 是策略选择了不合适 action、参数或停止时机。

**原理**：三类 failure 的责任主体和恢复动作不同：execution 可 retry/backoff，semantic 需要 relevance/verification，policy 需要数据或优化信号。

**trade-off**：过度 retry 会浪费预算；把 semantic failure 当 execution error 会掩盖检索质量；把 environment failure 全算给 policy 会污染 reward。

**项目映射**：M0 区分 `ToolErrorType`、semantic corruption metadata 和 `policy_error` termination。

### 面试官：为什么需要 fault injection？

**短答**：因为真实故障不可控，无法保证两个算法在同一 failure surface 上比较。Fault injection 把故障类型、位置、概率和 seed 变成实验变量。

**原理**：定点 schedule 用于机制测试；seeded stochastic injection 用于分布性试验。必须固定随机数 draw order，否则增加一个 fault type 就会改变所有旧实验。

**trade-off**：注入模型可能与真实故障分布不符；需要用真实 trace 校准，而不是把 synthetic recovery 率当生产可靠性。

**项目映射**：M0 已有 schedule 和 seeded injector，但没有真实 Web failure distribution。

### 面试官：如何评估 Search Agent，而不是只看最终答案？

**短答**：同时评 outcome、evidence、trajectory、cost 和 safety：答案正确性、证据召回/引用、无效搜索、工具失败、搜索深度、终止、延迟、token 与越权。

**原理**：最终 answer 可能正确但 trajectory 低效或碰巧；也可能答案失败但已找到正确证据。应拆开 policy quality、environment quality、evidence acquisition 与 final generation。

**trade-off**：指标越多越容易诊断，但也增加选择性汇报风险；必须预先定义 primary metric、denominator 和 missing run 处理。

**项目映射**：M0 当前有 exact match、search/visit recall、tool failure、termination 和 latency；没有 citation correctness、judge score 或真实成本。

### 面试官：数据泄漏有哪些形式？

**短答**：包括 answer leakage、gold evidence leakage、train/test overlap、检索库包含答案模板、prompt 泄漏、benchmark contamination 和可由单跳 shortcut 解出的伪多跳题。

**原理**：分别审计 task、corpus、model context、policy API 和 evaluator。Gold metadata 必须与 model-facing view 隔离。

**trade-off**：完全排除公开预训练污染很难，但可以保留 provenance、时间切分、去重、hidden holdout 和 adversarial shortcut test。

**项目映射**：普通 policy 使用 `Task.policy_view()`；Oracle policy 的 gold access 只允许用于 smoke validation。

## C. Agentic RL 与优化目标

### 面试官：SFT、RLHF 和 RLVR 有什么区别？

**短答**：SFT 拟合示范 token；RLHF 用人类偏好或 reward model 优化策略；RLVR 用可验证结果产生 reward。Search Agent 常混合使用：先 SFT 获得基本工具格式，再用 outcome/evidence verifier 做 RL。

**原理**：SFT 优化负对数似然；RL 优化策略期望回报。RLVR 的关键不是“没有人”，而是 reward 能由规则、答案或可验证环境稳定计算。

**trade-off**：SFT 稳定但受示范上限限制；RL 能探索却有高方差和 reward hacking；RLVR verifier 若有漏洞，模型会优化漏洞。

**项目映射**：TraceSearch-R1 当前没有 SFT 或 RL trainer。

### 面试官：REINFORCE 的基本公式是什么？

**短答**：REINFORCE 用采样回报加权 log-prob gradient：高回报动作概率上升，低回报动作概率下降。

**公式**：

\[
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau\sim\pi_\theta}
\left[\sum_t (G_t-b_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]
\]

其中 \(\tau\) 是 trajectory，\(G_t\) 是从 step \(t\) 开始的 return，\(b_t\) 是 baseline，\(A_t=G_t-b_t\) 是 advantage。

**trade-off**：无偏估计通常方差高；baseline、更多样本和 reward shaping 可降方差，但 shaping 不当会引入偏差或 hacking。

**项目映射**：M0 只记录 trajectory；还没有 policy log-prob、return 或 optimizer。

### 面试官：PPO 的 ratio、clipping 和 KL 分别做什么？

**短答**：PPO 用新旧策略概率比衡量更新幅度，用 clipping 限制单批数据上的过大策略变化，并常用 reference KL 约束语言模型偏离基线。

**公式**：

\[
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}
\]

\[
L_{\text{clip}}
=\mathbb{E}\left[\min\left(r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]
\]

**trade-off**：clip 太紧学习慢，太松可能不稳定；KL 太强限制探索，太弱可能语言退化或 reward hacking。token-level ratio 还会在长序列中积累方差。

**项目映射**：当前没有 old policy、reference policy 或 PPO loss。

### 面试官：GRPO 为什么不需要显式 critic？

**短答**：GRPO 对同一 prompt 采样一组 response，用组内 reward 的相对位置估计 advantage，以 group baseline 替代 learned value critic。

**公式**：常见简化写法是

\[
A_i=\frac{r_i-\mu_G}{\sigma_G+\varepsilon}
\]

其中 \(r_i\) 是组内第 \(i\) 个 rollout reward，\(\mu_G,\sigma_G\) 是组均值与标准差。随后用 importance ratio、clip 和可选 KL 优化 token log-prob。

**trade-off**：省去 critic，但需要同 prompt 多个 rollout；组内 reward 全相同会缺少学习信号，组规模、reward variance 和采样多样性非常关键。

**项目映射**：TraceSearch-R1 尚未实现 group rollout、GRPO loss 或训练框架接入。

### 面试官：on-policy 和 off-policy 有什么区别？

**短答**：On-policy 用当前或非常接近当前的 policy 生成数据；off-policy 复用旧 policy 或其他来源数据，需要处理分布偏移。

**原理**：importance sampling ratio 用于校正行为策略与目标策略差异，但长序列、旧数据和极端 ratio 会增加方差。

**trade-off**：on-policy 新鲜但昂贵；off-policy 样本效率高但可能训练不稳定。Search Agent 的 environment 成本会让数据复用很诱人，也更需要版本和 replay 边界。

**项目映射**：当前 artifact 没有 policy version/log-prob，不能用于真实 off-policy correction。

## D. Credit Assignment 与过程信号

### 面试官：Outcome reward 和 process reward 有什么区别？

**短答**：Outcome reward 评价最终答案或任务是否完成；process reward 评价中间步骤是否合理、有用或合规。前者便宜且目标直接，后者信号更密但需要标注或 judge。

**原理**：Search trajectory 中 outcome 可能只在末尾出现；process signal 可以作用到 query、visit、evidence selection 或 reasoning step。

**trade-off**：Outcome reward credit 稀疏；process reward 容易把 judge 偏好当真实目标，并增加推理成本和新的 hacking surface。

**项目映射**：M0 有最终答案与 trajectory metrics，没有 process reward model。

### 面试官：直接加入 process reward，与用 process quality 加权 outcome advantage 有什么区别？

**短答**：直接加入 process reward 会改变优化目标本身；用 process quality 乘 outcome advantage 则保持 outcome 的正负方向，只重新分配同一 trajectory 内各步骤的更新强度。

**公式**：

直接加 reward：

\[
r'_t=r_{\text{outcome}}+\lambda q_t
\]

加权 advantage：

\[
\tilde A_t=A_{\text{outcome}}\cdot
\frac{q_t}{\frac{1}{T}\sum_j q_j+\varepsilon}
\]

其中 \(q_t\) 是过程质量，\(T\) 是 step 数。

**trade-off**：直接相加可能让高 process score 抵消错误 outcome，优化“看起来过程好”；加权方式保留 outcome 方向，但若 \(q_t\ge0\)，可能无法显式惩罚成功 trajectory 中的坏步骤，也会受 judge calibration 影响。

**项目映射**：当前 `weighted_advantages` 接近第二类 step weighting helper，但没有真实 outcome advantage、judge score 或训练 integration。

**常见追问**：为什么 successful answer 不意味着每个 search 都有用？failed answer 中的正确 prefix 应该怎样更新？

### 面试官：trajectory、turn、step 和 token-level credit 怎么区分？

**短答**：Trajectory credit 给整条 rollout 一个信号；turn/step credit 区分 action 与 observation 阶段；token credit 再把信号分到具体生成 token。粒度越细，潜在 attribution 越准，标注与方差问题也越复杂。

**原理**：同一 search step 的 query token、tool call schema token 和 reasoning token 未必应获得相同 advantage。把 step weight广播到所有 token只是近似。

**trade-off**：粗粒度稳定但误归因；细粒度依赖更强 judge/causal estimate，计算成本和 hacking surface 更大。

**项目映射**：M0 schema 到 Step 粒度，没有 token log-prob 或 mask。

### 面试官：Fatal-aware masking 想解决什么，可能引入什么偏差？

**短答**：它尝试在不可恢复 failure 后停止对后缀 token 更新，避免坏环境状态产生大量噪声；但 fatal boundary 若判断错，会删除本可恢复的数据并引入选择偏差。

**原理**：一种做法是得到 fatal index \(t_f\)，令 mask \(m_t=1[t<t_f]\)，只优化 prefix；另一种是对后缀做不同权重而非硬 mask。

**trade-off**：hard mask 简单但敏感；soft weight 更平滑却需要校准。模型还可能学会提前触发被 mask 的 failure 来逃避负奖励。

**项目映射**：`fatal_step_index` 只是连续失败启发式，尚未接 loss。

### 面试官：为什么负 trajectory 不能简单把所有步骤都设成负 advantage？

**短答**：因为失败可能发生在后缀生成、环境 timeout 或单个错误 visit；早期 query 和 evidence acquisition 仍可能正确。全轨迹负更新会压低有价值 prefix。

**原理**：需要区分 environment failure、policy failure、recoverable failure 与 fatal failure，并比较 outcome-only、masking、step weighting、counterfactual 或 retrospective critic。

**trade-off**：保留 prefix 可减少误罚，但也可能让真正有害的早期动作逃过惩罚。任何方法都需要因果假设和消融。

**项目映射**：TraceSearch-R1 的研究方向正是比较这些分配方式，但当前没有结论。

## E. Rollout、Serving 与分布式系统

### 面试官：vLLM、PagedAttention 和 continuous batching 分别解决什么？

**短答**：vLLM 是高吞吐 LLM serving/rollout engine；PagedAttention 用分页式 KV cache 管理减少碎片；continuous batching 动态把新请求加入正在执行的 batch，提高 GPU 利用率。

**原理**：Agent rollout 的序列长度和工具等待差异大，静态 batch 容易被长尾拖慢；连续调度能提高吞吐，但工具 I/O 与生成阶段仍需协调。

**trade-off**：更高吞吐会增加调度复杂度、KV 内存压力、取消/恢复问题和结果顺序不确定性。

**项目映射**：当前没有 vLLM 或 GPU rollout。

### 面试官：Agent rollout 中为什么需要 backpressure？

**短答**：如果 policy 生成 action 的速度超过搜索后端、judge 或 trainer 消费速度，无界队列会耗尽连接、内存或配额。Backpressure 让上游根据下游容量减速。

**原理**：常用 bounded queue、semaphore、rate limit、timeout、circuit breaker 和 per-provider pool；还要区分 retryable 与 permanent failure。

**trade-off**：限制过紧浪费 GPU，过松压垮工具后端。优化目标应看端到端 samples/sec，而不是单独把模型并发拉满。

**项目映射**：M0 只有顺序 experiment runner，没有生产级 workflow pool。

### 面试官：RL rollout/training colocated 和 disaggregated 怎么选？

**短答**：Colocated 让 rollout 与 training 共享 GPU，切换时 sleep/wake 或卸载权重；disaggregated 用不同资源池，减少阶段切换但增加权重同步和通信成本。

**原理**：长 horizon search rollout 受工具 I/O 影响，GPU 可能空闲；disaggregation 可提高隔离性，colocation 可减少硬件总量。

**trade-off**：Colocated 容易 OOM 和切换抖动；disaggregated 容易 policy staleness、网络瓶颈与运维复杂。

**项目映射**：当前没有 Ray、verl、rLLM 或 hybrid engine。

### 面试官：长轨迹 RL 为什么容易 OOM？

**短答**：prompt/response 长度、并发 trajectory、group rollout 数、KV cache、activation、optimizer state 和 reference/old policy 都会叠加显存压力。

**原理**：rollout 侧主要是 KV cache 与并发序列；training 侧主要是 activation、gradient、optimizer 和多模型副本。工具 observation 还会让 context 快速膨胀。

**trade-off**：缩短上下文、降低并发或 group size 能省显存，却可能损害搜索深度、样本多样性和吞吐。

**项目映射**：当前没有 GPU 实验或 OOM 数据，不能编写显存数字。

## F. 数据难度、论文路线与多模态边界

### 面试官：为什么数据难度对 Agentic RL 特别重要？

**短答**：题太简单会被参数记忆或单跳 shortcut 解决，搜索 action 没有价值；题太难或环境不可解则 reward 几乎全零。RL 需要位于当前 policy 可探索、可验证但不 trivial 的难度带。

**原理**：可用 search depth、evidence hops、baseline success、tool necessity 和 verifier confidence 估计难度，并做 curriculum。

**trade-off**：按当前 policy 选题会引入 selection bias；固定 curriculum 又可能很快过时。必须保留 held-out 难度切片。

**项目映射**：M0 fixture 含 single-hop、two-hop 和 no-search calibration，但规模太小，不能得出 curriculum 结论。

### 面试官：multimodal model、multimodal agent、multimodal search agent 有什么区别？

**短答**：Multimodal model 能编码/生成多种模态；multimodal agent 能根据状态选择图像、OCR、裁剪等工具；multimodal search agent 还要跨文本/图像搜索、访问和验证证据。

**原理**：能力层级依次增加 action space、environment、observation schema 和 credit assignment 难度。会看图不等于会主动决定何时 ImageSearch、Crop 或 OCR。

**trade-off**：视觉工具提高可解任务范围，也增加 tool latency、图像 token、坐标/分辨率错误和 judge 成本。

**项目映射**：TraceSearch-R1 当前是 text-only，M8 式 multimodal 方向属于 `[PLANNED]`。

### 面试官：建立论文谱系时最重要的纪律是什么？

**短答**：分别写清每篇工作的 problem、data、environment、reward、optimizer、贡献和限制，再说明 TraceSearch 借鉴的是接口、假设还是实验设计；不能把论文结果改写成自己的结果。

**原理**：将论文按 search policy、credit assignment、reward sparsity、environment、data/self-evolution、retriever、memory 和 multimodal 分类，比简单按发布日期罗列更利于面试比较。

**trade-off**：谱系过大容易喧宾夺主。只有与当前 milestone 或明确 planned ablation 相关的论文才进入正文，其余保留为简短 reference。

**项目映射**：项目当前只声明参考 Search-R1、DeepResearcher、OpenSearch-VL、CW-GRPO；后续文稿必须核对原论文和官方 repo。

## 与 TraceSearch-R1 的项目映射

- `[WORKTREE IMPLEMENTED]`：typed M0 schema、async loop、Search/Visit environment、local BM25、budget/termination、fault injection、artifact writer、offline evaluator。
- `[ARTIFACT VALIDATED]`：synthetic fixture 的 smoke/fault run 能生成并重算 artifacts；这不是 model benchmark。
- `[IMPLEMENTED HELPER, NOT TRAINING]`：`fatal_step_index`、`weighted_advantages`。
- `[PLANNED]`：learned LLM policy、SFT、PPO、GRPO、GSPO、CW-GRPO、process judge、live Web、vLLM/Ray/verl/rLLM、多模态和 memory。

## 后续扩写顺序

1. M0 clean checkpoint 与真实 non-oracle baseline 落地后，补项目证据和 failure table。
2. M1 训练接入后，沿实际代码追踪 `reward → group advantage → token mask → loss → optimizer`。
3. 每个新算法只在有实现或明确 ablation plan 时扩写，不建立空壳章节。
4. 引用论文时使用原论文/官方 repo，并明确“论文结果”与“本项目结果”。
