# MCAgent 工程迭代指南（Codex 执行版，V4）

## 0. 文档目的

本文件用于指导 Codex 在当前 `mcagent` 仓库上继续迭代，实现与我们最终方法论一致的 **MCAgent 五动作空间版本**。

本文档的目标不是写论文，而是：

1. 让 Codex 明确当前仓库已经完成了什么。
2. 让 Codex 明确下一阶段哪些模块必须改，哪些模块先不要动。
3. 让 Codex 按步骤完成工程升级，使仓库从 **phase-one prototype** 迭代为 **可支撑正式实验的 action-calibration 系统**。

---

## 1. 当前仓库状态总结（承接上一版文档）

当前仓库已经具备一个最小闭环：

- 统一数据加载器
- 统一 prompt 构造与 JSON 决策解析
- sandbox + tool 接口
- rollout 日志生成
- 过程特征与语义 tag 的初版抽取
- oracle / 局部效用初版
- pair 构造
- TRL DPO 训练入口
- 基础 action/answer 评测入口

这意味着：

> 仓库已经完成了“基础原型”的主要内容，但还没有完成“正式研究版 MCAgent”。

Codex 接下来的工作不是重写一切，而是：

> **在现有仓库基础上按模块升级。**

---

## 2. 最新研究目标（本版唯一主线）

### 2.1 核心问题

本项目的目标不是单纯提升 verbal confidence，也不是构建一个炫技型复杂 agent，而是研究：

# **Action Calibration under Ability / Knowledge Boundaries**

即：模型在面对不同类型的能力边界与知识边界时，是否能够把不确定性正确映射成恰当动作。

### 2.2 最新动作空间：5 个主动作

从本版开始，动作空间统一定义为五类主动作：

- `ANSWER`
- `SEARCH`
- `CALCULATE`
- `CLARIFY`
- `REFUSE`

注意：

- `CALCULATE` 不再视为 `SEARCH` 或一般 `TOOL` 的次级子类，而是**一级主动作**。
- 这是因为当前仓库的数据中有两个数学数据源：`gsm8k` 与 `competition_math`。
- 因此方法论必须同时覆盖：
  - 数学边界：`ANSWER vs CALCULATE`
  - 知识边界：`ANSWER vs SEARCH vs REFUSE`
  - 信息缺失边界：`CLARIFY`

### 2.3 当前版本对 Meta-Reflection 的处理

保留 “轻量 meta-reflection” 思想，但不把它作为重型自由文本主目标。

具体实现上：

- `reason`：保留问题求解或判断过程
- `decision.action`：五选一主动作
- `decision.brief_rationale`：作为轻量 meta-reflection，仅用于：
  - 调试
  - 错误分析
  - 教师标注承载
  - 可解释性辅助

**`brief_rationale` 不是当前阶段的主要训练目标。**

---

## 3. 数据集角色与任务重构

### 3.1 当前本地数据目录

仓库中已存在：

- `dataset/gsm8k`
- `dataset/competition_math`
- `dataset/freshqa`
- `dataset/IN3`
- `dataset/MintQA-Ti-v0.1`

### 3.2 数据集角色重构

#### A. 数学边界（Math Boundary）

数据：
- `gsm8k`
- `competition_math`

主要动作：
- `ANSWER`
- `CALCULATE`

次要动作：
- `REFUSE`（仅在后续构造无解/自相矛盾数学题时可用）

工程要求：
- 数学域中 `SEARCH` 默认通常应被视为不必要动作。
- pair 构造时要重点关注：
  - `CALCULATE_helpful` vs `ANSWER_wrong`
  - `ANSWER_correct` vs `CALCULATE_unnecessary`

#### B. 知识边界 / factual boundary

数据：
- `freshqa`
- `MintQA-Ti-v0.1`

主要动作：
- `ANSWER`
- `SEARCH`
- `REFUSE`

工程要求：
- `freshqa` 更偏 OOD / 外测
- `MintQA-Ti-v0.1` 更适合作为训练主料之一
- pair 构造时要重点关注：
  - `SEARCH_helpful` vs `ANSWER_wrong`
  - `REFUSE_justified` vs `ANSWER_hallucinated`
  - `ANSWER_correct` vs `SEARCH_unnecessary`

#### C. 信息缺失边界（Intention / Missing-Info Boundary）

数据：
- `IN3`

主要动作：
- `CLARIFY`

次要动作：
- `ANSWER`
- `REFUSE`

工程要求：
- 需要 clarify oracle 环境
- 首版只支持“一次追问后回答”的简化闭环

---

## 4. 决策协议：保留 JSON 主协议，不强制迁移到 function-calling

### 4.1 当前决定

本版不强制把当前系统整体改造成 OpenAI function-calling / tools runtime。

原因：

- 当前主目标是 action calibration，而不是接口协议研究。
- 模型当前输出 JSON decision 已较为稳定。
- 整体迁移到 function-calling 会引入大量额外工程重构，但对当前研究主问题帮助有限。

### 4.2 当前主协议

仍然采用：

```json
{
  "reason": "...",
  "decision": {
    "action": "ANSWER|SEARCH|CALCULATE|CLARIFY|REFUSE",
    "action_input": {},
    "brief_rationale": "..."
  }
}
```

### 4.3 必须新增的稳定性措施

Codex 必须补：

- JSON schema validator （直接使用json_repair库即可）
- 解析失败时的 retry / fallback parser
- 对 `action` 枚举值做严格校验
- 对 `action_input` 做最小合法性检查

### 4.4 可选兼容层

可以新增一个可选 adapter：

- `src/agent/tool_call_adapter.py`

但这不是当前主路径，只作为未来扩展。

---

## 5. Search 工具实现思路（本版重点）

## 5.1 核心原则

当前阶段：

> **训练时不需要真实调用所有工具。**

训练中真正关心的是：

- 当前状态下，模型是否选择了正确动作
- 局部效用是否正确
- chosen/rejected 是否合理

因此：

- **训练阶段：可以不执行完整真实工具链**
- **最终评测 / 推理阶段：才执行完整工具流程**

这是本版设计的明确原则。

## 5.2 Search 工具分为两层

### A. 训练阶段用：代理检索 / mock retrieval

训练阶段 `SEARCH` 不必真的联网。

推荐做法：

- 使用本地检索器或 benchmark-aware retrieval
- 或使用缓存好的 evidence
- 或使用 teacher / annotation 阶段保存的 retrieval candidates

训练阶段只需要判断：

- 在该状态下 `SEARCH` 是否应被视为 **helpful / necessary**（这时可以使用数据集中标注的缺失信息/真实信息进行search内容比对）
- 而不是要求每次真的调用互联网并完成端到端回答

### B. 最终评测阶段用：真实检索后端

评测时再接入真正后端，例如：

- local retriever
- Serper
- Brave
- Tavily
- Exa

但必须实现为 **可插拔后端**，不能绑死在单一供应商上。

## 5.3 Search 模块重构要求

Codex 需要把当前 `search_tool.py` 重构为：

```text
src/tools/search/
  base.py
  dispatcher.py
  local_retriever.py
  mock_retriever.py
  serper_backend.py
  brave_backend.py
  tavily_backend.py
  exa_backend.py
```

### 5.3.1 后端分工

- `mock_retriever.py`
  - 用于训练 / smoke test
  - 只返回预构造候选证据
  - 不能直接泄漏 gold answer

- `local_retriever.py`
  - 用于主实验的可复现检索
  - 可从本地证据缓存或语料索引中查找

- 在线后端（`serper`, `brave`, `tavily`, `exa`）
  - 用于最终扩展评测
  - 不作为训练主依赖

### 5.3.2 配置示例

新增或重写：

```yaml
tools:
  search:
    backend: mock_retriever
    top_k: 3
    enable_online_backend: false
```

后续评测阶段可切换：

```yaml
tools:
  search:
    backend: serper
    top_k: 5
    enable_online_backend: true
```

---

## 6. 高能力教师模型指导标注（必须预留 API Key）

## 6.1 标注原则

### 过程特征 TAG

继续采用 rule-based：

- `STRUGGLE_LONG`
- `HAS_SELF_REPAIR`
- `LOW_LOGIT_MARGIN`
- `HIGH_BRANCHING`

这类特征必须由系统脚本提取，不要求模型自报。

### 语义 TAG 与标准动作

推荐采用：

> **rules 初筛 + 高能力教师模型 adjudication**

也就是：

1. 规则系统先给出候选语义标签
2. 教师模型做：
   - 语义 TAG 修正/补全
   - 标准动作建议
   - 简短 `brief_rationale`
3. 最终写入标注结果

## 6.2 推荐语义 TAG 集合

本版统一采用：

- `FRESH_FACT`
- `FALSE_PREMISE`
- `MISCONCEPTION_RISK`
- `NEW_OR_TAIL_KNOWLEDGE`
- `MISSING_INFO`
- `TOOL_REQUIRED`
- `JUSTIFIED_REFUSE`
- `CALCULATION_REQUIRED`

说明：
- `TIME_SENSITIVE` 不再作为中心标签
- 改为更一般的 factual / knowledge boundary 标签体系

## 6.3 标准动作标签

教师模型可输出：

- `ANSWER`
- `SEARCH`
- `CALCULATE`
- `CLARIFY`
- `REFUSE`

但必须同时保留：

- `rule_action`
- `teacher_action`
- `final_action`

教师模型不是绝对真理裁判，而是弱监督裁判 / adjudicator。

## 6.4 API Key 预留要求

Codex 必须新增环境变量配置位（实际上我计划使用poe提供的api），示例：

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL_FOR_LABELING=gpt-4o
```

建议写入：

- `.env.example`
- `configs/teacher.yaml`

### `configs/teacher.yaml` 建议内容

```yaml
teacher:
  enabled: false
  provider: openai
  model: gpt-4o
  api_key_env: OPENAI_API_KEY
  base_url_env: OPENAI_BASE_URL
  temperature: 0.0
  max_retries: 3
  batch_size: 8
```

## 6.5 教师标注模块

Codex 需要新增：

```text
src/teacher/
  client.py
  prompts.py
  label_semantic_tags.py
  label_standard_action.py
  label_brief_rationale.py
```

## 6.6 教师标注输出格式

统一保存为：

```json
{
  "id": "...",
  "rule_tags": [...],
  "teacher_tags": [...],
  "final_tags": [...],
  "rule_action": "...",
  "teacher_action": "...",
  "final_action": "...",
  "teacher_brief_rationale": "...",
  "teacher_note": "..."
}
```

---

## 7. shared-prefix Step-DPO：核心实现要求

## 7.1 训练对象重新澄清

当前正式训练对象应当是：

\[
U(a_t \mid s_t)
\]

其中：
- `s_t` = 决策前状态
- `a_t` = 当前候选动作

### 关键点

同一对 `chosen/rejected` 应尽量共享：

- 同一 `question`
- 同一 `reason prefix`
- 同一 `history / observations`

也就是说：

> **训练时固定前缀，只比较动作 continuation。**

## 7.2 不要求整条轨迹完全相同

shared-prefix 的真正含义是：

- 进入动作决策时的状态相同
- 而不是整条最终 rollout 完全一致

## 7.3 pair 数据格式（最新版）

Codex 需要把 DPO pair 主格式改成：

```json
{
  "id": "sample-id",
  "dataset": "mintqa",
  "task_type": "factual_boundary",
  "prompt": "<state prompt: question + reason prefix + observations + optional tags>",
  "chosen": "<decision block containing chosen action>",
  "rejected": "<decision block containing rejected action>",
  "chosen_action": "SEARCH",
  "rejected_action": "ANSWER",
  "utility_gap": 1.1,
  "state_tags": [...]
}
```

## 7.4 如果模型分叉不出来怎么办

按如下顺序处理：

1. 先让 completion 只覆盖动作部分，不重新生成长前缀
2. 先在高不确定样本上做 branching
3. 仅在动作位置提高探索度
4. 如果仍然分叉不足，再做小规模 action-unlock warm-up
5. 最后才考虑轻度受控探索构造 bootstrap 数据

---

## 8. 局部效用函数（升级为 5 动作）

Codex 需要把局部效用显式改成 5 动作版本。

### 8.1 初始离散效用表

- `ANSWER_correct = +1.0`
- `ANSWER_wrong = -1.0`
- `SEARCH_helpful = +0.6 - lambda_search`
- `SEARCH_unhelpful = -0.1 - lambda_search`
- `CALCULATE_helpful = +0.8 - lambda_calc`
- `CALCULATE_unhelpful = -0.1 - lambda_calc`
- `CLARIFY_helpful = +0.5 - lambda_clar`
- `CLARIFY_unhelpful = -0.1 - lambda_clar`
- `REFUSE_justified = +0.4`
- `REFUSE_unjustified = -0.6`

### 8.2 推荐默认参数

```yaml
scoring:
  lambda_search: 0.1
  lambda_calc: 0.05
  lambda_clar: 0.1
```

说明：
- 在数学域里，`CALCULATE` 应略优于 `SEARCH`
- 这符合数学工具与检索工具的任务差异

---

## 9. calibration pipeline（本版必须补齐）

## 9.1 当前问题

现有仓库中的 calibration 模块仍然是 placeholder。

这与项目动机不一致，因此必须补齐。

## 9.2 本版指标策略

### 主指标

**AUROC**

原因：
- 我们当前最关注模型对动作/答案偏好的排序能力
- 相比绝对数值校准，AUROC 更符合当前 action-calibration 研究目标

### 次级指标

- action-level AUROC
- risk-coverage / utility-coverage

### 辅助指标（附录/次要）

- ECE
- Brier

## 9.3 rollout 日志必须新增的字段

Codex 需要在 rollout 中补充：

- `verbal_confidence`
- `action_confidence`
- `score_answer`
- `score_search`
- `score_calculate`
- `score_clarify`
- `score_refuse`

如果当前模型无法稳定直接给出这些分数，第一阶段可先实现：

- verbal confidence（模型自报一个 0~1 值）
- action-level preference score（启发式或 teacher 辅助）

## 9.4 需要新增/重写的文件

```text
src/eval/evaluate_calibration.py
src/eval/plot_calibration.py
src/eval/calibration_utils.py
```

### 评测输出至少包含：

- overall AUROC
- per-dataset AUROC
- per-action AUROC
- optional ECE / Brier

---

## 10. Baseline harness（本版必须补）

## 10.1 第一批基础 baseline

Codex 先实现以下轻量基线：

1. `DirectAnswerBaseline`
2. `ThresholdRouterBaseline`
3. `MathHeuristicBaseline`
4. `SearchHeuristicBaseline`
5. `ClarifyHeuristicBaseline`

## 10.2 各 baseline 的最小定义

### A. DirectAnswerBaseline
- 无论什么问题，直接 `ANSWER`

### B. ThresholdRouterBaseline
- 根据 verbal confidence / action confidence 路由
- 低于阈值时不直接答

### C. MathHeuristicBaseline
- 在数学题上优先 `CALCULATE`
- 其他题不使用 `CALCULATE`

### D. SearchHeuristicBaseline
- 在 factual / fresh / long-tail / current-world 问题上优先 `SEARCH`

### E. ClarifyHeuristicBaseline
- 在 `IN3` 类 missing-info 问题上优先 `CLARIFY`

## 10.3 第二批正式 baseline（后续）

等主系统跑稳后，再逐步补：

- SMART-style supervised router
- Self-DC-style self-aware router

当前版本不要求优先 full reproduce KnowSelf。

## 10.4 baseline harness 文件建议

```text
src/baselines/
  base.py
  direct_answer.py
  threshold_router.py
  math_heuristic.py
  search_heuristic.py
  clarify_heuristic.py
  run_baselines.py
```

---

## 11. 训练阶段与评测阶段的执行分离

这是本版非常重要的工程原则。

## 11.1 训练阶段

训练阶段主要关注：

- 动作选择是否正确
- 局部效用是否合理
- chosen/rejected pair 是否高质量

因此：

- 不要求所有工具都真实执行
- 可以使用 mock / oracle / cached observation
- 重点在于动作边界学习

## 11.2 最终评测阶段

最终评测时再执行：

- 完整 search
- 完整 calculate
- 完整 clarify
- 完整 refuse/termination 流程

因此仓库需要把：

- **训练时工具代理层**
- **评测时真实执行层**

显式区分开。

---

## 12. Codex 的执行优先级（最新版）

## Step 1：动作空间升级为 5 动作

全局搜索并修改：
- prompt
- parser
- sandbox
- oracle
- utility
- eval
- pair builder

将动作枚举统一为：
- `ANSWER`
- `SEARCH`
- `CALCULATE`
- `CLARIFY`
- `REFUSE`

## Step 2：Search 模块重构

目标：
- 去掉 oracle leakage
- 加入 mock/local/online backend 抽象
- 默认训练时使用 mock/local

## Step 3：教师标注模块

目标：
- 新增 teacher client
- 新增语义 tag / standard action / brief_rationale 标注
- 预留 API Key

## Step 4：semantic tags v2

目标：
- 从 time-sensitive 中心版本升级为 factual / knowledge-boundary 版本
- 加入 `CALCULATION_REQUIRED`

## Step 5：shared-prefix pair 构造升级

目标：
- 把 pair 构造改成真正的 state-conditioned pair
- prompt 中固定前缀
- chosen/rejected 只比较动作 continuation

## Step 6：calibration pipeline

目标：
- rollout 中补 confidence 字段
- 实现 AUROC 主指标
- 保留 ECE / Brier 作为辅助

## Step 7：baseline harness

目标：
- 先补基础 baseline
- 与主系统统一输出格式
- 能与主评测脚本共用

## Step 8：再考虑正式大实验

包括：
- larger rollout
- DPO 正式训练
- OOD 测试
- 后续再接在线 search stress test

---

## 13. 推荐目录结构（本版新增）

```text
src/
  agent/
    tool_call_adapter.py             # optional adapter, not main path
  teacher/
    client.py
    prompts.py
    label_semantic_tags.py
    label_standard_action.py
    label_brief_rationale.py
  tools/
    search/
      base.py
      dispatcher.py
      mock_retriever.py
      local_retriever.py
      serper_backend.py
      brave_backend.py
      tavily_backend.py
      exa_backend.py
  baselines/
    base.py
    direct_answer.py
    threshold_router.py
    math_heuristic.py
    search_heuristic.py
    clarify_heuristic.py
    run_baselines.py
  eval/
    evaluate_calibration.py
    plot_calibration.py
    calibration_utils.py
```

---

## 14. 配置文件新增要求

Codex 需要新增或重写以下配置：

### `configs/teacher.yaml`

```yaml
teacher:
  enabled: false
  provider: openai
  model: gpt-4o
  api_key_env: OPENAI_API_KEY
  base_url_env: OPENAI_BASE_URL
  temperature: 0.0
  max_retries: 3
  batch_size: 8
```

### `configs/tools.yaml`

```yaml
tools:
  search:
    backend: mock_retriever
    top_k: 3
    enable_online_backend: false
  calculator:
    mode: safe_python
  clarify:
    mode: oracle_once
  refuse:
    mode: terminal
```

### `configs/scoring.yaml`

```yaml
scoring:
  lambda_search: 0.1
  lambda_calc: 0.05
  lambda_clar: 0.1
  answer_correct: 1.0
  answer_wrong: -1.0
  search_helpful: 0.6
  search_unhelpful: -0.1
  calculate_helpful: 0.8
  calculate_unhelpful: -0.1
  clarify_helpful: 0.5
  clarify_unhelpful: -0.1
  refuse_justified: 0.4
  refuse_unjustified: -0.6
```

---

## 15. 第一阶段验收标准（必须全部满足）

当 Codex 完成本轮迭代后，应至少满足：

1. 动作空间已统一升级为 5 动作
2. Search 模块已具备 mock/local/online backend 抽象
3. 训练阶段无需真实联网即可跑通
4. 教师标注模块已预留 API Key，并能在关闭状态下正常运行
5. pair 构造已支持 shared-prefix / state-conditioned DPO 数据
6. calibration pipeline 已输出 AUROC
7. baseline harness 已能跑基础 baseline
8. rollout / train / eval / baseline 四类脚本都能 smoke test

如果上述 8 条全部满足，则说明本轮工程升级完成。

---

## 16. 一句话总结（给 Codex 的最终任务定义）

> 在当前仓库基础上，将 MCAgent 从“四动作、time-sensitive 偏置、oracle-heavy prototype”升级为“五动作、factual/ability-boundary 对齐、支持教师辅助标注、训练与评测分离、以 AUROC 为主评测、具备基础 baseline harness 的正式实验系统。”
