# MCAgent Codex 修改任务清单（v5）

## 0. 文档目的

本文件用于指导 Codex 对当前仓库进行下一轮工程迭代。目标不是重写框架，而是在现有实现基础上，把仓库从“可运行预备版”升级为“可支撑正式实验的研究系统”。

当前仓库已经具备：

- 5 动作空间：`ANSWER / SEARCH / CALCULATE / CLARIFY / REFUSE`
- 统一数据加载
- JSON 决策输出与解析
- sandbox 工具环境
- rollout 记录
- 语义/过程 tag
- oracle 与局部效用
- shared-prefix 风格 pair 构造雏形
- TRL DPO 训练入口
- calibration 指标入口
- teacher 标注接口预留

但仍存在若干关键模块停留在 bootstrap/demo 级实现，需要重点升级。

---

## 1. 当前仓库状态总结

## 1.1 已完成且方向正确的部分

### A. 动作空间已正式升级为 5-way
仓库 README 与 prompt/action/oracle/utility 均已统一到：

- `ANSWER`
- `SEARCH`
- `CALCULATE`
- `CLARIFY`
- `REFUSE`

这与当前 MCAgent 方法主线一致。

### B. Search 已经模块化
当前仓库已实现 search backend dispatcher，支持：

- `mock_retriever`
- `local_retriever`
- `serper`
- `brave`
- `tavily`
- `exa`

说明 search 已从“单文件工具”升级为“可插拔后端”。

### C. 语义 TAG 已经从 time-sensitive 升级为 factual / knowledge boundary
当前仓库中语义标签已包含：

- `FRESH_FACT`
- `FALSE_PREMISE`
- `MISCONCEPTION_RISK`
- `NEW_OR_TAIL_KNOWLEDGE`
- `MISSING_INFO`
- `TOOL_REQUIRED`
- `JUSTIFIED_REFUSE`
- `CALCULATION_REQUIRED`

这已经较好对齐了我们最终的数据与方法叙事。

### D. calibration 模块不再是纯占位符
当前 `evaluate_calibration` 已支持：

- AUROC
- ECE
- Brier
- per-dataset 分析
- per-action 分析
- risk-coverage 摘要

### E. teacher 接口已预留
当前配置与代码中已预留：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `teacher.enabled`
- `teacher.model`
- semantic tag teacher labeling
- standard action teacher labeling

### F. pair 构造已开始朝 shared-prefix 思路靠近
当前 pair builder 已不再是简单整段 completion 对比，而是尝试把共享 prompt 与 action completion 分开。

---

## 1.2 当前最重要的不足

### A. SEARCH 仍不是“正式检索”
虽然结构已经拆好，但当前默认仍是：

- `mock_retriever`
- `local_retriever`

而它们主要仍依赖样本 metadata 拼接 evidence，不是真正基于外部/本地索引语料的检索。

### B. teacher client 仍是 stub
当前 teacher client 仅做配置读取与占位返回，还没有真正发起 OpenAI API 请求。

### C. DPO 主线仍偏 `oracle_pairs`
当前配置中的 curriculum 仍以 `oracle_pairs` 为主，尚未切换到真正的 `natural_branch_pairs` / `shared_state_pairs`。

### D. calibration 仍以 utility 差值代理模型动作置信
当前 `action_confidence` 不是来自模型真实 logits 或模型显式信心，而是来自候选 utility 排序差值，因此更适合开发诊断，不适合作为最终论文主结论。

### E. shared-prefix 还不够“真实状态共享”
当前 `reason_prefix` 主要还是固定模板，而不是真实 rollout 截断出来的 pre-action state。

### F. baseline harness 仍不完整
当前仓库尚未形成系统的 baseline 运行框架，至少缺少：

- `DirectAnswer`
- `ThresholdRouter`
- `MathHeuristicRouter`
- `SearchHeuristicRouter`
- `ClarifyHeuristicRouter`

---

## 2. 本轮迭代的总目标

本轮迭代不再新增复杂系统，而是完成以下四件事：

1. **把 SEARCH 从代理实现升级为可评测检索模块**
2. **把 teacher 标注从 stub 升级为真正可用**
3. **把 DPO 数据从 oracle-pairs 升级为 natural-branch shared-prefix pairs**
4. **把 calibration 从代理 confidence 升级为模型真实动作偏好评测**

只要这四步做成，仓库就能从“预备版”进入“正式实验版”。

---

## 3. Codex 修改任务（按优先级执行）

# P1. Search 模块正式化

## 3.1 目标

将当前 SEARCH 由 metadata-based proxy 升级为可复现实验使用的检索模块。

## 3.2 修改要求

### 3.2.1 保留现有多后端架构
保留当前目录结构：

```text
src/tools/search/
  dispatcher.py
  mock_retriever.py
  local_retriever.py
  serper_backend.py
  brave_backend.py
  tavily_backend.py
  exa_backend.py
```

### 3.2.2 调整后端角色

#### `mock_retriever`
- 仅用于 smoke test
- 默认不用于正式训练/评测

#### `local_retriever`
- 升级为默认训练/开发后端
- 不再直接读取 gold/metadata 作为 evidence
- 必须从本地索引语料中检索 top-k 结果

### 3.2.3 建立最小本地检索索引
新增模块：

```text
src/tools/search/indexing.py
src/tools/search/corpus_builder.py
```

要求：
- 从当前数据集可用字段中构造简易语料库
- 支持 BM25 或简单 embedding retrieval
- 至少返回：
  - `query`
  - `results`
  - `doc_ids`
  - `scores`
  - `metadata`

### 3.2.4 训练与最终评测分离
在配置里区分：

```yaml
tools:
  search:
    train_backend: local_retriever
    eval_backend: serper   # or brave/tavily/exa
```

说明：
- 训练阶段不要求真实联网调用所有工具
- 训练阶段主要关注动作选择是否正确
- 最终评测阶段再运行完整工具流程

## 3.3 验收标准

- `local_retriever` 不再直接拼接 metadata 作为结果
- 同一个 query 可从本地索引返回 top-k evidence
- rollout 日志中能区分 `train_proxy` 与 `real_eval_search`
- 配置切换后端不需要改主逻辑

---

# P2. Teacher 标注打通

## 4.1 目标

让 teacher 模块真正能够进行：

- 语义 TAG adjudication
- 标准动作 adjudication

## 4.2 修改要求

### 4.2.1 实现真正的 OpenAI teacher client
修改：

```text
src/teacher/client.py
```

要求：
- 从环境变量读取：
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
- 支持：
  - 默认 OpenAI 官方端点
  - 自定义兼容 base_url
- 支持 structured JSON 输出
- 出错时保留 fallback 行为

### 4.2.2 新增 teacher 标注脚本
新增：

```text
src/teacher/run_teacher_labeling.py
```

功能：
- 读取样本
- 先运行 rules 生成 `rule_tags` / `rule_action`
- 调用 teacher 补：
  - `teacher_tags`
  - `teacher_action`
  - `teacher_note`
- 输出 JSONL 标注文件

### 4.2.3 teacher 输出格式统一
统一输出：

```json
{
  "id": "...",
  "rule_tags": [...],
  "teacher_tags": [...],
  "final_tags": [...],
  "rule_action": "...",
  "teacher_action": "...",
  "final_action": "...",
  "teacher_note": "..."
}
```

## 4.3 验收标准

- 设置 `teacher.enabled=true` 且提供 key 后，可真实调用 teacher
- teacher 标注脚本可批量输出 JSONL
- key 缺失时自动 fallback 为 rule-only

---

# P3. Pair 构造升级为 natural-branch shared-prefix Step-DPO

## 5.1 目标

让 DPO 主数据不再主要来自 `oracle_pairs`，而来自共享状态上的自然动作分叉。

## 5.2 修改要求

### 5.2.1 新增真实状态导出
在 rollout 日志中新增：

- `reason_prefix`
- `history_prefix`
- `state_prompt`
- `state_snapshot`

要求：
- `state_prompt` 是“到动作发生前”的真实输入状态
- 不再只用固定模板文本代替

### 5.2.2 新增 branching pair builder
新增：

```text
src/pairs/build_natural_branch_pairs.py
```

功能：
- 对同一 `state_prompt` 采多条 continuation
- 允许 5 动作：
  - `ANSWER`
  - `SEARCH`
  - `CALCULATE`
  - `CLARIFY`
  - `REFUSE`
- 计算每个动作的局部效用
- 只保留高质量 pair

### 5.2.3 pair 选择规则
保留 pair 必须满足：

1. 同一 `state_prompt`
2. 动作不同
3. `utility_gap >= min_utility_gap`
4. 样本语义清晰

### 5.2.4 curriculum 切换
修改配置：

```yaml
training:
  curriculum:
    train_builder: natural_branch_pairs
    eval_builder: natural_branch_pairs
```

同时保留：
- `oracle_pairs` 作为 bootstrap / smoke / fallback

## 5.3 共享前缀实现原则

### 重要说明
共享前缀不要求整条轨迹完全一致，而要求：

> **在动作发生前的决策状态尽量一致。**

因此正式 pair 应采用：

- `prompt = state_prompt`
- `chosen/rejected = short action completion`

不要在 DPO 中重新生成完整前缀。

## 5.4 验收标准

- 仓库可切换 `oracle_pairs` 与 `natural_branch_pairs`
- `natural_branch_pairs` 使用真实 `state_prompt`
- 输出的 pair 数据仍兼容 TRL `prompt/chosen/rejected`

---

# P4. Calibration 升级为“真实模型信号”

## 6.1 目标

把当前基于 utility margin 的代理 confidence，升级为真实模型动作偏好或真实 verbal confidence 的评测链路。

## 6.2 修改要求

### 6.2.1 rollout 中补充真实信号
在 rollout 里新增以下字段中的至少一类：

#### 方案 A：模型显式输出
- `decision.confidence`
- 或 `decision.confidence_band`

#### 方案 B：模型隐式输出
- action logits / action token probs
- candidate action probabilities

优先建议：
- 先实现 `decision.confidence` 的轻量版本
- 后续再补 logit-level action confidence

### 6.2.2 calibration 指标策略
保留：
- AUROC
- ECE
- Brier
- risk-coverage

但在主报告中：
- **AUROC 作为 headline 指标**
- ECE / Brier 作为辅助分析

### 6.2.3 区分两种 calibration
在报告中明确区分：

- `decision_calibration`：动作是否选对
- `answer_calibration`：答案是否正确

必要时拆开评估。

## 6.3 验收标准

- `action_confidence` 不再仅由 utility gap 构造
- rollout 日志里存在真实模型相关 confidence 字段
- calibration 报告能区分 overall / per-action / per-dataset

---

# P5. baseline harness 补齐

## 7.1 目标

建立最小但正式的 baseline 对照体系。

## 7.2 新增目录

```text
src/baselines/
```

建议至少包含：

- `direct_answer.py`
- `threshold_router.py`
- `math_heuristic_router.py`
- `search_heuristic_router.py`
- `clarify_heuristic_router.py`
- `run_baselines.py`

## 7.3 baseline 最低要求

### DirectAnswer
- 所有样本直接输出 `ANSWER`

### ThresholdRouter
- 使用已有 confidence / utility proxy 做简单阈值路由

### MathHeuristicRouter
- 数学域优先 `CALCULATE`

### SearchHeuristicRouter
- factual boundary 域优先 `SEARCH`

### ClarifyHeuristicRouter
- `MISSING_INFO` 时优先 `CLARIFY`

## 7.4 验收标准

- 所有 baseline 可在同一 harness 下运行
- 输出与主系统一致的 rollout/eval 格式
- 可直接与 MCAgent 主模型对比 action-level 指标

---

## 4. 文件级修改清单

### 必改文件

```text
src/tools/search/mock_retriever.py
src/tools/search/local_retriever.py
src/teacher/client.py
src/pairs/build_pairs.py
src/rollout/generate_rollouts.py
src/eval/evaluate_calibration.py
configs/default.yaml
```

### 必新增文件

```text
src/tools/search/indexing.py
src/tools/search/corpus_builder.py
src/teacher/run_teacher_labeling.py
src/pairs/build_natural_branch_pairs.py
src/baselines/direct_answer.py
src/baselines/threshold_router.py
src/baselines/math_heuristic_router.py
src/baselines/search_heuristic_router.py
src/baselines/clarify_heuristic_router.py
src/baselines/run_baselines.py
```

---

## 5. 迭代顺序（Codex 必须遵守）

### Step 1
先改 `SEARCH`：
- local retriever 真正可用
- 不再直接依赖 metadata 泄漏

### Step 2
打通 teacher：
- 可真实调用 API
- 可输出 teacher 标注 JSONL

### Step 3
升级 rollout：
- 保存真实 `state_prompt`
- 保存 confidence 相关字段

### Step 4
新增 `natural_branch_pairs`
- 仍保留 `oracle_pairs`
- 配置可切换

### Step 5
补 calibration 主线
- AUROC headline
- ECE/Brier 辅助

### Step 6
补 baseline harness
- 形成可比矩阵

---

## 6. 本轮迭代完成后的成功标准

如果本轮迭代完成，仓库应满足：

1. `SEARCH` 可用真实本地索引或在线后端执行，不再主要依赖 metadata 占位
2. teacher 标注可真实运行，并输出结构化标签与动作建议
3. DPO 训练可切到 `natural_branch_pairs`
4. calibration 可基于真实模型信号评测，而不只基于 utility proxy
5. baseline harness 可运行至少 4 个基础基线
6. 主系统与基线可在统一评测脚本下比较 action-level 指标

---

## 7. 一句话总目标

> 在现有 5 动作 MCAgent 预备版基础上，把仓库升级成一个围绕 `ANSWER / SEARCH / CALCULATE / CLARIFY / REFUSE` 的正式 action-calibration 实验系统，其中训练主线使用 shared-prefix Step-DPO，search 具备可评测检索能力，teacher 标注可真实调用，calibration 以 AUROC 为核心，baseline 对照可系统复现。
