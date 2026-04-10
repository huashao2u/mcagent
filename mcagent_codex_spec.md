# MCAgent 工程实现说明（供 Codex 阅读与执行）

## 0. 文档用途

本文件用于指导 Codex 在当前工程目录中，基于已有数据与模型，逐步完成 **MCAgent** 的代码构建、实验规划与最小可运行原型实现。

本文档不是论文，而是 **工程规格说明 + 研究实现路线图**。

目标是让 Codex：

1. 理解本项目的核心研究问题。
2. 明确应优先实现哪些模块。
3. 使用当前目录中的数据与模型资源完成最小闭环。
4. 为后续训练、评测、消融实验预留清晰接口。

---

## 1. 当前工程状态

当前工程目录中已经具备：

- `dataset/`
  - `competition_math/`
  - `freshqa/`
  - `gsm8k/`
  - `IN3/`
  - `MintQA-Ti-v0.1/`
- `models/`
  - `Qwen2.5-7B-Instruct/`
- conda 环境已基本配置完成
- 后续代码开发依赖 VS Code + Codex

### 1.1 默认路径约定

除非项目中已存在其它约定，否则统一假设：

- 数据根目录：`./dataset`
- 模型根目录：`./models/Qwen2.5-7B-Instruct`
- 项目代码根目录：当前工程目录

Codex 在实现时应尽量通过配置文件或命令行参数指定路径，避免把绝对路径写死。

---

## 2. 研究目标（供实现时始终对齐）

### 2.1 核心问题

本项目不把问题定义为“让模型输出更准的 verbal confidence”，而定义为：

# **Action Calibration**

即：模型在面对不同能力边界/知识边界/信息边界时，能否把内部不确定性正确映射成合适动作。

### 2.2 统一动作空间

本项目中的动作空间统一为四类：

- `ANSWER`：直接回答
- `SEARCH`/`CALCULATE`：调用外部检索工具/调用外部运算工具
- `CLARIFY`：向用户追问缺失信息
- `REFUSE`：拒绝回答

注意：
- 工程上，除 `ANSWER` 外，其余动作均可实现为 tool-calling。
- `ANSWER` 是最终自然语言输出，而不是工具。
- 训练与评测都围绕这四类动作展开。

### 2.3 论文主张在工程中的落点

实现上不要把重点放在“生成漂亮的反思文字”，而要放在：

1. 在给定状态下，模型是否选择了合适动作。
2. 不同动作的局部效用如何计算。
3. 如何构造 DPO preference pairs。
4. 如何在已有 benchmark 上重构出 action-calibration 任务。

---

## 3. 方法总览（工程版）

## 3.1 统一决策结构

模型在每个样本上遵循如下结构：

```text
Question -> Reason -> Decision(Action) -> (Tool Observation if any) -> Final Output
```

为了简化第一阶段工程，不强制模型显式输出长篇 `Meta-Reflection` 文本。

推荐第一版实现：

- 先让模型产出 `Reason`
- 然后产出结构化 `Decision`
- `Decision` 包含：
  - `action`
  - `action_input`（如果动作需要参数）
  - 可选 `brief_rationale`

### 3.1.1 决策输出示意

推荐使用 JSON 结构，便于解析：

```json
{
  "reason": "...",
  "decision": {
    "action": "SEARCH",
    "action_input": {
      "query": "latest winner of ..."
    },
    "brief_rationale": "The question may require up-to-date factual knowledge."
  }
}
```

如果模型直接选择 `ANSWER`：

```json
{
  "reason": "...",
  "decision": {
    "action": "ANSWER",
    "action_input": {},
    "brief_rationale": "The problem is self-contained."
  }
}
```

---

## 4. 与前面对话一致的关键设计原则

### 4.1 不把过程特征要求模型自报

以下特征属于 **系统侧外生特征**，不要要求模型直接输出：

- trace length
- self-repair frequency
- branching / low-logit-margin proxy
- conflict words

它们的用途是：

1. 辅助 teacher/reflection 生成
2. 辅助构造 preference pairs
3. 辅助分析模型不确定性状态

而不是让模型“宣称自己生成了多少 token”。

### 4.2 局部效用，而非整轨迹终局效用

工程实现时，训练目标应围绕：

\[
U(a_t \mid s_t)
\]

而不是简单对整条轨迹定义一个终局奖励后，回头去惩罚某个早期动作。

理由：
- 避免 credit assignment 错误
- 避免“前一步工具选择正确，但后一步推理失败，导致前一步被误惩罚”

### 4.3 不把数据采集作为主线创新

本项目主线是 **方法论与工程实现**，不是新 benchmark 构建。

因此，应基于已有数据集进行 **task recasting**：

- 把原 QA / reasoning / intention 数据重构为 action-calibration 任务
- 而不是再单独建设一个庞大新数据集

---

## 5. 数据集角色划分（基于当前已有数据）

当前已有：

- `gsm8k`
- `competition_math`
- `freshqa`
- `IN3`
- `MintQA-Ti-v0.1`

### 5.1 数据集对应的任务角色

#### A. `gsm8k`

角色：
- 数学推理
- 自包含 closed-world 问题
- 用于评测 `ANSWER` 是否足够、是否存在不必要的 `SEARCH`

首版工程建议：
- 默认不允许 `REFUSE`
- `CLARIFY` 基本不应出现
- `SEARCH` 可以允许，但大多应被判为不必要

#### B. `competition_math`

角色：
- 更难的数学推理压力测试
- 用于制造“高挣扎但仍可能可答”的状态

首版工程建议：
- 动作重点仍是 `ANSWER vs SEARCH`
- 可接入计算器工具作为 external tool
- 后续可分析 trace-length 与 tool necessity 的关系

#### C. `freshqa`

角色：
- factual / knowledge boundary
- 动态知识
- false premise
- 需要 `SEARCH` 或 `REFUSE` 的重要场景

实现上：
- `ANSWER / SEARCH / REFUSE` 是主动作
- `CLARIFY` 一般不是主动作，但可以保留接口

#### D. `IN3`

角色：
- intention boundary
- 任务缺失信息
- `CLARIFY` 的主训练与主评测来源

实现上：
- `CLARIFY` 是关键动作
- 需要做 clarify-oracle 环境
- 可支持多轮，但首版建议先做“单次追问后再答”的简化版本

#### E. `MintQA-Ti-v0.1`

角色：
- new knowledge / long-tail knowledge / multi-hop factual reasoning
- 适合作为 `ANSWER vs SEARCH` 的主训练来源之一

实现上：
- 优先作为 factual boundary 训练主料
- 后续可用于生成更多 DPO pairs

### 5.2 推荐的数据集分工

第一阶段建议：

#### 训练/开发主料
- `gsm8k`
- `competition_math`
- `IN3`
- `MintQA-Ti-v0.1`

#### 主要 OOD/外测
- `freshqa`

说明：
- 第一阶段不必把所有数据集都强行塞进训练
- 先用 `MintQA + IN3 + GSM8K/competition_math` 建最小闭环
- 再用 `freshqa` 做外测更有说服力

---

## 6. 工程上要实现的动作工具（Tool API）

建议统一把非 `ANSWER` 动作实现为工具调用。

### 6.1 `search`

用途：
- factual knowledge retrieval
- external evidence gathering

输入：
```json
{
  "query": "string"
}
```

输出：
```json
{
  "results": ["...", "..."],
  "metadata": {...}
}
```

第一阶段可用简化版：
- 不真实联网
- 直接读取本地 corpus / gold evidence / mock retrieval
- 先保证接口跑通

### 6.2 `calculator`

用途：
- 数学计算

输入：
```json
{
  "expression": "string"
}
```

输出：
```json
{
  "result": "..."
}
```

第一阶段建议：
- 用 Python 安全计算器实现
- 限制表达式范围

### 6.3 `clarify`

用途：
- 对缺失信息进行追问

输入：
```json
{
  "question": "string"
}
```

输出：
```json
{
  "user_reply": "..."
}
```

实现方式：
- 不接真实用户
- 对接 `IN3` 的 gold missing-slot / hidden intention 信息
- 首版只支持一次 clarify

### 6.4 `refuse`

用途：
- 结构化拒答

输入：
```json
{
  "reason": "string"
}
```

输出：
```json
{
  "status": "refused"
}
```

说明：
- `refuse` 在工程上可实现为 tool
- 但语义上它是终止动作
- 调用后结束 rollout

---

## 7. 数据流水线（Codex 需要优先实现）

## 7.1 第一步：统一数据加载器

请实现统一的数据读取模块，建议路径：

- `src/data/loaders.py`

要求：
- 为每个数据集写独立 loader
- 再写一个统一 adapter，把样本转成统一字段：

```python
{
    "id": str,
    "dataset": str,
    "question": str,
    "gold_answer": str | list | None,
    "metadata": dict,
    "task_type": str,
}
```

`task_type` 初步可设为：
- `math`
- `factual_boundary`
- `intention_boundary`

## 7.2 第二步：统一 prompt builder

建议路径：

- `src/prompting/build_prompts.py`

任务：
- 给 Qwen2.5-7B-Instruct 构造统一 system/user prompt
- 模型需要输出结构化 decision
- 支持启用/关闭 tool schema

第一版 prompt 目标：
- 稳定输出 JSON
- 明确只允许四种动作之一

## 7.3 第三步：工具环境 sandbox

建议路径：

- `src/envs/sandbox.py`
- `src/tools/search_tool.py`
- `src/tools/calculator_tool.py`
- `src/tools/clarify_tool.py`
- `src/tools/refuse_tool.py`

要求：
- 使用统一接口
- 支持 step-by-step 执行
- 记录每轮 observation
- 可复现实验日志

### 7.3.1 环境统一 step 接口

建议接口：

```python
obs, done, info = env.step(action_name, action_input)
```

其中：
- `obs`: 工具返回内容
- `done`: 是否结束
- `info`: 额外元信息

## 7.4 第四步：rollout engine

建议路径：

- `src/rollout/generate_rollouts.py`

任务：
- 给定一个样本，运行模型
- 产出 reasoning
- 解析 decision
- 调用工具
- 必要时继续后续生成
- 保存完整 rollout 日志

输出日志建议统一成 JSONL：

```python
{
    "id": ...,
    "dataset": ...,
    "question": ...,
    "reason": ...,
    "decision": {
        "action": ...,
        "action_input": ...,
        "brief_rationale": ...,
    },
    "tool_observation": ...,
    "final_answer": ...,
    "final_status": ...,
    "correctness": ...,
    "raw_text": ...,
}
```

---

## 8. 外生特征提取（tag 系统）

建议路径：

- `src/features/extract_process_features.py`
- `src/features/semantic_tags.py`

### 8.1 过程特征（脚本直接算）

请实现以下 tag：

- `STRUGGLE_LONG`
- `HAS_SELF_REPAIR`
- `LOW_LOGIT_MARGIN`
- `HIGH_BRANCHING`

说明：
- 第一版即使拿不到完整 logits，也先实现可替代版本
- 例如：
  - 用 token 数近似 trace length
  - 用正则匹配 self-repair 词
  - logits 相关先留接口，后续补全

### 8.2 语义 tag

请实现以下语义 tag 的规则化判定接口：

- `TIME_SENSITIVE`
- `FALSE_PREMISE`
- `MISSING_INFO`
- `TOOL_REQUIRED`
- `JUSTIFIED_REFUSE`

注意：
- 第一版不要求完全自动正确
- 先实现 rule-based skeleton
- 后续可接 teacher / LLM judge

---

## 9. 动作 oracle 与局部效用

建议路径：

- `src/scoring/action_oracle.py`
- `src/scoring/local_utility.py`

## 9.1 动作 oracle 规则

第一版请实现一个简单可解释的 oracle：

```python
if missing_critical_info and clarify_allowed:
    oracle_action = "CLARIFY"
elif external_evidence_available and retrieval_allowed:
    oracle_action = "SEARCH"
elif premise_false or unverifiable_under_current_tools:
    oracle_action = "REFUSE"
else:
    oracle_action = "ANSWER"
```

注意：
- 这是工程第一版
- 不要求一开始就是最优 oracle
- 关键是形成闭环

## 9.2 局部效用初始定义

请实现离散版局部效用表：

- `ANSWER_correct = +1.0`
- `ANSWER_wrong = -1.0`
- `REFUSE_justified = +0.4`
- `REFUSE_unjustified = -0.6`
- `SEARCH_helpful = +0.6 - lambda_tool`
- `SEARCH_unhelpful = -0.1 - lambda_tool`
- `CLARIFY_helpful = +0.5 - lambda_clar`
- `CLARIFY_unhelpful = -0.1 - lambda_clar`

初始参数：
- `lambda_tool = 0.1`
- `lambda_clar = 0.1`

要求：
- 将这些写成配置项，而不是硬编码
- 方便后续做 sweep

---

## 10. Preference pair 构造

建议路径：

- `src/pairs/build_pairs.py`

### 10.1 pair 构造原则

只保留：

1. 同一问题/同一状态下的候选动作
2. 动作不同
3. 局部效用差异足够大
4. 样本语义清晰

### 10.2 推荐优先保留的 pair 类型

- `SEARCH_helpful` vs `ANSWER_wrong`
- `ANSWER_correct` vs `SEARCH_unnecessary`
- `CLARIFY_helpful` vs `ANSWER_premature`
- `REFUSE_justified` vs `ANSWER_hallucinated`
- `ANSWER_correct` vs `REFUSE_unjustified`

### 10.3 未被保留的分支如何处理

请实现：
- 放入 diagnostics pool
- 不进入当前轮 DPO
- 保留日志，便于后续分析或再采样

---

## 11. 训练阶段设计

## 11.1 第一阶段：不默认做 SFT 预热

由于我们现在采用 tool-native action space，首版工程中：

- **不要默认实现 SFT warm-up 为必要阶段**
- 先直接测模型自然动作覆盖率

请实现一个覆盖率统计模块，统计：
- `ANSWER` 比例
- `SEARCH` 比例
- `CLARIFY` 比例
- `REFUSE` 比例

如果非 `ANSWER` 动作覆盖率过低，再补 warm-up。

### 11.1.1 覆盖率门槛建议

作为第一版默认阈值：
- 非 `ANSWER` 动作总占比 < 15% 时，提示需要 warm-up

## 11.2 第二阶段：Step-DPO

建议路径：

- `src/training/run_dpo.py`

要求：
- 基于 `prompt/chosen/rejected` 数据格式
- 兼容 TRL `DPOTrainer`
- 参考模型为当前 base / warm-up 模型
- 支持 LoRA

### 11.2.1 当前推荐训练框架

优先使用：
- `transformers`
- `trl`
- `peft`
- `accelerate`

不要在第一版里优先实现：
- verl
- PPO/GRPO
- 高频在线异步 RL

理由：
- 当前方法核心是 tool-calling + preference optimization
- 先用 TRL 跑通最小闭环
- 后续再考虑更复杂 RL 基础设施

---

## 12. 评测模块

建议路径：

- `src/eval/evaluate_actions.py`
- `src/eval/evaluate_answers.py`
- `src/eval/evaluate_calibration.py`
- `src/eval/aggregate_results.py`

### 12.1 第一阶段必须实现的指标

#### 结果指标
- accuracy / EM（数学）
- factual correctness（事实题）
- intention task success（IN3）

#### 动作指标
- action accuracy
- unnecessary search rate
- justified refuse rate
- over-refusal rate
- clarify helpfulness
- over-answer rate

#### 校准指标（后续接入）
- AUROC
- ECE
- Brier

第一阶段可先留接口，不必一开始全做完。

---

## 13. Baseline 规划（首版实现）

首版建议只先做轻量 baseline：

1. `DirectAnswer`
2. `ThresholdRouter`（若已有 confidence 分数则接入，否则先留空）
3. `ToolHeuristic`
4. `ClarifyHeuristic`

说明：
- 暂时不要优先 full reproduce SMART / Self-DC / KnowSelf
- 先保证 MCAgent 主闭环可以运行
- 后续再补 benchmark-level baseline

---

## 14. 推荐的项目目录结构

Codex 实现时建议采用如下目录：

```text
.
├── dataset/
├── models/
├── configs/
│   ├── default.yaml
│   ├── data.yaml
│   ├── training.yaml
│   └── eval.yaml
├── scripts/
│   ├── run_rollout.sh
│   ├── build_pairs.sh
│   ├── run_dpo.sh
│   └── eval.sh
├── src/
│   ├── data/
│   ├── prompting/
│   ├── envs/
│   ├── tools/
│   ├── rollout/
│   ├── features/
│   ├── scoring/
│   ├── pairs/
│   ├── training/
│   ├── eval/
│   └── utils/
├── outputs/
│   ├── rollouts/
│   ├── pairs/
│   ├── checkpoints/
│   └── reports/
└── README.md
```

---

## 15. Codex 的分步执行要求（最重要）

请 Codex 按如下顺序完成实现，不要一上来尝试做全部功能。

### Step 0：扫描当前目录

- 确认 `dataset/` 下各子目录结构
- 确认 `models/Qwen2.5-7B-Instruct/` 是否为可加载 Hugging Face 格式
- 输出一个简短的资源检查报告

### Step 1：搭建项目骨架

- 创建 `src/`、`configs/`、`scripts/`、`outputs/`
- 写最小 `README`
- 写默认配置文件

### Step 2：实现数据加载器

至少支持：
- gsm8k
- competition_math
- freshqa
- IN3
- MintQA-Ti-v0.1

要求：
- 统一输出字段
- 支持采样若干条样本进行 smoke test

### Step 3：实现 tool-native prompt 与 JSON decision parser

- 构造 prompt
- 解析模型输出
- 可以使用 json_repair 库，已安装
- 如果 JSON 解析失败，给出 fallback parser

### Step 4：实现 sandbox 与四类工具

- search（可先 mock）
- calculator
- clarify
- refuse

### Step 5：实现 rollout engine

- 选取每个数据集少量样本
- 跑通一条完整流程
- 保存 JSONL rollout 日志

### Step 6：实现 tag 抽取与局部效用

- 过程 tag
- 语义 tag（先规则版）
- 动作 oracle
- 局部效用函数

### Step 7：实现 pair 构造

- 从 rollout 日志生成 chosen/rejected pair
- 导出 TRL 可用数据格式

### Step 8：实现最小 DPO 训练

- LoRA + DPOTrainer
- 跑一个极小训练样例
- 验证整个闭环无报错

### Step 9：实现基础评测

- action accuracy
- unnecessary search rate
- justified refuse rate
- math accuracy

### Step 10：整理实验脚本

- rollout
- pair build
- train
- eval

并保证这些脚本可通过命令行运行。

---

## 16. 第一阶段实验建议（给 Codex 的默认实验规划）

### Phase A：最小闭环

只使用：
- `gsm8k`
- `IN3`
- `MintQA-Ti-v0.1`

目标：
- 跑通工具调用
- 跑通局部效用
- 跑通 pair 构造
- 跑通 DPO

### Phase B：加入 `freshqa`

目标：
- 测试 factual boundary OOD
- 检查 `SEARCH/REFUSE/ANSWER` 边界

### Phase C：再考虑更大规模实验

包括：
- 更多样本
- 更多 pair 类型
- warm-up（若覆盖率不足）
- 更正式 baseline

---

## 17. 明确不该在第一阶段优先做的事情

Codex 在第一版中不要优先投入大量精力到以下内容：

1. 复杂在线 RL 基础设施
2. verl 迁移
3. vllm 深度集成
4. 全量 SMART / Self-DC / KnowSelf 复现
5. 新 benchmark 采集
6. 非必要的 fancy UI
7. 超大模型训练

第一阶段的唯一目标是：

# **在当前已有数据与模型上，完成一个可运行的 MCAgent 最小研究原型。**

---

## 18. 最终成功标准

Codex 完成第一阶段后，应至少满足：

1. 可以加载本地 Qwen2.5-7B-Instruct
2. 可以读取至少 3 个数据集
3. 模型可以在统一 schema 下输出四动作之一
4. 工具环境可以执行 search/calculator/clarify/refuse
5. 可以生成 rollout 日志
6. 可以从 rollout 构造 DPO pairs
7. 可以跑一个小规模 DPO 训练样例
8. 可以输出基础 action-level 指标

如果上述 8 条都满足，则说明本项目工程闭环已经形成。

---

## 19. 给 Codex 的实现风格要求

1. 代码优先可运行，而不是过度抽象。
2. 每个模块先做最小可用版本，再逐步重构。
3. 所有关键路径都保留日志输出。
4. 所有配置尽量通过 yaml 或 argparse 暴露。
5. 对不确定的数据格式，先写探测脚本，不要猜。
6. 所有训练/评测脚本都应支持小样本 smoke test。

---

## 20. 一句话总结（供 Codex 把握主线）

> 本项目要实现的不是一个复杂炫技 agent，而是一个能够在现有 benchmark 上，把 `ANSWER / SEARCH / CLARIFY / REFUSE` 统一成可训练动作空间，并通过局部效用与 Step-DPO 学习动作边界的最小研究系统。

