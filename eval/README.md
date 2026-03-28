# eval 模块开发说明

## 目录结构

```
eval/
├── data_loader.py               # 人员A - 数据加载
├── llm_client.py                # 人员C - LLM 调用
├── run_eval.py                  # 人员B - 主入口
├── requirements.txt
└── metrics/
    ├── __init__.py              # 人员B
    ├── tool_call_accuracy.py    # 人员A - 工具调用准确度
    ├── planning_rationality.py  # 人员B - 规划合理性
    └── task_completion.py       # 人员C - 任务完成度
```

## 快速运行

```bash
# 只跑不需要 LLM 的两个指标
python3 eval/run_eval.py --skip-llm

# 全部跑（默认用 claude cli 作为 LLM 后端）
python3 eval/run_eval.py

# 用 openai 兼容 API
LLM_BACKEND=openai LLM_BASE_URL=https://xxx LLM_API_KEY=xxx LLM_MODEL=glm-4-flash python3 eval/run_eval.py
```

## 公共接口约定

### data_loader.py（人员A维护）

给其他人用的函数：

- `load_eval_dataset(path=None) -> list[dict]` — 加载数据集，默认找项目根目录的 `eval_dataset.json`
- `parse_step_input(input_str) -> dict` — 把 step 里的 input json 字符串解析成 dict，解析失败返回 `{}`
- `get_steps_summary(steps) -> str` — 把步骤列表变成人能读的文本摘要，给 LLM prompt 用的

### llm_client.py（人员C维护）

给其他人用的函数：

- `chat(prompt, max_retries=3) -> str` — 发 prompt 拿回复，自带重试
- `extract_json(text) -> dict` — 从 LLM 回复里抠出 JSON，会试三种策略（直接解析 / ```json 代码块 / 花括号匹配），都失败就抛 ValueError

后端通过环境变量切换：
| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_BACKEND` | `cli` / `openai` / `ollama` | `cli` |
| `LLM_CLI_COMMAND` | cli 模式用的命令 | `claude-internal` |
| `LLM_MODEL` | 模型名 | (空) |
| `LLM_API_KEY` | openai 模式的 key | (空) |
| `LLM_BASE_URL` | openai 模式的 url | (空) |
| `LLM_OLLAMA_URL` | ollama 地址 | `http://localhost:11434` |

### 指标类的统一接口

三个指标类都长这样，`run_eval.py` 靠这个接口串起来的：

```python
class XxxMetric:
    name = "指标名称"

    def evaluate(self, dataset: list[dict]) -> dict:
        # dataset 就是 load_eval_dataset() 的返回值
        return {
            "metric": self.name,
            "per_sample": [...],   # 每个样本的详细结果
            "summary": {...},      # 汇总统计
        }
```

如果要加新指标，照这个格式写就行，然后在 `__init__.py` 和 `run_eval.py` 里注册一下。

## 各指标简要说明

| 指标 | 需要LLM? | 核心思路 |
|------|---------|---------|
| 工具调用准确度 | ❌ | 按顺序对比 steps vs expected_steps，比较工具名+参数，另外算个不管顺序的 F1 |
| 规划合理性 | ❌ | 三个子分：冗余检测（重复调用）、完整性（覆盖期望步骤）、连贯性（依赖关系是否满足） |
| 任务完成度 | ✅ | 把 query/ground_truth/final_answer/步骤摘要 丢给 LLM 判断完没完成，返回 0/1 |

## 注意事项

- 改公共模块（data_loader / llm_client）前跟对应负责人说一声，不然可能互相冲突
- 不需要 LLM 的指标本地直接跑 `--skip-llm` 就行，调试方便
- `run_eval.py` 输出的 JSON 结果文件名带时间戳，别提交到 git 里（已经在数据集同级目录下生成）
