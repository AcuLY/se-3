"""数据加载与预处理模块。"""

import json
import os
from typing import Any


def load_eval_dataset(path: str | None = None) -> list[dict[str, Any]]:
    """加载评估数据集。

    Args:
        path: eval_dataset.json 的路径。默认为项目根目录下的文件。

    Returns:
        评估样本列表。
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "eval_dataset.json")
    path = os.path.abspath(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 基本校验
    required_keys = {"task_id", "user_query", "ground_truth", "steps", "expected_steps", "final_answer"}
    for i, sample in enumerate(data):
        missing = required_keys - set(sample.keys())
        if missing:
            raise ValueError(f"样本 {i} 缺少字段: {missing}")

    print(f"成功加载 {len(data)} 个评估样本")
    return data


def parse_step_input(input_str: str) -> dict[str, Any]:
    """解析步骤的 input JSON 字符串为字典。

    Args:
        input_str: JSON 格式的输入参数字符串。

    Returns:
        解析后的字典。解析失败返回空字典。
    """
    try:
        return json.loads(input_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def get_steps_summary(steps: list[dict]) -> str:
    """将步骤列表转换为简要的文本摘要。

    Args:
        steps: 步骤列表。

    Returns:
        步骤摘要文本。
    """
    lines = []
    for step in steps:
        step_num = step.get("step", "?")
        thought = step.get("thought", "")
        tool = step.get("tool_call", "")
        lines.append(f"步骤{step_num}: {thought} -> 调用 {tool}")
    return "\n".join(lines)
