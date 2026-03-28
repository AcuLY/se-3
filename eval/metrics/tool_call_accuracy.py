"""人员A：工具调用准确度（Tool Call Accuracy）评估指标。

参考 Ragas ToolCallAccuracy 实现，评估 Agent 实际调用的工具是否与期望一致。
包含两个子指标：
1. Tool Call Accuracy：按顺序配对比较，考虑工具名称和参数匹配
2. Tool Call F1：不考虑顺序，计算 Precision / Recall / F1
"""

import json
from typing import Any


def _normalize_value(value: Any) -> str:
    """归一化参数值用于比较。

    将值转为小写字符串，去除多余空格。
    """
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, dict):
        # 递归归一化字典
        return json.dumps(
            {k: _normalize_value(v) for k, v in sorted(value.items())},
            ensure_ascii=False,
            sort_keys=True,
        )
    if isinstance(value, list):
        return json.dumps([_normalize_value(v) for v in value], ensure_ascii=False)
    return str(value).strip().lower()


def _parse_input(input_str: str) -> dict[str, Any]:
    """解析步骤的 input 字段为字典。"""
    try:
        return json.loads(input_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _compute_arg_score(actual_input: str, expected_input: str) -> float:
    """计算两个步骤的参数匹配分数。

    逐字段比较归一化后的参数值，返回匹配字段比例。

    Args:
        actual_input: 实际步骤的 input JSON 字符串。
        expected_input: 期望步骤的 input JSON 字符串。

    Returns:
        参数匹配分数 (0.0 ~ 1.0)。
    """
    actual_args = _parse_input(actual_input)
    expected_args = _parse_input(expected_input)

    if not expected_args and not actual_args:
        return 1.0
    if not expected_args or not actual_args:
        return 0.0

    # 以期望参数的 key 集合为基准
    all_keys = set(expected_args.keys()) | set(actual_args.keys())
    if not all_keys:
        return 1.0

    match_count = 0
    for key in all_keys:
        if key in actual_args and key in expected_args:
            if _normalize_value(actual_args[key]) == _normalize_value(expected_args[key]):
                match_count += 1

    return match_count / len(all_keys)


def _compute_step_score(actual_step: dict, expected_step: dict) -> float:
    """计算单个步骤的匹配分数。

    工具名称匹配占 0.4 权重，参数匹配占 0.6 权重。

    Args:
        actual_step: 实际执行的步骤。
        expected_step: 期望的步骤。

    Returns:
        步骤匹配分数 (0.0 ~ 1.0)。
    """
    # 工具名称匹配
    actual_tool = actual_step.get("tool_call", "").strip().lower()
    expected_tool = expected_step.get("tool_call", "").strip().lower()

    if actual_tool != expected_tool:
        return 0.0  # 工具名不匹配，整步为 0

    # 工具名匹配，计算参数分数
    arg_score = _compute_arg_score(
        actual_step.get("input", "{}"),
        expected_step.get("input", "{}"),
    )

    # 工具名匹配权重 0.4 + 参数匹配权重 0.6
    return 0.4 + 0.6 * arg_score


def compute_tool_call_accuracy(sample: dict) -> dict[str, float]:
    """计算单个样本的工具调用准确度（顺序匹配）。

    将 steps 和 expected_steps 按顺序配对，逐步比较。
    长度不一致时，多余/缺失步骤得 0 分。

    Args:
        sample: 包含 steps 和 expected_steps 的评估样本。

    Returns:
        包含 accuracy 分数的字典。
    """
    actual_steps = sample.get("steps", [])
    expected_steps = sample.get("expected_steps", [])

    if not expected_steps:
        return {"accuracy": 1.0 if not actual_steps else 0.0}

    max_len = max(len(actual_steps), len(expected_steps))
    total_score = 0.0

    for i in range(max_len):
        if i < len(actual_steps) and i < len(expected_steps):
            total_score += _compute_step_score(actual_steps[i], expected_steps[i])
        # 超出范围的步骤得 0 分

    accuracy = total_score / max_len
    return {"accuracy": round(accuracy, 4)}


def _get_tool_call_signature(step: dict) -> str:
    """生成工具调用的签名用于集合比较（F1 计算）。"""
    tool = step.get("tool_call", "").strip().lower()
    args = _parse_input(step.get("input", "{}"))
    normalized_args = {k: _normalize_value(v) for k, v in sorted(args.items())}
    return f"{tool}:{json.dumps(normalized_args, ensure_ascii=False, sort_keys=True)}"


def compute_tool_call_f1(sample: dict) -> dict[str, float]:
    """计算单个样本的工具调用 F1 分数（不考虑顺序）。

    Args:
        sample: 包含 steps 和 expected_steps 的评估样本。

    Returns:
        包含 precision, recall, f1 的字典。
    """
    actual_steps = sample.get("steps", [])
    expected_steps = sample.get("expected_steps", [])

    actual_sigs = [_get_tool_call_signature(s) for s in actual_steps]
    expected_sigs = [_get_tool_call_signature(s) for s in expected_steps]

    if not expected_sigs and not actual_sigs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # 计算 True Positives（使用多集合交集，处理重复调用）
    actual_remaining = list(actual_sigs)
    tp = 0
    for sig in expected_sigs:
        if sig in actual_remaining:
            tp += 1
            actual_remaining.remove(sig)

    precision = tp / len(actual_sigs) if actual_sigs else 0.0
    recall = tp / len(expected_sigs) if expected_sigs else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


class ToolCallAccuracyMetric:
    """工具调用准确度评估指标。"""

    name = "工具调用准确度 (Tool Call Accuracy)"

    def evaluate(self, dataset: list[dict]) -> dict[str, Any]:
        """对整个数据集进行评估。

        Args:
            dataset: 评估样本列表。

        Returns:
            包含每个样本的分数和汇总统计的结果字典。
        """
        results = []
        for sample in dataset:
            accuracy = compute_tool_call_accuracy(sample)
            f1_scores = compute_tool_call_f1(sample)
            results.append({
                "task_id": sample["task_id"],
                **accuracy,
                **f1_scores,
            })

        # 汇总统计
        accuracies = [r["accuracy"] for r in results]
        f1s = [r["f1"] for r in results]

        summary = {
            "avg_accuracy": round(sum(accuracies) / len(accuracies), 4),
            "avg_f1": round(sum(f1s) / len(f1s), 4),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "perfect_count": sum(1 for a in accuracies if a == 1.0),
            "zero_count": sum(1 for a in accuracies if a == 0.0),
        }

        return {
            "metric": self.name,
            "per_sample": results,
            "summary": summary,
        }
