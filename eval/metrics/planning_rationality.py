"""人员B：规划合理性（Planning Rationality）评估指标。

自定义指标，通过规则和启发式方法评估 Agent 执行轨迹的合理性。
包含三个子指标：
1. 冗余检测（Redundancy Score）：检测重复/无效步骤
2. 步骤完整性（Completeness Score）：检查是否覆盖所有必要步骤
3. 逻辑连贯性（Coherence Score）：检查工具调用的依赖关系是否满足
"""

import json
from typing import Any


# 工具调用的前置依赖关系定义
# key: 工具名称, value: 该工具需要的前置工具列表（至少有一个在前面出现过）
TOOL_DEPENDENCIES = {
    "search_products": ["search_restaurants"],  # 需要先搜索餐厅获取 restaurant_id
    "place_order": ["search_products"],          # 需要先搜索商品获取 product_id
    "pay_order": ["place_order"],                # 需要先下单获取 order_id
    "check_order_status": ["place_order"],       # 需要先下单获取 order_id
}

# 参数依赖关系：某个工具的参数应能从前置步骤的 observation 中获取
PARAM_DEPENDENCIES = {
    "search_products": {"restaurant_id": "search_restaurants"},
    "place_order": {"restaurant_id": "search_restaurants", "items": "search_products"},
    "pay_order": {"order_id": "place_order"},
    "check_order_status": {"order_id": "place_order"},
}


def _parse_input(input_str: str) -> dict[str, Any]:
    """解析步骤的 input 字段。"""
    try:
        return json.loads(input_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_observation(obs_str: str) -> Any:
    """解析步骤的 observation 字段。"""
    try:
        return json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return obs_str


def compute_redundancy_score(steps: list[dict]) -> dict[str, Any]:
    """计算冗余步骤检测分数。

    检测连续重复的工具调用（相同 tool_call + 相同 input）。

    Args:
        steps: Agent 实际执行的步骤列表。

    Returns:
        冗余检测结果，包含分数和冗余步骤详情。
    """
    if len(steps) <= 1:
        return {"score": 1.0, "redundant_steps": [], "redundant_count": 0}

    redundant_steps = []
    for i in range(1, len(steps)):
        prev = steps[i - 1]
        curr = steps[i]
        if (curr.get("tool_call", "") == prev.get("tool_call", "") and
                curr.get("input", "") == prev.get("input", "")):
            redundant_steps.append(curr.get("step", i + 1))

    redundancy_rate = len(redundant_steps) / len(steps)
    score = 1.0 - redundancy_rate

    return {
        "score": round(score, 4),
        "redundant_steps": redundant_steps,
        "redundant_count": len(redundant_steps),
    }


def compute_completeness_score(
    actual_steps: list[dict],
    expected_steps: list[dict],
) -> dict[str, Any]:
    """计算步骤完整性分数。

    检查实际轨迹是否覆盖了所有期望的工具调用类型。

    Args:
        actual_steps: Agent 实际执行的步骤。
        expected_steps: 期望的步骤序列。

    Returns:
        完整性检测结果。
    """
    if not expected_steps:
        return {"score": 1.0, "missing_tools": [], "extra_tools": []}

    expected_tools = [s.get("tool_call", "") for s in expected_steps]
    actual_tools = [s.get("tool_call", "") for s in actual_steps]

    # 检查覆盖率：期望的工具调用是否都被实际执行了
    remaining_actual = list(actual_tools)
    covered = 0
    missing_tools = []

    for tool in expected_tools:
        if tool in remaining_actual:
            covered += 1
            remaining_actual.remove(tool)
        else:
            missing_tools.append(tool)

    coverage = covered / len(expected_tools) if expected_tools else 1.0

    # 额外步骤（不在期望中的）
    extra_tools = remaining_actual

    return {
        "score": round(coverage, 4),
        "missing_tools": missing_tools,
        "extra_tools": extra_tools,
    }


def compute_coherence_score(steps: list[dict]) -> dict[str, Any]:
    """计算逻辑连贯性分数。

    检查每一步的工具调用是否满足前置依赖关系：
    1. 该工具的前置工具是否已在前面执行过
    2. 该工具的关键参数是否能从前面步骤的 observation 中获取

    Args:
        steps: Agent 实际执行的步骤列表。

    Returns:
        连贯性检测结果。
    """
    if not steps:
        return {"score": 1.0, "violations": []}

    executed_tools = []  # 已执行过的工具列表
    observations = []    # 前面步骤的所有 observation 文本
    violations = []
    dependency_checks = 0
    satisfied_checks = 0

    for step in steps:
        tool = step.get("tool_call", "")
        step_num = step.get("step", "?")

        # 检查工具级别依赖
        if tool in TOOL_DEPENDENCIES:
            dependency_checks += 1
            required_tools = TOOL_DEPENDENCIES[tool]
            if any(req in executed_tools for req in required_tools):
                satisfied_checks += 1
            else:
                violations.append({
                    "step": step_num,
                    "tool": tool,
                    "type": "missing_prerequisite",
                    "detail": f"{tool} 需要先执行 {required_tools} 中的至少一个",
                })

        # 检查参数级别依赖：关键参数的值是否出现在之前的 observation 中
        if tool in PARAM_DEPENDENCIES:
            step_input = _parse_input(step.get("input", "{}"))
            all_obs_text = " ".join(str(o) for o in observations)

            for param, source_tool in PARAM_DEPENDENCIES[tool].items():
                if param in step_input:
                    param_value = str(step_input[param])
                    # 简单检查：参数值是否出现在之前的 observation 中
                    if param_value and param_value not in all_obs_text:
                        # 参数值不在之前的输出中，可能是凭空捏造的
                        violations.append({
                            "step": step_num,
                            "tool": tool,
                            "type": "param_not_grounded",
                            "detail": f"参数 {param}={param_value} 未在之前步骤的输出中找到",
                        })

        executed_tools.append(tool)
        observations.append(step.get("observation", ""))

    if dependency_checks == 0:
        score = 1.0  # 没有依赖要求的步骤
    else:
        score = satisfied_checks / dependency_checks

    return {
        "score": round(score, 4),
        "violations": violations,
        "dependency_checks": dependency_checks,
        "satisfied_checks": satisfied_checks,
    }


class PlanningRationalityMetric:
    """规划合理性评估指标。"""

    name = "规划合理性 (Planning Rationality)"

    def __init__(self, weights: tuple[float, float, float] = (0.3, 0.3, 0.4)):
        """初始化。

        Args:
            weights: 三个子指标的权重 (冗余, 完整性, 连贯性)。
        """
        self.w_redundancy, self.w_completeness, self.w_coherence = weights

    def evaluate_sample(self, sample: dict) -> dict[str, Any]:
        """评估单个样本的规划合理性。"""
        actual_steps = sample.get("steps", [])
        expected_steps = sample.get("expected_steps", [])

        redundancy = compute_redundancy_score(actual_steps)
        completeness = compute_completeness_score(actual_steps, expected_steps)
        coherence = compute_coherence_score(actual_steps)

        # 综合分数
        composite = (
            self.w_redundancy * redundancy["score"]
            + self.w_completeness * completeness["score"]
            + self.w_coherence * coherence["score"]
        )

        return {
            "task_id": sample["task_id"],
            "redundancy": redundancy,
            "completeness": completeness,
            "coherence": coherence,
            "composite_score": round(composite, 4),
        }

    def evaluate(self, dataset: list[dict]) -> dict[str, Any]:
        """对整个数据集进行评估。"""
        results = []
        for sample in dataset:
            results.append(self.evaluate_sample(sample))

        composites = [r["composite_score"] for r in results]
        redundancies = [r["redundancy"]["score"] for r in results]
        completeness_scores = [r["completeness"]["score"] for r in results]
        coherences = [r["coherence"]["score"] for r in results]

        summary = {
            "avg_composite": round(sum(composites) / len(composites), 4),
            "avg_redundancy": round(sum(redundancies) / len(redundancies), 4),
            "avg_completeness": round(sum(completeness_scores) / len(completeness_scores), 4),
            "avg_coherence": round(sum(coherences) / len(coherences), 4),
            "perfect_count": sum(1 for c in composites if c == 1.0),
            "violation_samples": sum(1 for r in results if r["coherence"]["violations"]),
        }

        return {
            "metric": self.name,
            "per_sample": results,
            "summary": summary,
        }
