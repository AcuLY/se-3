"""人员C：任务完成度（Task Completion）评估指标。

参考 Ragas AgentGoalAccuracyWithReference，使用 LLM-as-Judge 方式
评估 Agent 的最终回答是否满足用户的原始请求。
"""

import sys
import os
from typing import Any

# 添加父目录到路径以支持直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import chat, extract_json
from data_loader import get_steps_summary


EVALUATION_PROMPT = """你是一个专业的 AI Agent 评估专家。请判断以下外卖 Agent 是否成功完成了用户的任务。

## 用户请求
{user_query}

## 期望结果
{ground_truth}

## Agent 执行步骤
{steps_summary}

## Agent 最终回答
{final_answer}

## 评估要求
请综合考虑以下几个方面：
1. Agent 的最终回答是否满足了用户的核心需求
2. 关键信息是否完整（如订单号、价格、餐厅名等）
3. 执行过程中是否出现了明显的错误或遗漏

请严格以 JSON 格式输出，不要包含其他内容：
{{"verdict": 0或1, "reason": "你的判断理由"}}

其中 verdict=1 表示任务成功完成，verdict=0 表示任务未完成或存在明显问题。"""


def evaluate_single_sample(sample: dict) -> dict[str, Any]:
    """使用 LLM 评估单个样本的任务完成度。

    Args:
        sample: 评估样本。

    Returns:
        包含 verdict 和 reason 的评估结果。
    """
    steps_summary = get_steps_summary(sample.get("steps", []))

    prompt = EVALUATION_PROMPT.format(
        user_query=sample["user_query"],
        ground_truth=sample["ground_truth"],
        steps_summary=steps_summary if steps_summary else "（无执行步骤）",
        final_answer=sample.get("final_answer", "（无最终回答）"),
    )

    try:
        response = chat(prompt)
        result = extract_json(response)
        verdict = int(result.get("verdict", 0))
        reason = result.get("reason", "无理由")
    except Exception as e:
        verdict = -1  # 标记为评估失败
        reason = f"LLM 评估失败: {e}"

    return {
        "task_id": sample["task_id"],
        "verdict": verdict,
        "reason": reason,
    }


class TaskCompletionMetric:
    """任务完成度评估指标（LLM-as-Judge）。"""

    name = "任务完成度 (Task Completion)"

    def evaluate(self, dataset: list[dict]) -> dict[str, Any]:
        """对整个数据集进行评估。

        Args:
            dataset: 评估样本列表。

        Returns:
            包含每个样本的判定和汇总统计的结果字典。
        """
        results = []
        for i, sample in enumerate(dataset):
            print(f"  [{i + 1}/{len(dataset)}] 评估样本 {sample['task_id']}...")
            result = evaluate_single_sample(sample)
            results.append(result)
            verdict_str = "✓ 完成" if result["verdict"] == 1 else ("✗ 未完成" if result["verdict"] == 0 else "⚠ 评估失败")
            print(f"    {verdict_str}: {result['reason'][:80]}")

        # 汇总
        valid_results = [r for r in results if r["verdict"] >= 0]
        completed = sum(1 for r in valid_results if r["verdict"] == 1)
        failed_eval = sum(1 for r in results if r["verdict"] < 0)

        summary = {
            "completion_rate": round(completed / len(valid_results), 4) if valid_results else 0.0,
            "completed_count": completed,
            "not_completed_count": len(valid_results) - completed,
            "eval_failed_count": failed_eval,
            "total_samples": len(dataset),
        }

        return {
            "metric": self.name,
            "per_sample": results,
            "summary": summary,
        }
