"""评估主入口脚本。

运行所有评估指标并输出结果报告。

用法：
    python eval/run_eval.py [--dataset PATH] [--output PATH] [--skip-llm]

环境变量：
    LLM_BACKEND: LLM 后端 (openai | cli | ollama)，默认 cli
    LLM_MODEL: 模型名称
    LLM_API_KEY: API Key (openai 后端)
    LLM_BASE_URL: API Base URL (openai 后端)
    LLM_CLI_COMMAND: CLI 命令 (cli 后端)，默认 "claude"
"""

import argparse
import json
import os
import sys
from datetime import datetime

# 确保能找到 eval 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_eval_dataset
from metrics.tool_call_accuracy import ToolCallAccuracyMetric
from metrics.planning_rationality import PlanningRationalityMetric
from metrics.task_completion import TaskCompletionMetric


def print_separator(title: str = "") -> None:
    """打印分隔线。"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print("-" * 60)


def print_tool_call_report(result: dict) -> None:
    """打印工具调用准确度报告。"""
    summary = result["summary"]
    print(f"\n  平均准确度 (Accuracy): {summary['avg_accuracy']:.4f}")
    print(f"  平均 F1 分数:          {summary['avg_f1']:.4f}")
    print(f"  完美匹配样本数:        {summary['perfect_count']}/{len(result['per_sample'])}")
    print(f"  零分样本数:            {summary['zero_count']}/{len(result['per_sample'])}")
    print()

    # 显示每个样本的分数
    print("  详细结果：")
    print(f"  {'Task ID':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-' * 60}")
    for r in result["per_sample"]:
        print(f"  {r['task_id']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")


def print_planning_report(result: dict) -> None:
    """打印规划合理性报告。"""
    summary = result["summary"]
    print(f"\n  综合平均分:    {summary['avg_composite']:.4f}")
    print(f"  平均冗余分:    {summary['avg_redundancy']:.4f}")
    print(f"  平均完整性分:  {summary['avg_completeness']:.4f}")
    print(f"  平均连贯性分:  {summary['avg_coherence']:.4f}")
    print(f"  完美样本数:    {summary['perfect_count']}/{len(result['per_sample'])}")
    print(f"  有违规样本数:  {summary['violation_samples']}/{len(result['per_sample'])}")
    print()

    # 显示每个样本的分数
    print("  详细结果：")
    print(f"  {'Task ID':<20} {'Composite':>10} {'Redundancy':>10} {'Complete':>10} {'Coherence':>10}")
    print(f"  {'-' * 60}")
    for r in result["per_sample"]:
        print(f"  {r['task_id']:<20} {r['composite_score']:>10.4f} {r['redundancy']['score']:>10.4f} {r['completeness']['score']:>10.4f} {r['coherence']['score']:>10.4f}")

    # 显示有违规的样本详情
    violation_samples = [r for r in result["per_sample"] if r["coherence"]["violations"]]
    if violation_samples:
        print("\n  逻辑违规详情：")
        for r in violation_samples:
            print(f"    样本 {r['task_id']}:")
            for v in r["coherence"]["violations"]:
                print(f"      - 步骤{v['step']}: [{v['type']}] {v['detail']}")


def print_task_completion_report(result: dict) -> None:
    """打印任务完成度报告。"""
    summary = result["summary"]
    print(f"\n  任务完成率:    {summary['completion_rate']:.2%}")
    print(f"  完成数:        {summary['completed_count']}/{summary['total_samples']}")
    print(f"  未完成数:      {summary['not_completed_count']}/{summary['total_samples']}")
    if summary["eval_failed_count"] > 0:
        print(f"  评估失败数:    {summary['eval_failed_count']}/{summary['total_samples']}")
    print()

    print("  详细结果：")
    print(f"  {'Task ID':<20} {'Verdict':>8}  Reason")
    print(f"  {'-' * 70}")
    for r in result["per_sample"]:
        v = "✓" if r["verdict"] == 1 else ("✗" if r["verdict"] == 0 else "⚠")
        reason = r["reason"][:50] + "..." if len(r["reason"]) > 50 else r["reason"]
        print(f"  {r['task_id']:<20} {v:>8}  {reason}")


def main():
    parser = argparse.ArgumentParser(description="Agent 评估实验")
    parser.add_argument("--dataset", type=str, default=None, help="评估数据集路径")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径 (JSON)")
    parser.add_argument("--skip-llm", action="store_true", help="跳过需要 LLM 的指标（任务完成度）")
    args = parser.parse_args()

    # 加载数据集
    print_separator("加载评估数据集")
    dataset = load_eval_dataset(args.dataset)

    all_results = {}

    # 指标1：工具调用准确度
    print_separator("指标1: 工具调用准确度 (Tool Call Accuracy)")
    metric1 = ToolCallAccuracyMetric()
    result1 = metric1.evaluate(dataset)
    print_tool_call_report(result1)
    all_results["tool_call_accuracy"] = result1

    # 指标2：规划合理性
    print_separator("指标2: 规划合理性 (Planning Rationality)")
    metric2 = PlanningRationalityMetric()
    result2 = metric2.evaluate(dataset)
    print_planning_report(result2)
    all_results["planning_rationality"] = result2

    # 指标3：任务完成度（需要 LLM）
    if not args.skip_llm:
        print_separator("指标3: 任务完成度 (Task Completion) - LLM-as-Judge")
        metric3 = TaskCompletionMetric()
        result3 = metric3.evaluate(dataset)
        print_task_completion_report(result3)
        all_results["task_completion"] = result3
    else:
        print_separator("指标3: 任务完成度 (跳过 - 使用 --skip-llm)")

    # 输出总结
    print_separator("评估总结")
    print(f"  数据集样本数: {len(dataset)}")
    print(f"  工具调用准确度: {result1['summary']['avg_accuracy']:.4f} (Accuracy), {result1['summary']['avg_f1']:.4f} (F1)")
    print(f"  规划合理性:     {result2['summary']['avg_composite']:.4f} (Composite)")
    if not args.skip_llm:
        print(f"  任务完成度:     {result3['summary']['completion_rate']:.2%}")
    print()

    # 保存 JSON 结果
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    output_path = os.path.abspath(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
