"""统一的 LLM 调用封装，支持多后端。

支持的后端：
- openai: OpenAI 兼容 API（glm-4-flash, hunyuan-lite 等）
- cli: 本地 CLI 工具（如 claude）
- ollama: 本地 Ollama 服务

通过环境变量配置：
- LLM_BACKEND: 后端类型 (openai | cli | ollama)，默认 cli
- LLM_MODEL: 模型名称
- LLM_API_KEY: API Key（openai 后端）
- LLM_BASE_URL: API Base URL（openai 后端）
- LLM_CLI_COMMAND: CLI 命令（cli 后端），默认 "claude"
- LLM_OLLAMA_URL: Ollama URL（ollama 后端），默认 "http://localhost:11434"
"""

import json
import os
import re
import subprocess
import time
from typing import Any

import requests


def _get_config() -> dict[str, str]:
    """从环境变量读取 LLM 配置。"""
    return {
        "backend": os.environ.get("LLM_BACKEND", "cli"),
        "model": os.environ.get("LLM_MODEL", ""),
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "base_url": os.environ.get("LLM_BASE_URL", ""),
        "cli_command": os.environ.get("LLM_CLI_COMMAND", "claude-internal"),
        "ollama_url": os.environ.get("LLM_OLLAMA_URL", "http://localhost:11434"),
    }


def _call_openai(prompt: str, config: dict[str, str]) -> str:
    """通过 OpenAI 兼容 API 调用 LLM。"""
    url = f"{config['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_cli(prompt: str, config: dict[str, str]) -> str:
    """通过本地 CLI 调用 LLM（如 claude）。"""
    cmd = config["cli_command"]
    result = subprocess.run(
        [cmd, "-p", prompt, "--output-format", "text"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CLI 调用失败 (code={result.returncode}): {result.stderr}")
    return result.stdout.strip()


def _call_ollama(prompt: str, config: dict[str, str]) -> str:
    """通过 Ollama 本地 API 调用 LLM。"""
    url = f"{config['ollama_url'].rstrip('/')}/api/generate"
    payload = {
        "model": config["model"],
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def chat(prompt: str, max_retries: int = 3) -> str:
    """统一的 LLM 调用接口。

    Args:
        prompt: 输入提示。
        max_retries: 最大重试次数。

    Returns:
        LLM 的文本回复。
    """
    config = _get_config()
    backend = config["backend"]

    call_fn = {
        "openai": _call_openai,
        "cli": _call_cli,
        "ollama": _call_ollama,
    }.get(backend)

    if call_fn is None:
        raise ValueError(f"不支持的 LLM 后端: {backend}，可选: openai, cli, ollama")

    last_error = None
    for attempt in range(max_retries):
        try:
            return call_fn(prompt, config)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}，{wait}s 后重试...")
                time.sleep(wait)

    raise RuntimeError(f"LLM 调用在 {max_retries} 次尝试后仍然失败: {last_error}")


def extract_json(text: str) -> dict[str, Any]:
    """从 LLM 回复文本中提取 JSON 对象。

    尝试多种策略：直接解析、正则提取代码块、正则提取花括号。

    Args:
        text: LLM 返回的文本。

    Returns:
        解析出的字典。
    """
    # 策略1：直接解析
    text_stripped = text.strip()
    try:
        return json.loads(text_stripped)
    except json.JSONDecodeError:
        pass

    # 策略2：提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 策略3：提取第一个 { ... } 块
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 回复中提取 JSON: {text[:200]}...")
