# Generate `response` fields for AdvancedIF inputs using any
# OpenAI-compatible chat-completions endpoint (OpenAI, local servers,
# Gemini's OpenAI-compat endpoint, OpenRouter, vLLM, etc.).
#
# For each input row it reads `conversation_history`, sends those messages
# to the target model, and writes the model's reply back into the row as
# `response` (JSON string of [{"role": "assistant", "content": ...}]) so the
# output is ready to feed straight into `cli.py evaluate`.

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import click
import openai

logger = logging.getLogger(__name__)


def _parse_messages(conversation_history: Any) -> list[dict[str, str]]:
    """Parse `conversation_history` (JSON string or list) into chat messages.

    Roles (system/user/assistant) are preserved as-is, so a leading
    role="system" message for `if_system_steerability_oss` tasks is passed
    through as the system prompt automatically.
    """
    if isinstance(conversation_history, str):
        conversation_history = json.loads(conversation_history)
    if not isinstance(conversation_history, list):
        raise ValueError(
            f"conversation_history must be a list, got {type(conversation_history)}"
        )
    return [
        {"role": m["role"], "content": m["content"]} for m in conversation_history
    ]


def _generate_one(
    client: openai.OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
) -> str:
    """Call the chat endpoint once and return the assistant text."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        resp = client.chat.completions.create(**kwargs)
    except openai.BadRequestError:
        # Some OpenAI-compatible servers only accept the older `max_tokens`.
        kwargs.pop("max_completion_tokens", None)
        kwargs["max_tokens"] = max_completion_tokens
        resp = client.chat.completions.create(**kwargs)

    content = resp.choices[0].message.content
    if content is None:
        raise ValueError("Model returned empty content")
    return content


def _process_row(
    client: openai.OpenAI,
    model: str,
    row: dict[str, Any],
    max_completion_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Generate a response for one row, returning the row with `response` set."""
    try:
        messages = _parse_messages(row["conversation_history"])
        content = _generate_one(
            client, model, messages, max_completion_tokens, temperature,
            reasoning_effort,
        )
        # Match the format the judge expects: a JSON string holding a
        # single-element list with the final assistant turn.
        row["response"] = json.dumps([{"role": "assistant", "content": content}])
        # Drop any stale judge output carried over from a previous model.
        row.pop("judge_result", None)
        row["_generation_error"] = None
    except Exception as e:  # noqa: BLE001 - per-row fault tolerance
        logger.error("Failed to generate response: %s", e)
        row["response"] = json.dumps([{"role": "assistant", "content": ""}])
        row["_generation_error"] = str(e)
    return row


@click.command()
@click.option("--input", "-i", "input_file", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Input JSONL with conversation_history/prompt_metadata/benchmark_name")
@click.option("--output", "-o", "output_file", required=True,
              type=click.Path(path_type=Path), help="Output JSONL with `response` filled in")
@click.option("--base-url", "base_url", required=True,
              help="OpenAI-compatible base URL (e.g. https://api.openai.com/v1)")
@click.option("--api-key", "-k", "api_key", envvar="OPENAI_API_KEY", required=True,
              help="API key (or set OPENAI_API_KEY)")
@click.option("--model", "-m", required=True, help="Model name to generate with")
@click.option("--max_completion_tokens", default=32768, type=int,
              help="Max tokens for the generated response")
@click.option("--temperature", default=None, type=float,
              help="Sampling temperature (omit for reasoning models that reject it)")
@click.option("--reasoning-effort", "reasoning_effort", default=None,
              type=click.Choice(["minimal", "low", "medium", "high"]),
              help="Reasoning effort for reasoning models (omit if unsupported)")
@click.option("--max_concurrency", default=10, type=int, help="Concurrent requests")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]))
def generate(
    input_file: Path,
    output_file: Path,
    base_url: str,
    api_key: str,
    model: str,
    max_completion_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    max_concurrency: int,
    log_level: str,
) -> None:
    """Generate `response` fields for AdvancedIF inputs via an OpenAI-compatible API."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    rows = [
        json.loads(line)
        for line in input_file.read_text().splitlines()
        if line.strip()
    ]
    logger.info("Loaded %d rows from %s", len(rows), input_file)
    logger.info("Generating with model=%s base_url=%s", model, base_url)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        results = list(
            executor.map(
                lambda r: _process_row(
                    client, model, r, max_completion_tokens, temperature,
                    reasoning_effort,
                ),
                rows,
            )
        )

    with output_file.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    failed = sum(1 for r in results if r.get("_generation_error"))
    logger.info(
        "Done. Wrote %d rows to %s (%d succeeded, %d failed)",
        len(results), output_file, len(results) - failed, failed,
    )


if __name__ == "__main__":
    generate()
