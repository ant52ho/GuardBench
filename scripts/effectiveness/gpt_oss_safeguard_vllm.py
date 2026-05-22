"""Benchmark GPT-OSS-Safeguard-20B via OpenAI-compatible vLLM (Harmony chat).

Docs:
  https://huggingface.co/openai/gpt-oss-safeguard-20b
  https://github.com/openai/gpt-oss-safeguard
  https://cookbook.openai.com/articles/gpt-oss-safeguard-guide
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from guardbench import benchmark
from guardbench.moderators import GptOssSafeguardModerator


def discover_first_model_id(base_url: str, timeout_s: float = 10.0) -> str:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach models endpoint {url!r}: {e}") from e
    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"No models returned from {url!r}: {payload!r}")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError(f"Unexpected models payload: {payload!r}")
    return str(model_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="GuardBench + GPT-OSS-Safeguard (OpenAI API)")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL including /v1",
    )
    parser.add_argument("--model", default=None, help="Server model id (default: GET /v1/models)")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        help="Embedded in default HarmBench policy (gpt-oss-safeguard guide)",
    )
    parser.add_argument(
        "--policy-file",
        default=None,
        help="Optional UTF-8 file overriding the built-in HarmBench-aligned system prompt",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["harmbench_behaviors"],
        help="Dataset aliases",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out-dir", default="results", type=str)
    parser.add_argument("--metrics", nargs="+", default=["f1", "recall"])
    parser.add_argument(
        "--save-raw-outputs",
        action="store_true",
        help="Write <model_name>.generations.jsonl next to score JSON",
    )
    parser.add_argument("--max-examples", type=int, default=None, metavar="N")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Same as --max-examples 8",
    )
    args = parser.parse_args()

    max_examples = args.max_examples
    if args.smoke_test and max_examples is None:
        max_examples = 8

    model = args.model or discover_first_model_id(args.base_url)
    print(f"Using model id: {model!r}", file=sys.stderr)

    moderator = GptOssSafeguardModerator.from_openai_compatible(
        base_url=args.base_url,
        model=model,
        api_key=args.api_key,
        model_name="GPT-OSS-Safeguard-20B",
        reasoning_effort=args.reasoning_effort,
        policy_system_prompt_file=args.policy_file,
    )

    benchmark(
        moderate=moderator,
        model_name=moderator.model_name,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        datasets=args.datasets,
        metrics=args.metrics,
        pass_categories=True,
        max_examples=max_examples,
        save_raw_outputs=args.save_raw_outputs,
    )


if __name__ == "__main__":
    main()
