"""Run Qwen3Guard-Gen via OpenAI-compatible vLLM (HF README pattern).

See https://huggingface.co/Qwen/Qwen3Guard-Gen-4B/blob/main/README.md

Example::

    pip install 'guardbench[openai]'
    python scripts/effectiveness/qwen3_guard_vllm.py \\
        --base-url http://127.0.0.1:8000/v1 \\
        --mode user_query \\
        --datasets harmbench_behaviors
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from guardbench import benchmark
from guardbench.moderators import Qwen3GuardModerator


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
    parser = argparse.ArgumentParser(description="GuardBench + Qwen3Guard (OpenAI API)")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL including /v1",
    )
    parser.add_argument("--model", default=None, help="Server model id (default: /v1/models)")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--mode",
        choices=("user_query", "assistant_response"),
        default="user_query",
        help="Prompt moderation vs last-assistant response moderation (HF README)",
    )
    parser.add_argument(
        "--controversial-as",
        choices=("unknown", "safe", "unsafe"),
        default="unknown",
        help="Map Controversial severity to unsafe probability policy",
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
    args = parser.parse_args()

    model = args.model or discover_first_model_id(args.base_url)
    print(f"Using model id: {model!r}", file=sys.stderr)

    moderator = Qwen3GuardModerator.from_openai_compatible(
        base_url=args.base_url,
        model=model,
        api_key=args.api_key,
        model_name="Qwen3Guard",
        mode=args.mode,
        controversial_as=args.controversial_as,
    )

    benchmark(
        moderate=moderator,
        model_name=moderator.model_name,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        datasets=args.datasets,
        metrics=args.metrics,
        pass_categories=False,
        save_raw_outputs=args.save_raw_outputs,
    )


if __name__ == "__main__":
    main()
