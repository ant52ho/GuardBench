"""Benchmark Meta Llama Guard 3 8B via OpenAI-compatible HTTP (vLLM / TGI).

Model card (gated on Hugging Face):
  https://huggingface.co/meta-llama/Llama-Guard-3-8B

Use the special-token single-user prompt shape from the reference tokenizer template.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from guardbench import benchmark
from guardbench.moderators import LlamaGuard3Moderator


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
    parser = argparse.ArgumentParser(description="GuardBench + Llama Guard 3 (OpenAI API)")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible base URL including /v1",
    )
    parser.add_argument("--model", default=None, help="Server model id")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--assessment-role",
        choices=("User", "Agent"),
        default="User",
        help="User = prompt moderation; Agent = moderate last assistant turn (needs assistant msg)",
    )
    parser.add_argument("--placeholder-context", default="(No prior user context.)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["harmbench_behaviors"],
        help="Dataset aliases",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out-dir", default="results", type=str)
    parser.add_argument("--metrics", nargs="+", default=["f1", "recall"])
    parser.add_argument("--save-raw-outputs", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None, metavar="N")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    max_examples = args.max_examples
    if args.smoke_test and max_examples is None:
        max_examples = 8

    model = args.model or discover_first_model_id(args.base_url)
    print(f"Using model id: {model!r}", file=sys.stderr)

    moderator = LlamaGuard3Moderator.from_openai_compatible(
        base_url=args.base_url,
        model=model,
        api_key=args.api_key,
        model_name="LlamaGuard3-8B",
        assessment_role=args.assessment_role,
        placeholder_user_context=args.placeholder_context,
    )

    benchmark(
        moderate=moderator,
        model_name=moderator.model_name,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        datasets=args.datasets,
        metrics=args.metrics,
        pass_categories=False,
        max_examples=max_examples,
        save_raw_outputs=args.save_raw_outputs,
    )


if __name__ == "__main__":
    main()
