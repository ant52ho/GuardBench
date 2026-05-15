"""Run GSPR via an OpenAI-compatible vLLM server (HTTP Chat Completions).

Example::

    pip install 'guardbench[openai]'
    python scripts/effectiveness/gspr_vllm.py \\
        --base-url http://127.0.0.1:19801/v1 \\
        --datasets harmbench_behaviors

If ``--model`` is omitted, the first model id from ``GET /v1/models`` is used.

Smoke test (few examples per dataset, via ``benchmark(..., max_examples=…)``)::

    python scripts/effectiveness/gspr_vllm.py --smoke-test \\
        --base-url http://127.0.0.1:19801/v1 \\
        --datasets harmbench_behaviors
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from guardbench import benchmark
from guardbench.moderators import GSPRModerator


def discover_first_model_id(base_url: str, timeout_s: float = 10.0) -> str:
    """Return the first ``id`` from the server's ``/v1/models`` listing."""
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
    parser = argparse.ArgumentParser(description="GuardBench + GSPR (vLLM OpenAI API)")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:19801/v1",
        help="OpenAI-compatible base URL including /v1",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id on the server (default: first entry from /v1/models)",
    )
    parser.add_argument("--api-key", default="EMPTY", help="API key header value")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["harmbench_behaviors"],
        help="Dataset aliases (default: harmbench_behaviors)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out-dir", default="results", type=str)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["f1", "recall"],
        help="GuardBench metric keys",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        metavar="N",
        help="Only run N examples per dataset (metrics are approximate)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Same as --max-examples 8 (quick sanity check)",
    )
    parser.add_argument(
        "--save-raw-outputs",
        action="store_true",
        help="Write <model_name>.generations.jsonl next to score JSON (raw completions)",
    )
    args = parser.parse_args()

    max_examples = args.max_examples
    if args.smoke_test and max_examples is None:
        max_examples = 8

    model = args.model or discover_first_model_id(args.base_url)
    print(f"Using model id: {model!r}", file=sys.stderr)

    moderator = GSPRModerator.from_openai_compatible(
        base_url=args.base_url,
        model=model,
        api_key=args.api_key,
        model_name="GSPR",
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
