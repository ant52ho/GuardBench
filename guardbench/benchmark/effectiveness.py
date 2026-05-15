from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
from unified_io import create_path, write_json, write_jsonl

from ..datasets import DATASETS, load_dataset
from ..evaluate import evaluate

_GUARDBENCH_COLLECT_RAW_KW = "_guardbench_collect_raw"

metric_mapping = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
    "auprc": "AUPRC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "g_mean": "G-Mean",
    "fpr": "FPR",
    "fnr": "FNR",
}


def format_score(score: float) -> str:
    return "{:.3f}".format(round(score, 3))


def get_results_table(headers, results) -> None:
    return tabulate(results, headers=headers, tablefmt="github", disable_numparse=True)


def collate_fn(batch):
    id = [x["id"] for x in batch]
    label = [x["label"] for x in batch]
    conversation = [x["conversation"] for x in batch]
    return id, label, conversation


def _call_moderate_with_optional_raw(
    moderate: callable,
    *,
    conversations: list,
    moderate_kwargs: dict,
    collect_raw: bool,
) -> tuple[list[float], list | None]:
    """Invoke ``moderate``; optionally collect parallel raw outputs.

    Supported mechanisms:

    * Pass ``_guardbench_collect_raw=True`` (stripped by :class:`~guardbench.moderators.base.BaseModerator`
      before :meth:`~guardbench.moderators.base.BaseModerator.infer`).
    * Or return ``(scores, raw_outputs)`` from ``moderate`` when ``collect_raw``.

    If the callable rejects the extra keyword (plain ``def moderate(...)``), retries without it;
    raw texts are then ``None`` unless a tuple is returned.
    """
    if not collect_raw:
        out = moderate(conversations=conversations, **moderate_kwargs)
        if isinstance(out, tuple):
            raise ValueError(
                "moderate returned a tuple while save_raw_outputs=False; return only scores."
            )
        return out, None

    mk = dict(moderate_kwargs)
    mk[_GUARDBENCH_COLLECT_RAW_KW] = True
    try:
        out = moderate(conversations=conversations, **mk)
    except TypeError:
        logger.warning(
            "moderate() does not accept '_guardbench_collect_raw'; "
            "retrying without it (no raw text unless moderate returns a tuple)."
        )
        out = moderate(conversations=conversations, **moderate_kwargs)

    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError(
                "moderate must return (scores: list[float], raw_outputs: list) when using a tuple."
            )
        scores, raws = out
        if raws is not None and len(raws) != len(scores):
            raise ValueError(
                "moderate tuple raw_outputs length must match scores length."
            )
        raws_out = list(raws) if raws is not None else [None] * len(scores)
        return list(scores), raws_out

    scores = out
    return list(scores), [None] * len(scores)


def benchmark(
    moderate: callable,
    model_name: str = "moderator",
    out_dir: str = "results",
    batch_size: int = 1,
    datasets: list = "all",
    metrics: list = None,
    pass_categories: bool = False,
    max_examples: int | None = None,
    save_raw_outputs: bool = False,
    **kwargs,
) -> None:
    """Benchmark the effectiveness of the provided moderation function/model.     Additional keyword arguments are passed to the provided moderation function. For example, you can pass the tokenizer and the model to the moderation function. Check the official repository for examples and tutorials.

    Args:
        moderate (callable): Moderation function. It must have at least one parameter named `conversations`.
        model_name (str, optional): "Name of the moderation model". Defaults to "moderator".
        out_dir (str, optional): Directory for the moderation outputs. Defaults to "results".
        batch_size (int, optional): Batch size. Defaults to 32.
        metrics (list, optional): Metrics used to evaluate results. If None, defaults to `["f1", "recall"]`. Available metrics are: `precision`, `recall`, `f1`, `mcc`, `auprc`, `sensitivity`, `specificity`, `g_mean`, `fpr`, `fnr`, `tn`, `fp`, `fn`, `tp`.
        datasets (list, optional): Datasets selected for evaluation. Defaults to "all".
        pass_categories (bool, optional): If True, each call also passes `hazard_categories` (dataset-level list from the loader) and `sample_categories` (per-row `category` field, or None when absent). Your `moderate` must accept these keyword arguments (e.g. via **kwargs).
        max_examples (int, optional): If set, only run inference on this many examples **per dataset** (metrics reflect the truncated run).
        save_raw_outputs (bool, optional): If True, write ``{model_name}.generations.jsonl`` next to
            score JSON with one object per example: ``id``, ``unsafe_probability``, ``raw_output``
            (stringified model output before parsing when available).
    """

    # Set datasets if "all"  ---------------------------------------------------
    if datasets == "all":
        datasets = list(DATASETS)

    # Set metrics if None  -----------------------------------------------------
    if isinstance(metrics, str):
        metrics = [metrics]
    if metrics is None:
        metrics = ["f1", "recall"]

    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be >= 1 when set.")

    for m in metrics:
        if m not in metric_mapping:
            raise ValueError(
                f"Metric `{m}` is not supported. Supported metrics are: `precision`, `recall`, `f1`, `mcc`, `auprc`, `sensitivity`, `specificity`, `g_mean`, `fpr`, `fnr`, `tn`, `fp`, `fn`, `tp`."
            )

    # Benchmarking Effectiveness -----------------------------------------------
    logger.start("Benchmarking Effectiveness")
    if max_examples is not None:
        logger.info(f"Capped at {max_examples} example(s) per dataset (smoke mode).")
    if save_raw_outputs:
        logger.info("Saving raw model outputs to *.generations.jsonl (per dataset).")
    results = []
    for i, dataset_name in enumerate(datasets):
        # Dataset --------------------------------------------------------------
        dataset = load_dataset(dataset_name)
        batch_generator = dataset.generate_batches(batch_size)

        # Inference ------------------------------------------------------------
        idx = str(i + 1).zfill(2) if len(datasets) > 9 else i + 1
        desc = f"{idx}/{len(datasets)} - {dataset.name}"
        tqdm_kwargs = dict(desc=desc, dynamic_ncols=True, mininterval=1.0)

        ids, y_true, y_pred_prob = [], [], []
        generation_rows: list[dict] = []

        hazard_categories = list(getattr(dataset, "hazard_categories", None) or [])

        processed = 0
        for batch in tqdm(list(batch_generator), **tqdm_kwargs):
            if max_examples is not None and processed >= max_examples:
                break
            if max_examples is not None:
                room = max_examples - processed
                if room <= 0:
                    break
                if len(batch) > room:
                    batch = batch[:room]

            batch_ids, batch_y_true, batch_conversations = collate_fn(batch)
            ids += batch_ids
            y_true += batch_y_true
            moderate_kwargs = dict(kwargs)
            if pass_categories:
                moderate_kwargs["hazard_categories"] = hazard_categories
                moderate_kwargs["sample_categories"] = [
                    x.get("category") for x in batch
                ]
            batch_scores, batch_raws = _call_moderate_with_optional_raw(
                moderate,
                conversations=batch_conversations,
                moderate_kwargs=moderate_kwargs,
                collect_raw=save_raw_outputs,
            )
            y_pred_prob += batch_scores
            if save_raw_outputs:
                for sid, score, raw in zip(batch_ids, batch_scores, batch_raws):
                    generation_rows.append(
                        {
                            "id": sid,
                            "unsafe_probability": score,
                            "raw_output": ""
                            if raw is None
                            else (raw if isinstance(raw, str) else repr(raw)),
                        }
                    )
            processed += len(batch_ids)

        y_true = dict(zip(ids, y_true))
        y_pred_prob = dict(zip(ids, y_pred_prob))

        # Save predictions -----------------------------------------------------
        pred_path = create_path(out_dir)
        ds_out = pred_path / dataset_name
        ds_out.mkdir(parents=True, exist_ok=True)
        write_json(y_pred_prob, ds_out / f"{model_name}.json")
        if save_raw_outputs and generation_rows:
            write_jsonl(generation_rows, ds_out / f"{model_name}.generations.jsonl")

        # Evaluate -------------------------------------------------------------
        report = evaluate(y_true, y_pred_prob)
        results.append([dataset.name] + [format_score(report[m]) for m in metrics])

    headers = ["Dataset"] + [metric_mapping[m] for m in metrics]
    results_table = get_results_table(headers, results)
    logger.info(f"Results:\n{results_table}")

    logger.success("Done")
