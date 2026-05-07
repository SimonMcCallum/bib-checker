"""Local Natural Language Inference (NLI) scorer for claim-inversion detection.

Embedding similarity scores topical match. It cannot tell you whether the cited
paper's findings *agree* or *contradict* the claim being made — the canonical
example is Makransky et al. 2021 ("VR increases liking but **not** learning")
being cited as evidence of learning gains. Embedding distance stays small;
the polarity is flipped.

Cross-encoder NLI models score a (premise, hypothesis) pair into
{entailment, neutral, contradiction}. They run on CPU, are small (~80–500MB),
and are specifically trained for exactly this task — no LLM needed.

Default model: `cross-encoder/nli-deberta-v3-base` (~440MB on disk, ~100ms per
pair on a modern CPU). For tighter resource budgets, swap in
`cross-encoder/nli-MiniLM2-L6-H768` (~80MB) via the model arg.

Premise = cited paper's abstract / tldr / sentence
Hypothesis = the claim made around the citation marker

If contradiction probability is high, the citation is potentially using the
work to support a claim its own findings refute. The score reported is
`contradiction - entailment` so positive = warning, negative = consistent.
"""

from __future__ import annotations
from functools import lru_cache
import numpy as np


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
LIGHT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

# Label index varies by model; both models above use this order:
LABELS = ("contradiction", "entailment", "neutral")


@lru_cache(maxsize=2)
def _load(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for the NLI scorer. "
            "Install with: pip install bib-checker[nli]"
        ) from e
    return CrossEncoder(model_name)


def score_pairs(
    pairs: list[tuple[str, str]],
    model_name: str = DEFAULT_NLI_MODEL,
) -> list[dict]:
    """Run NLI on (premise, hypothesis) pairs.

    Returns one dict per pair: {entailment, neutral, contradiction, polarity}
    where polarity = contradiction_prob - entailment_prob (positive = warning).
    """
    if not pairs:
        return []
    model = _load(model_name)
    raw = model.predict(pairs, apply_softmax=True, show_progress_bar=False)
    arr = np.asarray(raw)
    out = []
    for row in arr:
        d = {LABELS[i]: float(row[i]) for i in range(len(LABELS))}
        d["polarity"] = d["contradiction"] - d["entailment"]
        out.append(d)
    return out


def score_citation_polarity(
    contexts: dict,
    entries: dict,
    model_name: str = DEFAULT_NLI_MODEL,
) -> dict:
    """For each cited entry, find the most-contradicting reference sentence.

    contexts: {key: [context_string, ...]} from the citation extractor
    entries:  {key: {field: value, ...}} bib entries with abstract/tldr

    Returns {key: {best_polarity, best_premise, best_label, scores}}.
    A high best_polarity (e.g. > 0.3) means the cited paper's most contradicting
    sentence appears to refute the claim — a polarity-flip warning.
    """
    from .alignment import _build_ref_snippets, _clean_text  # local import to avoid cycle

    pair_jobs: list[tuple[str, str]] = []
    job_index: list[tuple[str, str, str]] = []  # (key, premise, hypothesis)
    for key, ctx_list in contexts.items():
        entry = entries.get(key, {})
        snippets, _ = _build_ref_snippets(entry)
        if not snippets:
            continue
        hypothesis = _clean_text(" ".join(ctx_list))[:1000]
        for snippet in snippets:
            premise = _clean_text(snippet)[:1000]
            pair_jobs.append((premise, hypothesis))
            job_index.append((key, premise, hypothesis))

    if not pair_jobs:
        return {}

    raw = score_pairs(pair_jobs, model_name=model_name)

    out: dict[str, dict] = {}
    for (key, premise, _), result in zip(job_index, raw):
        prev = out.get(key)
        if prev is None or result["polarity"] > prev["best_polarity"]:
            out[key] = {
                "best_polarity": result["polarity"],
                "best_premise": premise,
                "best_label": max(LABELS, key=lambda L: result[L]),
                "scores": result,
            }
    return out
