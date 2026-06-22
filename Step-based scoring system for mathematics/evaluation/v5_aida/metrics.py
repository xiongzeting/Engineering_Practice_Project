"""v4 OCR metrics: CER, BLEU-4, exact-match, CER<=0.1 rate.

Aggressive LaTeX normalization before comparison — both GT and OCR output go
through normalize_latex() so we measure recognition quality, not formatting
differences.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "per_sample"

# LaTeX rendering commands / noise to strip before comparison.
NOISE_PATTERNS = [
    (re.compile(r"\\displaystyle\b"), ""),
    (re.compile(r"\\limits\b"), ""),
    (re.compile(r"\\left\b"), ""),
    (re.compile(r"\\right\b"), ""),
    (re.compile(r"\\!"), ""),
    (re.compile(r"\\,"), ""),
    (re.compile(r"\\;"), ""),
    (re.compile(r"\\>"), ""),
    (re.compile(r"\\dfrac\b"), "\\frac"),
    (re.compile(r"\\tfrac\b"), "\\frac"),
    (re.compile(r"\\cdot\b"), "*"),
    (re.compile(r"\\times\b"), "*"),
    (re.compile(r"\\div\b"), "/"),
    (re.compile(r"\\quad\b"), " "),
    (re.compile(r"\\qquad\b"), " "),
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),  # \text{abc} -> abc
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\mathbb\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\,|\\;|\\!|\\ "), " "),
]

# Tokens that should not appear in clean LaTeX output (OCR prompt forbids them).
UNICODE_MATH_RE = re.compile(
    r"[∫∑∏√×÷±≤≥≠≈→⇔⇒∞πθαβγΔμλΩ₀-₉⁰-⁹½⅓¼]"
)


def normalize_latex(s: str) -> str:
    if not s:
        return ""
    # NFKC normalize (turns fancy unicode into ascii where possible).
    s = unicodedata.normalize("NFKC", s)
    # Strip $...$ and $$...$$ wrappers.
    s = re.sub(r"\$\$([^$]*)\$\$", r"\1", s)
    s = s.replace("$", "")
    # Apply noise patterns.
    for pat, repl in NOISE_PATTERNS:
        s = pat.sub(repl, s)
    # Lowercase, collapse whitespace.
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    # Strip trailing punctuation noise.
    s = s.strip().rstrip(".,;:")
    return s


def edit_distance_tokens(a: list[str], b: list[str]) -> int:
    """Levenshtein on token lists — robust to LaTeX spacing differences
    (e.g., '\\omega _ { i j }' vs '\\omega_{ij}' tokenize to the same sequence)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def token_cer(pred: str, gt: str) -> float:
    """Token-level CER = token-edit-distance / len(gt_tokens)."""
    pt = tokenize(pred)
    gt_t = tokenize(gt)
    if not gt_t:
        return 0.0
    return edit_distance_tokens(pt, gt_t) / len(gt_t)


def edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein on characters."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|[^a-zA-Z\s]|\w+")


def tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


def bleu4(pred: str, gt: str) -> float:
    """Sentence-level BLEU-4 with smoothing."""
    pred_tokens = tokenize(pred)
    gt_tokens = tokenize(gt)
    if not pred_tokens or not gt_tokens:
        return 0.0

    def ngram_counts(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    precisions: list[float] = []
    for n in (1, 2, 3, 4):
        pred_ngrams = ngram_counts(pred_tokens, n)
        gt_ngrams = ngram_counts(gt_tokens, n)
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum((pred_ngrams & gt_ngrams).values())
        # +1 smoothing for n>1
        if overlap == 0 and n > 1:
            precisions.append(1.0 / max(sum(pred_ngrams.values()), 1))
        else:
            precisions.append(overlap / max(sum(pred_ngrams.values()), 1))

    # Brevity penalty.
    bp = 1.0 if len(pred_tokens) >= len(gt_tokens) else \
        math.exp(1 - len(gt_tokens) / max(len(pred_tokens), 1))

    if min(precisions) == 0:
        # geometric mean of zero is zero; use tiny smoothing
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / 4
    return bp * math.exp(log_avg)


def cer_bucket(cer: float) -> str:
    if cer == 0:
        return "perfect (CER=0)"
    if cer <= 0.1:
        return "near (CER<=0.1)"
    if cer <= 0.3:
        return "ok (CER<=0.3)"
    return "bad (CER>0.3)"


def aggregate() -> dict:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"no per_sample results in {RESULTS_DIR}")

    rows: list[dict] = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    n = len(rows)
    n_failed = sum(1 for r in rows if not r.get("ok"))

    cer_list: list[float] = []
    bleu_list: list[float] = []
    em_count = 0
    cer_le_01_count = 0
    bucket_counts: dict[str, int] = {
        "perfect (CER=0)": 0,
        "near (CER<=0.1)": 0,
        "ok (CER<=0.3)": 0,
        "bad (CER>0.3)": 0,
        "ocr_failed": 0,
    }
    case_details: list[dict] = []

    for r in rows:
        gt_norm = normalize_latex(r.get("ground_truth", ""))
        if not r.get("ok"):
            bucket_counts["ocr_failed"] += 1
            case_details.append({
                "id": r["id"],
                "gt": r.get("ground_truth", ""),
                "pred": "",
                "cer": 1.0,
                "error": r.get("error", "unknown"),
            })
            cer_list.append(1.0)
            bleu_list.append(0.0)
            continue

        pred_norm = normalize_latex(r.get("ocr_text", ""))
        if not gt_norm:
            cer = 0.0
        else:
            # Token-level CER: robust to LaTeX spacing differences between
            # IM2LATEX's "every token spaced" style and MiMo's compact output.
            cer = token_cer(pred_norm, gt_norm)
        bleu = bleu4(pred_norm, gt_norm)

        cer_list.append(cer)
        bleu_list.append(bleu)
        if cer == 0:
            em_count += 1
        if cer <= 0.1:
            cer_le_01_count += 1
        bucket_counts[cer_bucket(cer)] += 1

        case_details.append({
            "id": r["id"],
            "gt": r.get("ground_truth", ""),
            "gt_norm": gt_norm,
            "pred": r.get("ocr_text", ""),
            "pred_norm": pred_norm,
            "cer": round(cer, 4),
            "bleu": round(bleu, 4),
        })

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    summary = {
        "n": n,
        "n_failed": n_failed,
        "failure_rate": round(n_failed / n if n else 0.0, 4),
        "cer_mean": round(avg(cer_list), 4),
        "cer_median": round(sorted(cer_list)[len(cer_list) // 2], 4),
        "bleu_mean": round(avg(bleu_list), 4),
        "exact_match_rate": round(em_count / n if n else 0.0, 4),
        "cer_le_01_rate": round(cer_le_01_count / n if n else 0.0, 4),
        "buckets": bucket_counts,
        "cases": case_details,
    }
    return summary


def main() -> None:
    m = aggregate()
    cases = m.pop("cases")
    out_path = Path(__file__).parent / "results" / "summary.json"
    # Save without cases for compactness; cases live in per_sample files.
    out_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")
    print(f"(per-sample cases: {len(cases)})")


if __name__ == "__main__":
    main()
