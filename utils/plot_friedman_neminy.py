#!/usr/bin/env python
"""Friedman + Nemenyi critical-difference (CD) diagram for the LEMUR results.

The statistical ranking and the CD diagram are produced with
`labicompare <https://github.com/jose-gilberto/labicompare>`_.

Installation
------------
The published package metadata pins ``python = "^3.11"`` so pip refuses to
install it on the 3.10 ``mae`` environment.  The library is pure Python and
runs fine on 3.10, so bypass the guard::

    git clone https://github.com/jose-gilberto/labicompare.git /tmp/labicompare
    pip install --no-deps --ignore-requires-python /tmp/labicompare

Usage
-----
::

    # default: best-checkpoint table, val/f1, datasets as blocks, Nemenyi CD
    python utils/plot_friedman_neminy.py

    # treat every (dataset, seed) pair as a block (N=20 instead of N=4)
    python utils/plot_friedman_neminy.py --blocks dataset_seed

    # use the library's Wilcoxon-Holm post-hoc instead of Nemenyi
    python utils/plot_friedman_neminy.py --posthoc holm

    # compare the last-epoch table instead of the best-checkpoint table
    python utils/plot_friedman_neminy.py --csv results/results_wandb.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import studentized_range, wilcoxon  # noqa: E402

try:
    from labicompare.core.data import EvaluationData
    from labicompare.core.results import ComparisonSummary, PairwiseResult
    from labicompare.plots.ranking import plot_cd_diagram
    from labicompare.stats.friedman import friedman_test
    from labicompare.stats.posthoc import wilcoxon_holm
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "labicompare is required but not installed.\n\n"
        "  git clone https://github.com/jose-gilberto/labicompare.git /tmp/labicompare\n"
        "  pip install --no-deps --ignore-requires-python /tmp/labicompare\n\n"
        f"(original error: {exc})"
    ) from exc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "results_best_ckpt.csv"
DEFAULT_OUTPUT_DIR = ROOT / "images" / "friedman_neminy"

# Display names already used by plot_auroc_runs.py / plot_efficiency_from_csv.py
MODE_TO_MODEL = {
    "mobilenet": "MobileNet",
    "efficientnet": "EfficientNet",
    "vmamba": "VMamba",
    "dvmamba": "LEMUR",
    "unet": "UNet",
    "retfound_finetune": "RETFound",
    "tinyvit": "TinyViT",
    "dtinyvit": "TinyViT-D",
}
MODEL_ORDER = [
    "MobileNet",
    "EfficientNet",
    "VMamba",
    "LEMUR",
    "UNet",
    "RETFound",
    "TinyViT",
    "TinyViT-D",
]

# The proposed method, highlighted in the CD diagram by default.
DEFAULT_HIGHLIGHT = ["LEMUR"]

POSTHOC_LABELS = {"nemenyi": "Nemenyi", "holm": "Wilcoxon-Holm"}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _parse_metric(raw: object, metric: str) -> float:
    """Pull ``metric`` out of a serialised ``val_metrics`` / ``train_metrics`` blob."""
    try:
        return float(json.loads(raw)[metric])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def load_matrix(
    csv_path: Path,
    metric: str,
    blocks: str,
    agg: str,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Return a wide ``rows x models`` score matrix ready for ``EvaluationData``.

    Each row is one statistical block (a dataset, or a dataset+seed pair) and
    each column is one model.
    """
    df = pd.read_csv(csv_path)

    missing_cols = {"mode", "dataset", "seed"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_cols)}")

    if metric in df.columns:
        df = df.assign(score=pd.to_numeric(df[metric], errors="coerce"))
        source = f"column '{metric}'"
    elif "val_metrics" in df.columns:
        df = df.assign(score=df["val_metrics"].apply(lambda raw: _parse_metric(raw, metric)))
        source = f"val_metrics['{metric}']"
    else:
        raise ValueError(
            f"Metric '{metric}' is neither a column nor present in 'val_metrics' "
            f"of {csv_path}."
        )

    if seeds:
        df = df[df["seed"].isin(seeds)]
        if df.empty:
            raise ValueError(f"No rows left after filtering to seeds {seeds}.")
        source += f" (seeds={seeds})"

    index = ["dataset", "seed"] if blocks == "dataset_seed" else ["dataset"]
    matrix = df.pivot_table(index=index, columns="mode", values="score", aggfunc=agg)

    # Friedman / Wilcoxon need a complete, balanced design.
    dropped_rows = matrix.index[matrix.isna().any(axis=1)].tolist()
    if dropped_rows:
        print(f"[!] Dropping incomplete blocks (missing models): {dropped_rows}")
        matrix = matrix.dropna(axis=0)
    dropped_cols = matrix.columns[matrix.isna().any(axis=0)].tolist()
    if dropped_cols:
        print(f"[!] Dropping incomplete models (missing blocks): {dropped_cols}")
        matrix = matrix.dropna(axis=1)

    if matrix.shape[1] < 3:
        raise ValueError(
            f"Friedman test needs >= 3 models, got {matrix.shape[1]}. "
            "Check --metric / --seeds / the input CSV."
        )
    if matrix.shape[0] < 2:
        raise ValueError(
            f"Friedman test needs >= 2 blocks, got {matrix.shape[0]}. "
            "Try --blocks dataset_seed to use every (dataset, seed) pair."
        )

    matrix = matrix.rename(columns=MODE_TO_MODEL)
    order = [m for m in MODEL_ORDER if m in matrix.columns]
    matrix = matrix[order + [c for c in matrix.columns if c not in order]]
    matrix.index.names = index

    print(f"[*] Loaded scores for metric {source}  ->  {matrix.shape[0]} blocks x {matrix.shape[1]} models")
    return matrix


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def nemenyi_cd(n_models: int, n_blocks: int, alpha: float) -> float:
    """Two-tailed Nemenyi critical difference (Demsar, 2006).

    ``CD = q_alpha * sqrt(k (k + 1) / (6 N))`` where ``q_alpha`` is the
    Studentized-range statistic with infinite degrees of freedom divided by
    ``sqrt(2)``, which matches the published Nemenyi tables.
    """
    q_alpha = float(studentized_range.ppf(1.0 - alpha, n_models, np.inf)) / np.sqrt(2.0)
    return q_alpha * float(np.sqrt(n_models * (n_models + 1) / (6.0 * n_blocks)))


def _model_means(data: EvaluationData) -> dict[str, float]:
    return {name: float(mean) for name, mean in zip(data.model_names, data.scores.mean(axis=0))}


def _raw_pairs(data: EvaluationData) -> list[PairwiseResult]:
    """Uncorrected pairwise Wilcoxon results (used by the Nemenyi path)."""
    scores = data.scores
    models = data.model_names
    pairs: list[PairwiseResult] = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_a, model_b = models[i], models[j]
            mean_diff = float(np.mean(scores[:, i] - scores[:, j]))

            try:
                p_value = float(
                    wilcoxon(scores[:, i], scores[:, j], zero_method="pratt").pvalue
                )
            except ValueError:
                # All paired differences are zero -> identical distributions.
                p_value = 1.0

            winner = None
            if mean_diff != 0.0:
                if data.higher_is_better:
                    winner = model_a if mean_diff > 0 else model_b
                else:
                    winner = model_a if mean_diff < 0 else model_b

            pairs.append(
                PairwiseResult(
                    model_a=model_a,
                    model_b=model_b,
                    p_value=p_value,
                    is_significant=False,
                    winner=winner,
                    mean_diff=mean_diff,
                )
            )
    return pairs


def nemenyi_summary(data: EvaluationData, alpha: float) -> tuple[ComparisonSummary, float]:
    """Friedman omnibus test followed by the Nemenyi post-hoc test."""
    f_stat, f_p = friedman_test(data)
    avg_ranks = data.ranks_df.mean()
    cd = nemenyi_cd(len(data.model_names), len(data.dataset_names), alpha)

    pairs = _raw_pairs(data)
    for pair in pairs:
        pair.is_significant = (
            abs(float(avg_ranks[pair.model_a]) - float(avg_ranks[pair.model_b])) > cd
        )

    summary = ComparisonSummary(
        friedman_stat=f_stat,
        friedman_p_value=f_p,
        is_global_sig=f_p <= alpha,
        pairwise_results=pairs,
        model_means=_model_means(data),
        alpha=alpha,
        higher_is_better=data.higher_is_better,
        n_samples=len(data.dataset_names),
    )
    return summary, cd


def holm_summary(data: EvaluationData, alpha: float) -> tuple[ComparisonSummary, None]:
    """Friedman omnibus test followed by Wilcoxon-Holm (labicompare built-in)."""
    try:
        return wilcoxon_holm(data, alpha=alpha), None
    except ValueError as exc:
        # labicompare refuses to build a summary when the omnibus test is not
        # significant.  That is not an error for us: it just means no pair can
        # be declared different, so we plot every model as a single tie group.
        f_stat, f_p = friedman_test(data)
        print(f"[!] {exc}")
        print("[!] Friedman H0 was not rejected -> all models will be drawn as tied.")
        summary = ComparisonSummary(
            friedman_stat=f_stat,
            friedman_p_value=f_p,
            is_global_sig=False,
            pairwise_results=_raw_pairs(data),
            model_means=_model_means(data),
            alpha=alpha,
            higher_is_better=data.higher_is_better,
            n_samples=len(data.dataset_names),
        )
        return summary, None


# --------------------------------------------------------------------------- #
# Reporting / plotting
# --------------------------------------------------------------------------- #
def report(data: EvaluationData, summary: ComparisonSummary, cd: float | None, alpha: float) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 50)

    avg_ranks = data.ranks_df.mean().sort_values()
    ranks_tbl = pd.DataFrame({"Average Rank": avg_ranks.round(3)})

    h0 = "REJECTED" if summary.is_global_sig else "NOT REJECTED"
    print(
        f"\n[*] Friedman omnibus test ({len(data.model_names)} models, "
        f"{summary.n_samples} blocks): chi2 = {summary.friedman_stat:.4f}, "
        f"p = {summary.friedman_p_value:.6g}  ->  H0 {h0} at alpha = {alpha}"
    )
    if cd is not None:
        print(f"[*] Nemenyi critical difference (alpha = {alpha}): CD = {cd:.4f}")

    print("\n[*] Average ranks (1 = best):")
    print(ranks_tbl.to_string())

    print("\n[*] Leaderboard by mean score:")
    print(summary.get_leaderboard().round(4).to_string())

    print("\n[*] Pairwise tests (sorted by p-value):")
    print(summary.to_dataframe().round(4).to_string(index=False))


def make_plot(
    data: EvaluationData,
    summary: ComparisonSummary,
    title: str,
    highlight: list[str] | None,
    figsize: tuple[float, float],
) -> plt.Figure:
    present = [m for m in (highlight or []) if m in data.model_names]
    unknown = [m for m in (highlight or []) if m not in data.model_names]
    if unknown:
        print(f"[!] Ignoring highlight models absent from the data: {unknown}")

    return plot_cd_diagram(
        data=data,
        summary=summary,
        title=title,
        figsize=figsize,
        highlight_models=present or None,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Friedman + Nemenyi critical-difference diagram for LEMUR results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RESULTS, help="Results CSV to read.")
    parser.add_argument("--metric", default="val/f1", help="Metric column or key inside val_metrics.")
    parser.add_argument(
        "--blocks",
        choices=["dataset", "dataset_seed"],
        default="dataset",
        help="Statistical block definition (rows of the evaluation matrix).",
    )
    parser.add_argument("--agg", choices=["mean", "median"], default="mean", help="Seed aggregation.")
    parser.add_argument(
        "--posthoc",
        choices=["nemenyi", "holm"],
        default="nemenyi",
        help="Post-hoc test defining the CD diagram cliques.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "--highlight",
        nargs="*",
        default=DEFAULT_HIGHLIGHT,
        help="Display names to highlight (default: the proposed model).",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Restrict to these seeds (default: all seeds found).",
    )
    parser.add_argument("--title", default=None, help="Override the figure title.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated figure(s) and summary CSV.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(12.0, 5.0),
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    dataframe = load_matrix(
        csv_path=args.csv,
        metric=args.metric,
        blocks=args.blocks,
        agg=args.agg,
        seeds=args.seeds,
    )

    data = EvaluationData(dataframe, higher_is_better=True)
    print(f"[*] {data!r}")

    if args.posthoc == "nemenyi":
        summary, cd = nemenyi_summary(data, args.alpha)
    else:
        summary, cd = holm_summary(data, args.alpha)

    report(data, summary, cd, args.alpha)

    blocks_label = "datasets" if args.blocks == "dataset" else "dataset+seed"
    title = args.title or (
        f"Critical Difference Diagram - Friedman + {POSTHOC_LABELS[args.posthoc]} "
        f"({args.metric}, {summary.n_samples} {blocks_label}, alpha={args.alpha:g})"
    )

    fig = make_plot(
        data=data,
        summary=summary,
        title=title,
        highlight=args.highlight,
        figsize=tuple(args.figsize),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "cd_diagram.pdf"
    csv_path = out_dir / "cd_diagram_summary.csv"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    summary.to_dataframe().to_csv(csv_path, index=False)
    plt.close(fig)

    print(f"\n[\u2713] Figure saved to {png_path}")
    print(f"[\u2713] Pairwise summary saved to {csv_path}")


if __name__ == "__main__":
    main()
