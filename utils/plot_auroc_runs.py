import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "results_best_ckpt.csv"
DEFAULT_OUTPUT = ROOT / "images" / "auroc_runs"

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
MODEL_COLORS = {
    "MobileNet": "#0072B2",
    "EfficientNet": "#E69F00",
    "VMamba": "#009E73",
    "LEMUR": "#CC79A7",
    "UNet": "#D55E00",
    "RETFound": "#56B4E9",
    "TinyViT": "#F0E442",
    "TinyViT-D": "#7A5195",
}


def parse_metrics(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_runs(csv_path):
    frame = pd.read_csv(csv_path, dtype={"seed": "string", "id": "string"})
    frame["model"] = frame["mode"].map(MODE_TO_MODEL)
    frame = frame[frame["model"].notna()].copy()
    frame["dataset"] = frame["dataset"].astype("string").str.lower()
    frame["seed"] = frame["seed"].str.strip()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["val_auroc"] = pd.to_numeric(
        frame["val_metrics"].map(lambda value: parse_metrics(value).get("val/auroc")),
        errors="coerce",
    )
    frame = frame.dropna(subset=["dataset", "seed", "val_auroc"])

    # If a seed was run more than once, retain its latest recorded result.
    frame = frame.sort_values("timestamp", na_position="first").drop_duplicates(
        ["dataset", "model", "seed"], keep="last"
    )
    return frame


def summarize_auroc(data):
    summary = (
        data.groupby("model", observed=True)["val_auroc"]
        .agg(mean="mean", std="std", n="count")
        .reindex([model for model in MODEL_ORDER if model in data["model"].unique()])
    )
    summary["std"] = summary["std"].fillna(0)
    return summary


def add_model_bars(fig, summary, data, *, row=None, col=None):
    models = summary.index.tolist()
    labels = models  # Use exact model names for cleaner axis matching the reference
    
    customdata = [
        [float(summary.loc[model, "std"]), int(summary.loc[model, "n"])]
        for model in models
    ]
    
    # Trace 1: The Main Bars
    bar_trace = go.Bar(
        x=labels,
        y=summary["mean"].tolist(),
        marker_color=[MODEL_COLORS[model] for model in models],
        error_y={
            "type": "data",
            "array": summary["std"].tolist(),
            "visible": True,
            "thickness": 1.5,
            "width": 6,
            "color": "black",
        },
        customdata=customdata,
        hovertemplate=(
            "%{x}<br>Mean AUROC: %{y:.4f}"
            "<br>SD: %{customdata[0]:.4f}"
            "<br>Runs: %{customdata[1]}<extra></extra>"
        ),
        showlegend=False,
    )

    # Trace 2: Individual Run Points (Scatter Overlay)
    scatter_x = []
    scatter_y = []
    for _, row_data in data.iterrows():
        if row_data["model"] in models:
            scatter_x.append(row_data["model"])
            scatter_y.append(row_data["val_auroc"])

    scatter_trace = go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode="markers",
        marker=dict(
            color="white",
            size=7,
            line=dict(color="black", width=1),
            opacity=0.9
        ),
        showlegend=False,
        hoverinfo="skip"
    )

    if row is None:
        fig.add_trace(bar_trace)
        fig.add_trace(scatter_trace)
        fig.update_xaxes(
            categoryorder="array", categoryarray=labels, tickangle=-25,
            showline=True, linewidth=1.5, linecolor="black"
        )
        fig.update_yaxes(
            title_text="AUROC", range=[0.5, 1.05], dtick=0.2, # Y-axis starts at 0.5 to match reference visual
            showline=True, linewidth=1.5, linecolor="black"
        )
    else:
        fig.add_trace(bar_trace, row=row, col=col)
        fig.add_trace(scatter_trace, row=row, col=col)
        fig.update_xaxes(
            categoryorder="array", categoryarray=labels, tickangle=-45, 
            showline=True, linewidth=1.5, linecolor="black", row=row, col=col
        )
        fig.update_yaxes(
            title_text="AUROC", range=[0.5, 1.05], dtick=0.2, 
            showline=True, linewidth=1.5, linecolor="black", row=row, col=col
        )


def make_dataset_chart(dataset, runs, output_dir):
    data = runs[runs["dataset"] == dataset].copy()
    if data.empty:
        print(f"[skip] {dataset}: no validation AUROC values")
        return

    summary = summarize_auroc(data)
    fig = go.Figure()
    add_model_bars(fig, summary, data) # Passed raw data here

    fig.update_layout(
        title={"text": f"Mean validation AUROC by model: {dataset.upper()}", "x": 0.04},
        template="plotly_white",
        width=1000,
        height=600,
        bargap=0.28,
        margin={"l": 75, "r": 35, "t": 95, "b": 125},
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 14, "color": "black"},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"auroc_mean_{dataset}.png"
    pdf_path = output_dir / f"auroc_mean_{dataset}.pdf"
    
    fig.write_image(png_path, width=1000, height=600, scale=2)
    fig.write_image(pdf_path, width=1000, height=600)
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")

    run_counts = data.groupby("model")["seed"].nunique().to_dict()
    incomplete = {model: count for model, count in run_counts.items() if count != 5}
    if incomplete:
        print(f"[note] {dataset} run counts differ from five: {incomplete}")


def make_combined_chart(runs, output_dir):
    datasets = sorted(runs["dataset"].unique())
    fig = make_subplots(
        rows=2,
        cols=2, # Or adapt based on len(datasets)
        subplot_titles=[dataset.upper() for dataset in datasets],
        horizontal_spacing=0.09,
        vertical_spacing=0.25,
    )
    for index, dataset in enumerate(datasets):
        row, col = divmod(index, 2)
        data = runs[runs["dataset"] == dataset]
        add_model_bars(fig, summarize_auroc(data), data, row=row + 1, col=col + 1)

    fig.update_layout(
        title={"text": "Mean validation AUROC by model and dataset", "x": 0.04},
        template="plotly_white",
        width=1600,
        height=1000,
        bargap=0.28,
        margin={"l": 75, "r": 35, "t": 105, "b": 120},
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 12, "color": "black"},
    )
    png_path = output_dir / "auroc_mean_all_datasets.png"
    pdf_path = output_dir / "auroc_mean_all_datasets.pdf"
    
    fig.write_image(png_path, width=1600, height=1000, scale=2)
    fig.write_image(pdf_path, width=1600, height=1000)
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create per-dataset validation-metric bars showing seed mean and standard deviation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    runs = load_runs(args.input)
    if runs.empty:
        raise SystemExit(f"No classifier validation metrics found in {args.input}")

    print(f"Loaded {len(runs)} unique model/dataset/seed results from {args.input}")
    for dataset in sorted(runs["dataset"].unique()):
        make_dataset_chart(dataset, runs, args.output_dir)
    make_combined_chart(runs, args.output_dir)


if __name__ == "__main__":
    main()