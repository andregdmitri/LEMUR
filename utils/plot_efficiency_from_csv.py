import argparse
import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. SETUP AND UPDATED DATA DICTIONARIES
# ==========================================
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "results_best_ckpt.csv"
DEFAULT_OUTPUT = ROOT / "images" / "efficiency_by_dataset"

MODEL_NAMES = {
    "mobilenet": "MobileNet",
    "efficientnet": "EfficientNet",
    "vmamba": "VMamba",
    "dvmamba": "LEMUR",
    "unet": "UNet",
    "retfound_finetune": "RETFound",
    "tinyvit": "TinyVit",
    "dtinyvit": "TinyVit-D",
}

GFLOPS = {
    "MobileNet": 0.111,
    "EfficientNet": 0.769,
    "VMamba": 0.476,
    "LEMUR": 0.476,
    "UNet": 6.561,
    "RETFound": 119.292,
    "TinyVit": 2.510,
    "TinyVit-D": 2.510,
}

PARAMS_M = {
    "MobileNet": 1.522981,
    "EfficientNet": 4.013953,
    "VMamba": 5.018949,
    "LEMUR": 5.018949,
    "UNet": 4.846821,
    "RETFound": 303.331882,
    "TinyVit": 5.073,
    "TinyVit-D": 5.073,
}

TEXT_POSITIONS = {
    "MobileNet": "bottom right",
    "EfficientNet": "top left",
    "VMamba": "bottom center",
    "LEMUR": "top center",
    "UNet": "top center",
    "RETFound": "bottom center",
    "TinyVit": "top center",
    "TinyVit-D": "bottom center",
}

# ==========================================
# 2. DATA PROCESSING PIPELINE
# ==========================================
def _parse_metrics(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}

def load_median_results(csv_path):
    print(f"[*] Loading data from {csv_path}...")
    frame = pd.read_csv(csv_path, dtype={"seed": "string"})
    frame["model"] = frame["mode"].map(MODEL_NAMES)
    frame = frame[frame["model"].notna()].copy()
    frame["val_auroc"] = frame["val_metrics"].map(
        lambda value: _parse_metrics(value).get("val/auroc")
    )
    frame["val_auroc"] = pd.to_numeric(frame["val_auroc"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["dataset", "seed", "val_auroc"])

    # Keep the latest run per seed
    frame = frame.sort_values("timestamp").drop_duplicates(
        subset=["dataset", "model", "seed"], keep="last"
    )

    medians = (
        frame.groupby(["dataset", "model"], as_index=False)
        .agg(median_auroc=("val_auroc", "median"), runs=("val_auroc", "size"))
    )
    medians["GFLOPs"] = medians["model"].map(GFLOPS)
    medians["Parameters (M)"] = medians["model"].map(PARAMS_M)
    medians["TextPosition"] = medians["model"].map(TEXT_POSITIONS)
    return medians


# ==========================================
# 3. PLOTLY GENERATION (STYLED)
# ==========================================
def make_dataset_plot(dataset, frame, output_dir):
    data = frame[frame["dataset"] == dataset].copy()
    if data.empty:
        print(f"[skip] {dataset}: no classification AUROC values")
        return None

    print(f"\n--- Processing Dataset: {dataset.upper()} ---")
    data["Label"] = data.apply(lambda row: f"{row['model']}", axis=1)
    
    # Dynamically determine the Y-axis range so the plot doesn't look squished
    min_auroc = data["median_auroc"].min()
    # y_min = 0
    y_min = max(0.0, math.floor(min_auroc * 10) / 10.0 - 0.1) # Drops one decimal step down
    y_max = 1.0

    fig = px.scatter(
        data,
        x="GFLOPs",
        y="median_auroc",
        size="Parameters (M)",
        color="model",
        text="Label",
        log_x=True,
        size_max=70,  # Matched to the new styling script
        labels={
            "GFLOPs": "Compute (GFLOPs, log scale)",
            "median_auroc": "Median validation AUROC",
            "model": "Model",
            "Parameters (M)": "Parameters (millions)",
        },
        title=f"Model efficiency on {dataset.upper()}",
    )

    # 1. Bubble & Text Styling
    fig.update_traces(
        textfont=dict(size=11, color="black"),
        marker=dict(line=dict(width=1.5, color="DarkSlateGrey")),
        cliponaxis=False,
    )
    # Apply individual specific text positions
    for trace in fig.data:
        trace.textposition = TEXT_POSITIONS.get(trace.name, "top center")

    # 2. Background Box for Custom Artefact Legend
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.72, y0=0.02, x1=0.98, y1=0.28,
        fillcolor="rgba(255, 255, 255, 0.85)",
        line=dict(color="black", width=1),
        layer="below"
    )

    # 3. Title for the Custom Legend
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.85, y=0.24,
        text="<b>Size (Parameters)</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(size=12, color="black")
    )

    # 4. Dummy traces for 1M, 5M, and 300M sizes plotted exactly to scale
    # Dynamically position them ~12% up the Y-axis so they stay in the box across different datasets
    legend_y = y_min + 0.12 * (y_max - y_min)
    max_params = data["Parameters (M)"].max()
    legend_sizes = [1, 5, 300]
    
    # Calculates sizes perfectly matching Plotly's px.scatter logic with size_max=70
    marker_diameters = [70 * math.sqrt(v / max_params) for v in legend_sizes]

    fig.add_trace(go.Scatter(
        x=[27, 45, 100],  # Raw X-values visually mapped onto the log scale box width
        y=[legend_y, legend_y, legend_y], 
        mode="markers+text",
        marker=dict(
            size=marker_diameters,
            color="rgba(200, 200, 200, 0.4)", 
            line=dict(width=1.5, color="DarkSlateGrey")
        ),
        text=["1M", "5M", "300M"],
        textposition="bottom center",
        textfont=dict(size=11, color="black"),
        showlegend=False, 
        hoverinfo="skip"
    ))

    # 5. Layout and Grids
    fig.update_layout(
        plot_bgcolor="white",
        width=1000,
        height=650,
        margin=dict(l=60, r=60, t=60, b=60),
        showlegend=False,
        xaxis=dict(
            type="log",
            gridcolor="lightgrey",
            showgrid=True,
            zeroline=False,
            range=[-1, 2.2], 
            tickvals=[0.1, 1, 10, 100],
            ticktext=["0.1", "1", "10", "100"]
        ),
        yaxis=dict(
            gridcolor="lightgrey",
            showgrid=True,
            zeroline=False,
            range=[y_min, y_max],
            dtick=0.1
        )
    )

    # 6. Save Formats
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"efficiency_{dataset}.pdf"
    # html_path = output_dir / f"efficiency_{dataset}.html"
    
    fig.write_image(pdf_path, scale=2)
    # fig.write_html(html_path)
    print(f"[✓] Saved PDF: {pdf_path}")
    # print(f"[✓] Saved HTML: {html_path}")
    
    for row in data.itertuples():
        if row.runs < 5:
            print(f"[note] {row.model}: median uses {row.runs} run(s), not 5")
            
    return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot median validation AUROC versus model complexity by dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    medians = load_median_results(args.input)
    if medians.empty:
        raise SystemExit(f"No classifier validation AUROC values found in {args.input}")

    print("\n--- Final Gathered Data (Medians) ---")
    print(medians.sort_values(["dataset", "model"]).to_string(index=False))
    print("--------------------------------------\n")

    for dataset in sorted(medians["dataset"].unique()):
        make_dataset_plot(dataset, medians, args.output_dir)


if __name__ == "__main__":
    main()