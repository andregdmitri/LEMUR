import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math

# ==========================================
# 1. ADD YOUR ACTUAL AUROC SCORES HERE
# ==========================================
# AUROC
AUROC_SCORES = {
    "MobileNet": 0.87213,
    "EfficientNet": 0.8765,
    "VMamba": 0.85692,
    "VMamba_DIST": 0.88052,
    "UNet": 0.93717,
    "RETFound": 0.931068,
    "TinyVit": 0.7639
}

# GFLOPs
GFLOPS = {
    "MobileNet": 0.111,
    "EfficientNet": 0.769,
    "VMamba": 0.476,
    "VMamba_DIST": 0.476,
    "UNet": 6.561,
    "RETFound": 119.292,
    "TinyVit": 2.510
}

# Milhoes
PARAMS = {
    "MobileNet": 1.522981,
    "EfficientNet": 4.013953,
    "VMamba": 5.018949,
    "VMamba_DIST": 5.018949,
    "UNet": 4.846821,
    "RETFound": 303.331882,
    "TinyVit": 5.073
}

# Custom text placements to prevent overlapping
TEXT_POSITIONS = {
    "MobileNet": "top center",
    "EfficientNet": "middle right",
    "VMamba": "bottom center",
    "VMamba_DIST": "top left",
    "UNet": "top center",
    "RETFound": "bottom center"
}

def main():
    results = []
    
    print("Gathering data from dictionaries...")
    for model_name in AUROC_SCORES.keys():
        params_m = PARAMS.get(model_name, 0.0)
        flops = GFLOPS.get(model_name, 0.0)
        auroc = AUROC_SCORES.get(model_name, 0.0)
        
        results.append({
            "Model": model_name,
            "Number of parameters": round(params_m, 2),
            "GFLOPs": flops,
            "AUROC": auroc,
            "TextPosition": TEXT_POSITIONS.get(model_name, "top center")
        })
        
        print(f"[*] {model_name} -> Params: {params_m:.2f}M | Compute: {flops:.3f} GFLOPs | AUROC: {auroc:.5f}")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("[!] No data gathered.")
        return

    print("\n--- Final Gathered Data ---")
    print(df)
    print("---------------------------\n")

    # ==========================================
    # 2. PLOTLY BUBBLE CHART
    # ==========================================
    print("[*] Generating Plotly chart...")
    fig = px.scatter(
        df,
        x="GFLOPs",
        y="AUROC",
        size="Number of parameters",
        color="Model",
        text="Model",
        log_x=True,
        title="AUROC vs Complexity on IDRID",
        labels={
            "GFLOPs": "GFLOPs (log scale)",
            "AUROC": "Median AUROC"
        },
        size_max=70  
    )

    # Styling improvements and individual text positioning
    fig.update_traces(
        textposition=df['TextPosition'].tolist(), # Applies custom positions
        textfont=dict(size=11, color='black'),
        marker=dict(line=dict(width=1.5, color='DarkSlateGrey')),
        cliponaxis=False # Prevents bubbles from being cut off at the grid borders
    )

    # ---------------------------------------------------------
    # CUSTOM ARTEFACT "LEGEND" HANDLING (BOTTOM RIGHT)
    # ---------------------------------------------------------
    
    # 1. Background Box for the artefact legend
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.72, y0=0.02, x1=0.98, y1=0.28,
        fillcolor="rgba(255, 255, 255, 0.85)",
        line=dict(color="black", width=1),
        layer="below"
    )

    # 2. Title for the artefact legend
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.85, y=0.24,
        text="<b>Size (Parameters)</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(size=12, color="black")
    )

    # 3. Dummy traces for 1M, 5M, and 300M sizes plotted exactly to scale
    # We calculate exact diameters relative to the real data
    max_params = df["Number of parameters"].max()
    legend_sizes = [1, 5, 300]
    marker_diameters = [70 * math.sqrt(v / max_params) for v in legend_sizes]
    
    # We plot them at specific data coordinates that fall inside the box we drew above
    fig.add_trace(go.Scatter(
        x=[41, 80, 170],      # X data coordinates corresponding to paper positions inside the box
        y=[0.828, 0.828, 0.828], # Y data coordinates corresponding to paper positions inside the box
        mode='markers+text',
        marker=dict(
            size=marker_diameters,
            color='rgba(200, 200, 200, 0.4)', # Neutral semi-transparent color
            line=dict(width=1.5, color='DarkSlateGrey')
        ),
        text=["1M", "5M", "300M"],
        textposition="bottom center",
        textfont=dict(size=11, color="black"),
        showlegend=False, # We are building an artefact, so turn off the actual Plotly legend entry
        hoverinfo='skip'
    ))
    # ---------------------------------------------------------

    fig.update_layout(
        xaxis=dict(
            gridcolor='lightgrey', 
            showgrid=True, 
            zeroline=False,
            # Explicit log bounds: log10(0.1) = -1 to log10(300+) = ~2.5
            range=[-1.2, 2.5] 
        ),
        yaxis=dict(
            gridcolor='lightgrey', 
            showgrid=True, 
            zeroline=False,
            range=[0.8, 1.0], 
            dtick=0.1
        ),
        plot_bgcolor='white',
        margin=dict(l=60, r=60, t=60, b=60), 
        showlegend=False # Disable standard legend entirely since names are in bubbles and sizes are in the artefact
    )

    # Save as HTML
    output_html = "efficiency_bubble_chart.html"
    fig.write_html(output_html)
    print(f"[✓] HTML Plot saved to {output_html}.")
    
    # Save as PNG
    output_png = "efficiency_bubble_chart.png"
    try:
        # Requires 'kaleido' package
        fig.write_image(output_png, width=900, height=600, scale=2) 
        print(f"[✓] PNG Image saved to {output_png}.")
    except Exception as e:
        print(f"[!] Could not save PNG. Make sure 'kaleido' is installed (pip install kaleido). Error: {e}")

    # Show in browser
    print("[*] Opening in browser...")
    fig.show()

if __name__ == "__main__":
    main()