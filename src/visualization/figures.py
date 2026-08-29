"""
src/visualization/figures.py
============================
Publication-ready figures (figure-reviewer corrected).

Changes vs previous version:
  - Wilson 95% CI error bars on the zone charts (Fig 3)         [issue 3.1]
  - WHO reference line at 1.000 drawn + labelled                [issue 3.2]
  - MUAC panel uses a zoomed y-axis so 0.91-1.00 is readable    [issue 3.3]
  - Binary pass/fail colours (CVD-safe), no severity gradient   [issues 3.5/3.6]
  - Value labels at 3 decimals to match the manuscript          [issue 3.7]
  - Legend moved outside the plotting area                      [issue 3.8]
  - Fig 4 y-axis zero-based (no truncation), consistent w/ Fig3 [issues 4.1/3.4]
  - Fig 4 shading only where Sc3 < Sc1 (correct at crossover)   [issue 4.3]
  - Fig 4 legend upper-right (no data overlap); STEEL bug fixed [issue 4.4 + crash]
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ── Okabe-Ito (colour-vision-deficiency safe) palette ─────────────────────────
NAVY      = "#1F2A5A"
BLUE      = "#0072B2"   # Sc1 curve / generic blue
GREEN     = "#009E73"   # meets criterion (>= 0.90)
AMBER     = "#E69F00"   # below criterion (< 0.90)
VERMILION = "#D55E00"   # Sc3 curve
WHITE     = "#FAFAFA"
GRID      = "#EEEEEE"

TARGET = 0.90

plt.rcParams.update({
    "font.family":       "DejaVu Sans",

    "font.size":         12,

    # Figure titles
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",

    # Axis labels
    "axes.labelsize":    13,

    # Tick labels
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,

    # Legends
    "legend.fontsize":   11,

    "figure.dpi":        220,
    "savefig.dpi":       220,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def _save(name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(output_dir, f"{name}.{ext}"),
                    bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  -> results/figures/{name}.pdf/.png")


def _clean_ax():
    plt.gca().spines["left"].set_edgecolor("#CCCCCC")
    plt.gca().spines["bottom"].set_edgecolor("#CCCCCC")


def _plot_single_zone_chart(
    data: dict,
    short_labels: list,
    title: str,
    output_dir: str,
    filename: str,
    y_min: float = 0.0,
    who_ref: bool = True,
) -> None:
    """
    Bar chart of SAM Recall by zone, with:
      - Wilson 95% CI error bars (if data carries ci_low/ci_high),
      - binary pass/fail colours (GREEN >= 0.90, AMBER < 0.90),
      - a 0.90 target line and an optional WHO reference line at 1.000,
      - a legend placed below the axes.

    Expected per-zone dict:
        {"recall": float, "ci_low": float, "ci_high": float, ...}
    ci_low/ci_high are optional; error bars are skipped if absent.
    """
    plt.figure(figsize=(8, 6), facecolor=WHITE)
    plt.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.24)

    recalls = [v["recall"] for v in data.values()]
    colors  = [GREEN if r >= TARGET else AMBER for r in recalls]  # binary

    x = np.arange(len(short_labels))
    bars = plt.bar(x, recalls, color=colors, edgecolor=WHITE,
                   width=0.6, zorder=3)

    # Wilson CI error bars (only if provided)
    ci_low  = [v.get("ci_low")  for v in data.values()]
    ci_high = [v.get("ci_high") for v in data.values()]
    has_ci  = all(c is not None for c in ci_low + ci_high)
    if has_ci:
        err_lo = [max(0.0, r - cl) for r, cl in zip(recalls, ci_low)]
        err_hi = [max(0.0, ch - r) for r, ch in zip(recalls, ci_high)]
        plt.errorbar(x, recalls, yerr=[err_lo, err_hi], fmt="none",
                     ecolor="#333333", elinewidth=1.4, capsize=4,
                     capthick=1.4, zorder=5)

    # Target line (black dashed) and optional WHO reference (navy dash-dot)
    plt.axhline(TARGET, color="black", ls="--", lw=0.9, zorder=6)
    if who_ref:
        plt.axhline(1.0, color=NAVY, ls="-.", lw=0.9, alpha=0.9, zorder=6)

    plt.xticks(x, short_labels)
    plt.ylim(y_min, 1.04 if y_min > 0 else 1.15)
    if y_min > 0:
        plt.yticks([round(y_min + i * (1.0 - y_min) / 5, 2) for i in range(6)])
    else:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel("SAM Recall")
    plt.title(title, color=NAVY, pad=12)
    plt.grid(axis="y", color=GRID, zorder=0, lw=0.8)

    # Value labels (3 decimals; placed above the error-bar top, clear of 0.90)
    span = (1.04 if y_min > 0 else 1.15) - y_min
    for bar, r, ch in zip(bars, recalls, ci_high):
        top = max(r, ch) if ch is not None else r
        y_lbl = top + 0.012 * span / 1.15 + (0.02 if y_min == 0 else 0.004)
        if y_min == 0 and 0.86 < r < 0.94:   # avoid sitting on the 0.90 line
            y_lbl = 0.965
        plt.text(bar.get_x() + bar.get_width() / 2, y_lbl,
                 f"{r:.3f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color="#1a1a1a")

    _clean_ax()

    # Legend below the axes (no overlap with bars/labels)
    handles = [
        mpatches.Patch(color=GREEN, label=r"$\geq$ 0.90 (meets criterion)"),
        mpatches.Patch(color=AMBER, label="< 0.90 (below criterion)"),
        mlines.Line2D([], [], color="black", ls="--", lw=1.8,
                      label="Target = 0.90"),
    ]
    if who_ref:
        handles.append(mlines.Line2D([], [], color=NAVY, ls="-.", lw=1.6,
                                     label="WHO ideal = 1.000"))
    handles.append(mlines.Line2D([], [], color="#333333", lw=1.4,
                                 label="Wilson 95% CI"))
    plt.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False,
            #    fontsize=9.5
               )
    

    _save(filename, output_dir)


# ── Figure 3a: WHZ boundary (primary finding) ────────────────────────────────

def plot_whz_boundary(
    whz_data:   dict,
    output_dir: str = "results/figures",
    filename:   str = "fig1_whz_boundary",
) -> None:
    """WHZ zones (XGBoost, Sc0). Zero-based axis; full pass/fail range."""
    short = ["SAM\nzone", "Boundary", "Grey\nzone", "Normal\nzone"]
    _plot_single_zone_chart(
        whz_data, short,
        "WHZ diagnostic zones (XGBoost, Sc0)",
        output_dir, filename,
        y_min=0.0, who_ref=True,
    )


# ── Figure 3b: MUAC boundary (control) ───────────────────────────────────────

def plot_muac_boundary(
    muac_data:  dict,
    output_dir: str = "results/figures",
    filename:   str = "fig1b_muac_boundary",
) -> None:
    """MUAC zones (XGBoost, Sc0). Zoomed y-axis so 0.91-1.00 is perceptible."""
    short = ["SAM\nzone", "Boundary", "MAM\nzone", "Normal\nzone"]
    _plot_single_zone_chart(
        muac_data, short,
        "MUAC diagnostic zones (XGBoost, Sc0)",
        output_dir, filename,
        y_min=0.84, who_ref=True,
    )


# ── Figure 4: Degradation curves ─────────────────────────────────────────────

def plot_degradation_curves(
    rates:       list,
    sc1_recalls: list,
    sc3_recalls: list,
    field_rate:  float = 30.0,
    wilcoxon_p:  float = 0.001,
    output_dir:  str   = "results/figures",
    filename:    str   = "fig2_degradation",
) -> None:
    """SAM Recall vs missingness rate: Sc1 (MCAR) and Sc3 (MNAR bound)."""
    plt.figure(figsize=(8, 6), facecolor=WHITE)
    plt.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.24)

    sc1 = np.asarray(sc1_recalls, dtype=float)
    sc3 = np.asarray(sc3_recalls, dtype=float)
    rr  = np.asarray(rates, dtype=float)

    plt.plot(rr, sc1, "o-", color=BLUE, lw=2.5, ms=7,
             markerfacecolor=WHITE, markeredgewidth=2,
             label="Sc1 (MCAR)", zorder=3)
    plt.plot(rr, sc3, "s-", color=VERMILION, lw=2.5, ms=7,
             markerfacecolor=WHITE, markeredgewidth=2,
             label="Sc3 (MNAR sensitivity bound)", zorder=3)

    # Shade ONLY where Sc3 < Sc1 (correct around the r<=5% crossover)
    plt.fill_between(rr, sc3, sc1, where=(sc3 < sc1),
                     interpolate=True, alpha=0.12, color=VERMILION, zorder=1)

    plt.axvline(field_rate, color=AMBER, ls=":", lw=2.0, zorder=2)
    plt.axhline(TARGET, color="#555555", ls="--", lw=1.3, alpha=0.8, zorder=2)

    plt.xlim(-2, max(rates) + 9)
    plt.ylim(0.0, 1.02)                       # zero-based (no truncation)
    plt.xlabel("Missingness rate r (%)", 
            #    fontsize=12.5, 
               labelpad =4)
    plt.ylabel("SAM Recall", 
            #    fontsize=12.5
                )
    plt.title("SAM Recall degradation under missingness",
              color=NAVY, pad=12)

    # Legend upper-right: the curves descend, so that corner is free
    # plt.legend(loc="upper right", framealpha=0.95, fontsize=10)

    plt.grid(axis="y", color=GRID, zorder=0, lw=0.8)
    _clean_ax()

        # Legend below the axes (same style as WHZ/MUAC)
    handles = [

        mlines.Line2D(
            [], [],
            color=BLUE,
            marker="o",
            lw=2.5,
            markerfacecolor=WHITE,
            markeredgewidth=2,
            label="Sc1 (MCAR)"
        ),

        mlines.Line2D(
            [], [],
            color=VERMILION,
            marker="s",
            lw=2.5,
            markerfacecolor=WHITE,
            markeredgewidth=2,
            label="Sc3 (MNAR bound)"
        ),

        mlines.Line2D(
            [], [],
            color="#555555",
            ls="--",
            lw=1.6,
            label="Target = 0.90"
        ),

        mlines.Line2D(
            [], [],
            color=AMBER,
            ls=":",
            lw=2,
            label=f"Observed field missingness ({field_rate:.0f}%)"
        ),
    ]

    plt.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        # fontsize=9.5,
    )

    _save(filename, output_dir)


# ── Backward compat ───────────────────────────────────────────────────────────

def plot_boundary_failure(whz_data, muac_data,
                           output_dir="results/figures", filename=None):
    plot_whz_boundary(whz_data, output_dir=output_dir)
    plot_muac_boundary(muac_data, output_dir=output_dir)


def plot_global_vs_boundary(*args, **kwargs):
    pass


def plot_oracle_gating(*args, **kwargs):
    pass