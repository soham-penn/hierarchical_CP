"""Shared utilities for joint (X,Y) marginal DGP runners."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


O_COLORS = {
    0: "#0072B2",
    5: "#00A1D5",
    10: "#00B894",
    15: "#F39C12",
    20: "#E74C3C",
    25: "#8E44AD",
    30: "#C0392B",
    35: "#2C3E50",
    40: "#7F8C8D",
}

BASELINE_GRAYS = {
    "hcp": "#333333",
    "pool": "#666666",
    "sub": "#999999",
    "rep": "#bbbbbb",
}


def joint_mean(u_vec):
    u = np.asarray(u_vec, dtype=float).ravel()
    return u ** 2


def joint_covariance(u_vec, rho):
    u = np.asarray(u_vec, dtype=float).ravel()
    d = len(u)
    sigma = (1.0 - rho) * np.diag(u) + rho * np.ones((d, d), dtype=float)
    sigma = 0.5 * (sigma + sigma.T)
    sigma += 1e-8 * np.eye(d)
    return sigma


def draw_group_joint_xy(u_vec, n_obs, rho):
    mean = joint_mean(u_vec)
    cov = joint_covariance(u_vec, rho)
    z = np.random.multivariate_normal(mean=mean, cov=cov, size=n_obs)
    out = []
    for i in range(n_obs):
        out.append({"X": z[i, :-1].astype(float), "Y": float(z[i, -1])})
    return out


def _boxplot_style(
    ax,
    data,
    pos,
    color,
    width=0.58,
    edge_color=None,
    hatch=None,
    alpha=0.75,
    linestyle="-",
):
    if edge_color is None:
        edge_color = color

    bp = ax.boxplot(
        data,
        positions=[pos],
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color=edge_color, linewidth=1.0, linestyle=linestyle),
        capprops=dict(color=edge_color, linewidth=1.0, linestyle=linestyle),
        flierprops=dict(marker=".", color=edge_color, markersize=3, alpha=0.4),
        boxprops=dict(
            facecolor=color,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=1.0,
            linestyle=linestyle,
        ),
    )
    if hatch:
        for patch in bp["boxes"]:
            patch.set_hatch(hatch)
    return bp


def _metric_values(df, col, metric):
    vals = df[col].dropna().values
    if metric == "width":
        vals = vals[np.isfinite(vals)]
    return vals


def plot1_side_by_side_dhcp_vs_hcp(results_o, alpha, save_path, o_max=20):
    """Coverage/width side-by-side with D-HCP(o) and HCP."""
    o_vals = sorted(results_o["test_sample_size_o"].unique())
    if o_max is not None:
        o_vals = [o for o in o_vals if o <= o_max]
    results_use = results_o[results_o["test_sample_size_o"].isin(o_vals)].copy()
    baseline_df = results_use.drop_duplicates(subset=["experiment"])

    fig, axes = plt.subplots(1, 2, figsize=(18.0, 6.4))
    fig.subplots_adjust(wspace=0.35)

    for ax, metric, ylabel in [
        (axes[0], "coverage", "Coverage"),
        (axes[1], "width", "Width"),
    ]:
        pos = 0.0
        tick_positions = []
        tick_labels = []

        col_dhcp = f"{metric}_donor_hcp_randomized"
        for o in o_vals:
            vals = _metric_values(
                results_o.loc[results_o["test_sample_size_o"] == o],
                # keep style identical while restricting to requested o range
                col_dhcp,
                metric,
            )
            if len(vals) == 0:
                pos += 1.0
                continue
            _boxplot_style(ax, vals, pos, O_COLORS.get(o, "#666666"), width=0.58)
            tick_positions.append(pos)
            tick_labels.append(f"o={o}")
            pos += 1.0

        pos += 1.1
        vals_hcp = _metric_values(baseline_df, f"{metric}_hcp", metric)
        if len(vals_hcp) > 0:
            _boxplot_style(
                ax,
                vals_hcp,
                pos,
                color="#111111",
                edge_color="#111111",
                alpha=0.28,
                linestyle="--",
                hatch="///",
                width=0.58,
            )
            tick_positions.append(pos)
            tick_labels.append("HCP")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=18, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.set_xlabel("Method", fontsize=22)
        ax.tick_params(axis="y", labelsize=18)

        if metric == "coverage":
            target = 1.0 - alpha
            ax.axhline(target, color="black", linewidth=1.2, linestyle="--", alpha=0.55)
            ax.set_ylim(0.6, 1.05)
            ax.set_title("Coverage: D-HCP vs HCP", fontsize=24, fontweight="bold")
        else:
            ax.set_title("Width: D-HCP vs HCP", fontsize=24, fontweight="bold")

    # Simplified legend: show 4 color boxes + D-HCP label in center, and HCP
    legend_handles = []

    # Show first 4 o values as color boxes
    o_vals_display = o_vals[:4] if len(o_vals) >= 4 else o_vals
    for o in o_vals_display:
        legend_handles.append(
            mpatches.Patch(
                facecolor=O_COLORS.get(o, "#666666"),
                edgecolor=O_COLORS.get(o, "#666666"),
                alpha=0.75,
                label="",  # No individual labels
            )
        )

    # Add D-HCP label in the middle
    legend_handles.append(
        mpatches.Patch(
            facecolor="white",
            edgecolor="white",
            alpha=0.0,
            label="D-HCP",
        )
    )

    # Add HCP
    legend_handles.append(
        mpatches.Patch(
            facecolor="#111111",
            edgecolor="#111111",
            alpha=0.28,
            hatch="///",
            label="HCP",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        ncol=1,
        fontsize=20,
        framealpha=0.9,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_dhcp_stdcp_hcp_by_o(results_o, alpha, metric, save_path, o_max=40):
    """For each o, draw three boxes: D-HCP(o), Std-CP(o), and fixed HCP."""
    o_vals = sorted(results_o["test_sample_size_o"].unique())
    if o_max is not None:
        o_vals = [o for o in o_vals if o <= o_max]
    if len(o_vals) == 0:
        return

    results_use = results_o[results_o["test_sample_size_o"].isin(o_vals)].copy()
    baseline_df = results_use.drop_duplicates(subset=["experiment"])

    fig, ax = plt.subplots(figsize=(18.0, 6.6))

    group_gap = 3.8
    d_off, s_off, h_off = 0.0, 1.0, 2.0
    tick_positions = []
    tick_labels = []

    for gi, o in enumerate(o_vals):
        base = gi * group_gap

        vals_d = _metric_values(
            results_use.loc[results_use["test_sample_size_o"] == o],
            f"{metric}_donor_hcp_randomized",
            metric,
        )
        vals_s = _metric_values(
            results_use.loc[results_use["test_sample_size_o"] == o],
            f"{metric}_stdcp",
            metric,
        )
        vals_h = _metric_values(baseline_df, f"{metric}_hcp", metric)

        if len(vals_d) > 0:
            _boxplot_style(ax, vals_d, base + d_off, O_COLORS.get(o, "#666666"), width=0.58)
        if len(vals_s) > 0:
            _boxplot_style(
                ax,
                vals_s,
                base + s_off,
                color="#111111",
                edge_color="#111111",
                alpha=0.22,
                linestyle="--",
                hatch="xxx",
                width=0.58,
            )
        if len(vals_h) > 0:
            _boxplot_style(
                ax,
                vals_h,
                base + h_off,
                color="#111111",
                edge_color="#111111",
                alpha=0.35,
                linestyle="--",
                hatch="///",
                width=0.58,
            )

        tick_positions.append(base + 1.0)
        tick_labels.append(f"o={o}")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=16, rotation=0)
    ax.set_xlabel("Observed history size o", fontsize=20)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    if metric == "coverage":
        target = 1.0 - alpha
        ax.axhline(target, color="black", linewidth=1.2, linestyle="--", alpha=0.55)
        ax.set_ylabel("Coverage", fontsize=24)
        ax.set_ylim(0.20, 1.05)
        ax.set_title("Coverage by o: D-HCP vs Std-CP vs HCP", fontsize=24, fontweight="bold")
    else:
        ax.set_ylabel("Width", fontsize=24)
        ax.set_title("Width by o: D-HCP vs Std-CP vs HCP", fontsize=24, fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor="#777777", edgecolor="#777777", alpha=0.75, label="D-HCP (color encodes o)"),
        mpatches.Patch(facecolor="#111111", edgecolor="#111111", alpha=0.22, hatch="xxx", linestyle="--", label="Std-CP"),
        mpatches.Patch(facecolor="#111111", edgecolor="#111111", alpha=0.35, hatch="///", linestyle="--", label="HCP"),
    ]
    if metric == "coverage":
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linewidth=1.2,
                linestyle="--",
                alpha=0.55,
                label=f"Target ({1.0 - alpha:.0%})",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=16,
        framealpha=0.9,
        ncol=1,
        title="Legend",
        title_fontsize=18,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_with_all_baselines(results_o, alpha, metric, save_path):
    """Coverage/width with D-HCP(o) and all baseline methods."""
    o_vals = sorted(results_o["test_sample_size_o"].unique())
    baseline_df = results_o.drop_duplicates(subset=["experiment"])

    fig, ax = plt.subplots(figsize=(15.8, 6.4))
    pos = 0.0
    tick_positions = []
    tick_labels = []

    col_dhcp = f"{metric}_donor_hcp_randomized"
    for o in o_vals:
        vals = _metric_values(
            results_o.loc[results_o["test_sample_size_o"] == o],
            col_dhcp,
            metric,
        )
        if len(vals) == 0:
            pos += 1.0
            continue
        _boxplot_style(ax, vals, pos, O_COLORS.get(o, "#666666"), width=0.58)
        tick_positions.append(pos)
        tick_labels.append(f"o={o}")
        pos += 1.0

    pos += 1.2
    for key, label in [
        ("hcp", "HCP"),
        ("pool", "Pooling"),
        ("sub", "Subsampling"),
        ("rep", "Repeated"),
    ]:
        vals = _metric_values(baseline_df, f"{metric}_{key}", metric)
        if len(vals) == 0:
            pos += 1.6
            continue
        _boxplot_style(
            ax,
            vals,
            pos,
            color=BASELINE_GRAYS[key],
            edge_color="#222222",
            alpha=0.45,
            linestyle="--",
            hatch="//",
            width=0.58,
        )
        tick_positions.append(pos)
        tick_labels.append(label)
        pos += 1.6

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=18, rotation=18, ha="right")
    ax.set_xlim(-0.9, pos - 0.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Method", fontsize=22)
    ax.tick_params(axis="y", labelsize=18)

    if metric == "coverage":
        target = 1.0 - alpha
        ax.axhline(target, color="black", linewidth=1.2, linestyle="--", alpha=0.55)
        ax.set_ylabel("Coverage", fontsize=24)
        ax.set_ylim(0.6, 1.05)
        ax.set_title("Coverage: D-HCP vs baselines", fontsize=24, fontweight="bold")
    else:
        target = None
        ax.set_ylabel("Width", fontsize=24)
        ax.set_title("Width: D-HCP vs baselines", fontsize=24, fontweight="bold")

    legend_handles = []
    for o in o_vals:
        legend_handles.append(
            mpatches.Patch(
                facecolor=O_COLORS.get(o, "#666666"),
                edgecolor=O_COLORS.get(o, "#666666"),
                alpha=0.75,
                label=f"D-HCP, o={o}",
            )
        )
    legend_handles.extend(
        [
            mpatches.Patch(facecolor=BASELINE_GRAYS["hcp"], edgecolor="#222222", alpha=0.45, hatch="//", label="HCP"),
            mpatches.Patch(facecolor=BASELINE_GRAYS["pool"], edgecolor="#222222", alpha=0.45, hatch="//", label="Pooling"),
            mpatches.Patch(facecolor=BASELINE_GRAYS["sub"], edgecolor="#222222", alpha=0.45, hatch="//", label="Subsampling"),
            mpatches.Patch(facecolor=BASELINE_GRAYS["rep"], edgecolor="#222222", alpha=0.45, hatch="//", label="Repeated"),
        ]
    )
    if target is not None:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linewidth=1.2,
                linestyle="--",
                alpha=0.55,
                label=f"Target ({1.0 - alpha:.0%})",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=20,
        framealpha=0.9,
        ncol=1,
        title="Legend",
        title_fontsize=24,
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_requested_plots_from_results(results_o, alpha, dir_plots):
    # (1) HCP-only baseline view, up to o=20.
    plot1_side_by_side_dhcp_vs_hcp(
        results_o,
        alpha=alpha,
        save_path=str(dir_plots / "set1_side_by_side_dhcp_hcp_o_upto20.pdf"),
        o_max=20,
    )

    # (2) For each o (up to 40): D-HCP, Std-CP(o), and fixed HCP.
    plot_metric_dhcp_stdcp_hcp_by_o(
        results_o,
        alpha=alpha,
        metric="coverage",
        save_path=str(dir_plots / "set2_coverage_dhcp_stdcp_hcp_by_o_upto40.pdf"),
        o_max=40,
    )
    plot_metric_dhcp_stdcp_hcp_by_o(
        results_o,
        alpha=alpha,
        metric="width",
        save_path=str(dir_plots / "set2_width_dhcp_stdcp_hcp_by_o_upto40.pdf"),
        o_max=40,
    )

    # (3) All baselines except Std-CP, up to o=40.
    plot_metric_with_all_baselines(
        results_o,
        alpha=alpha,
        metric="coverage",
        save_path=str(dir_plots / "set3_coverage_dhcp_with_baselines_no_stdcp_o_upto40.pdf"),
    )
    plot_metric_with_all_baselines(
        results_o,
        alpha=alpha,
        metric="width",
        save_path=str(dir_plots / "set3_width_dhcp_with_baselines_no_stdcp_o_upto40.pdf"),
    )
