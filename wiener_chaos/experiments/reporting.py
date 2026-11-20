"""Load saved studies and produce the figures used to compare fitted models.

Creates implied-volatility surfaces, error surfaces, smile comparisons, and
SPX bid/ask plots from stored results. Also computes coefficient distances and
summary diagnostics. Study-path resolution follows the latest completed run;
plotting uses saved prices and coefficients without repeating calibration.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ..core.utils import create_Lambda


MODEL_COLORS = ("#1f77b4", "#ff7f0e")
MODEL_MARKERS = ("s", "^")
_SHOW_FIGURES = False


def set_show_figures(enabled=True):
    """Display each saved Matplotlib figure in an active notebook."""
    global _SHOW_FIGURES
    _SHOW_FIGURES = bool(enabled)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def resolve_study_directory(directory):
    """Find the latest completed study, using paths relative to its base folder."""
    directory = Path(directory)
    pointer = directory.parent / f"{directory.name}_latest.json"
    if pointer.is_file():
        directory = pointer.parent / read_json(pointer)["directory"]
        if not (directory / "results.json").is_file():
            raise FileNotFoundError(f"No completed study at {directory}")
    return directory


def saved_run(study_dir, run_id=1):
    study_dir = resolve_study_directory(study_dir)
    path = study_dir / "runs" / f"run_{run_id:03d}" / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"Saved calibration run is missing: {path}")
    return read_json(path)


def _quote_arrays(maturities, strikes, iv, s0):
    if len(maturities) != len(strikes) or len(strikes) != len(iv):
        raise ValueError("Maturity, strike and IV row counts do not match.")
    rows = []
    for maturity, strike_row, iv_row in zip(maturities, strikes, iv):
        k = np.asarray(strike_row, dtype=float)
        v = np.asarray(iv_row, dtype=float)
        if k.shape != v.shape or k.ndim != 1:
            raise ValueError("A strike row and its IV row have different shapes.")
        valid = np.isfinite(k) & np.isfinite(v) & (k > 0)
        rows.append((np.log(k[valid] / s0), np.full(valid.sum(), maturity), v[valid]))
    return tuple(np.concatenate([row[i] for row in rows]) for i in range(3))


def _save(fig, path, *, pad_inches=0.18):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
    if _SHOW_FIGURES:
        from IPython.display import display
        display(fig)
    plt.close(fig)
    return path


def save_surface_pair(maturities, strikes, market_iv, model_ivs, labels, s0, folder):
    """Save the paper's 3D IV comparison and absolute-error surfaces in bp."""
    if len(model_ivs) != len(labels) or not model_ivs:
        raise ValueError("Provide one label for each model IV surface.")
    market = _quote_arrays(maturities, strikes, market_iv, s0)
    models = [_quote_arrays(maturities, strikes, iv, s0) for iv in model_ivs]
    for model in models:
        if not np.array_equal(model[0], market[0]) or not np.array_equal(model[1], market[1]):
            raise ValueError("Market and model quote grids do not match.")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(right=0.8)
    for i, (model, label) in enumerate(zip(models, labels)):
        ax.scatter(model[0], model[1], model[2], color=MODEL_COLORS[i % 2],
                   marker=MODEL_MARKERS[i % 2], s=32, alpha=0.55, label=label)
    ax.scatter(market[0], market[1], market[2], facecolors="none",
               edgecolors="black", marker="o", s=50, linewidths=1.3,
               depthshade=False, label="Reference")
    ax.set(xlabel="Log-moneyness", ylabel="Maturity", zlabel="Implied volatility")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5))
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3)
    surface_path = _save(fig, Path(folder) / "iv_surface_3d.pdf", pad_inches=0.45)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(right=0.8)
    legend = []
    for i, (model, label) in enumerate(zip(models, labels)):
        # Triangulation uses the same quote grid as the IV comparison.
        error_bp = np.abs(model[2] - market[2]) * 1e4
        ax.plot_trisurf(model[0], model[1], error_bp,
                        color=MODEL_COLORS[i % 2], alpha=0.60,
                        edgecolor="none", linewidth=0)
        legend.append(Patch(color=MODEL_COLORS[i % 2], alpha=0.60, label=label))
    ax.set(xlabel="Log-moneyness", ylabel="Maturity", zlabel="Absolute error (bps)")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5))
    ax.view_init(elev=25, azim=-60)
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    error_path = _save(fig, Path(folder) / "iv_surface_3d_errors.pdf", pad_inches=0.45)
    mae = {label: float(np.mean(np.abs(model[2] - market[2])) * 1e4)
           for model, label in zip(models, labels)}
    return [surface_path, error_path], mae


def save_combined_smiles(maturities, strikes, market_iv, model_ivs, labels, s0, folder):
    """Overlay every maturity on one set of axes, as in the paper."""
    if len(model_ivs) != len(labels):
        raise ValueError("Provide one label for each model IV surface.")
    if len(maturities) != len(strikes) or len(strikes) != len(market_iv):
        raise ValueError("Maturity, strike and IV row counts do not match.")
    if any(len(iv) != len(maturities) for iv in model_ivs):
        raise ValueError("A fitted IV surface has the wrong maturity count.")
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(maturities)))
    model_markers = ("o", "D", "s", "^")
    for i, (maturity, strike_row, market_row) in enumerate(zip(maturities, strikes, market_iv)):
        x = np.log(np.asarray(strike_row, float) / s0)
        color = colors[i]
        if len(x) != len(market_row):
            raise ValueError(f"Market smile at T={maturity} has the wrong strike count.")
        ax.plot(x, market_row, color=color, linestyle="-", marker="x",
                markersize=7, markeredgewidth=1.3, linewidth=1.1, zorder=2)
        for j, iv in enumerate(model_ivs):
            if len(iv[i]) != len(x):
                raise ValueError(f"Fitted smile at T={maturity} has the wrong strike count.")
            ax.plot(x, iv[i], color=color, linestyle="none", marker=model_markers[j % 4],
                    markerfacecolor="none", markeredgewidth=1.1,
                    markersize=7, zorder=3)
    handles = [Line2D([0], [0], color=colors[i], linewidth=2,
                      label=f"T = {maturity:.6g}")
               for i, maturity in enumerate(maturities)]
    handles.append(Line2D([0], [0], color="black", linestyle="none", marker="x",
                          markersize=7, label="Market"))
    handles.extend(Line2D([0], [0], color="black", linestyle="none",
                          marker=model_markers[j % 4], markerfacecolor="none",
                          markersize=7, label=label)
                   for j, label in enumerate(labels))
    ax.set(xlabel="Log-moneyness", ylabel="Implied volatility")
    ax.grid(alpha=0.20)
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    return _save(fig, Path(folder) / "iv_smiles_combined_.pdf")


def save_single_smile(maturity, strikes, market_iv, model_iv, s0, path,
                      reference_label="Reference", model_label="Piecewise constant",
                      reference_color="black", model_color=MODEL_COLORS[0],
                      reference_line=True, figsize=(8, 3.5), marker_size=7,
                      additional_model_iv=None, additional_model_label="Legendre"):
    x = np.log(np.asarray(strikes, float) / s0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, market_iv, linestyle="-" if reference_line else "none",
            marker="x", color=reference_color,
            markersize=marker_size, linewidth=1.2, label=reference_label)
    ax.plot(x, model_iv, linestyle="none", marker="o", markerfacecolor="none",
            markeredgecolor=model_color, markersize=marker_size, label=model_label)
    if additional_model_iv is not None:
        ax.plot(x, additional_model_iv, linestyle="none", marker="^",
                markerfacecolor="none", markeredgecolor=MODEL_COLORS[1],
                markersize=marker_size, label=additional_model_label)
    ax.set(xlabel="Log-moneyness", ylabel="Implied volatility")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def save_spx_spread_smile(strikes, bid_iv, ask_iv, model_ivs, labels, s0, path,
                          show_legend=False, legend_fontsize=14):
    """Plot fitted SPX smiles against the observed IV bid/ask band."""
    k = np.asarray(strikes, dtype=float)
    bid = np.asarray(bid_iv, dtype=float)
    ask = np.asarray(ask_iv, dtype=float)
    if len(model_ivs) != len(labels) or not model_ivs:
        raise ValueError("Provide one label for each fitted SPX smile.")
    fitted = [np.asarray(iv, dtype=float) for iv in model_ivs]
    if any(row.ndim != 1 or row.shape != k.shape for row in (bid, ask, *fitted)):
        raise ValueError("SPX strike, bid, ask and fitted IV rows must have the same shape.")
    if not np.all(np.isfinite(k)) or not np.all(k > 0):
        raise ValueError("SPX strikes must be finite and positive.")
    if not all(np.all(np.isfinite(row)) for row in (bid, ask, *fitted)):
        raise ValueError("SPX bid, ask and fitted IVs must be finite.")
    if np.any(bid > ask):
        raise ValueError("SPX bid IV exceeds ask IV.")
    x = np.log(k / s0)
    order = np.argsort(x)
    x, bid, ask = (row[order] for row in (x, bid, ask))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(x, bid, ask, color="gray", alpha=0.25,
                    linewidth=0, label="Market bid-ask", zorder=1)
    for i, (iv, label) in enumerate(zip(fitted, labels)):
        ax.plot(x, iv[order], linestyle="none", marker=MODEL_MARKERS[i % 2],
                markerfacecolor="none", markeredgecolor=MODEL_COLORS[i % 2],
                markersize=7, markeredgewidth=1.3, label=label, zorder=2 + i)
    ax.set(xlabel="Log-moneyness", ylabel="Implied volatility")
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=legend_fontsize)
    fig.tight_layout()
    return _save(fig, path)


def compare_coefficients(coefficients, config, run_labels=None):
    """Return scale- and sign-invariant comparisons of chaos factors.

    A factor theta represents the same stock model as c*theta for any nonzero
    scalar c. Its chaos expansion uses He_alpha/alpha!, whose squared L2 norm
    is 1/alpha!. Normalize by that norm, then choose a common global sign.
    """
    theta = np.asarray(coefficients, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[0] < 1:
        raise ValueError("Expected a nonempty matrix of coefficient vectors.")
    if not np.isfinite(theta).all():
        raise ValueError("Coefficient vectors must be finite.")

    order = int(config["P"])
    if "pi_M" in config:
        basis_count = len(config["pi_M"]) - 1
    elif "M" in config:
        basis_count = int(config["M"])
    else:
        raise ValueError("Provide pi_M or M to identify the chaos basis size.")
    dimension = int(config["d"])
    multi_indices, _, factorials = create_Lambda(order, basis_count, dimension, True)
    factorials = np.r_[1., np.asarray(factorials, dtype=np.float64)]
    if theta.shape[1] != len(factorials):
        raise ValueError(
            f"Expected {len(factorials)} chaos coefficients per run, "
            f"including the constant; got {theta.shape[1]}."
        )
    degrees = np.r_[0, np.sum(np.asarray(multi_indices, dtype=np.int64), axis=1)]
    weighted = theta / np.sqrt(factorials)
    norms = np.linalg.norm(weighted, axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Every fitted factor must have a positive finite norm.")

    unit = weighted / norms[:, None]
    signs = np.where(unit[:, 0] < 0, -1., 1.)
    # If a constant vanishes, use the largest coordinate to fix the sign.
    zero_constant = unit[:, 0] == 0
    if zero_constant.any():
        rows = np.flatnonzero(zero_constant)
        largest = np.abs(unit[rows]).argmax(axis=1)
        signs[rows] = np.where(unit[rows, largest] < 0, -1., 1.)
    unit *= signs[:, None]

    cosine = np.clip(unit @ unit.T, -1., 1.)
    distances = np.sqrt(np.maximum(0., 2. * (1. - cosine)))
    np.fill_diagonal(distances, 0.)
    medoid_index = int(distances.mean(axis=1).argmin())
    labels = list(range(1, len(theta) + 1)) if run_labels is None else list(run_labels)
    if len(labels) != len(theta):
        raise ValueError("run_labels must match the number of coefficient vectors.")
    pairs = distances[np.triu_indices(len(theta), 1)]

    return {
        "raw": theta,
        "unit": unit,
        "factorials": factorials,
        "degrees": degrees,
        "norms": norms,
        "signs": signs,
        "cosine": cosine,
        "distances": distances,
        "medoid_index": medoid_index,
        "run_labels": labels,
        "pairwise_distances": pairs,
        "summary": {
            "n_runs": len(theta), "n_coefficients": theta.shape[1],
            "medoid_run": labels[medoid_index],
            "mean_pairwise_distance": float(pairs.mean()) if len(pairs) else 0.,
            "median_pairwise_distance": float(np.median(pairs)) if len(pairs) else 0.,
            "p90_pairwise_distance": float(np.quantile(pairs, .9)) if len(pairs) else 0.,
            "maximum_pairwise_distance": float(pairs.max()) if len(pairs) else 0.,
            "mean_cosine_similarity": float(cosine[np.triu_indices(len(theta), 1)].mean())
                if len(pairs) else 1.,
        },
    }


def save_coefficient_comparison(comparison, output_dir, save_json):
    """Save exact raw vectors, aligned vectors, numbers and plots for a study."""
    output_dir = Path(output_dir)
    for folder in ("coefficients", "tables", "figures", "diagnostics"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "coefficients" / "calibrated_coefficients.npy", comparison["raw"])
    np.save(output_dir / "coefficients" / "normalized_coefficients.npy",
            comparison["unit"] * np.sqrt(comparison["factorials"])[None, :])
    save_json(output_dir / "diagnostics" / "coefficient_comparison.json", comparison["summary"])

    import pandas as pd

    medoid = comparison["medoid_index"]
    rows = pd.DataFrame({
        "run": comparison["run_labels"],
        "raw_factor_norm": comparison["norms"],
        "sign_flipped": comparison["signs"] < 0,
        "cosine_to_medoid": comparison["cosine"][:, medoid],
        "distance_to_medoid": comparison["distances"][:, medoid],
    })
    rows.to_csv(output_dir / "tables" / "coefficient_distances.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    heatmap = axes[0].imshow(comparison["distances"], cmap="viridis", vmin=0.,
                             vmax=max(1e-12, float(comparison["distances"].max())),
                             interpolation="nearest", aspect="auto")
    fig.colorbar(heatmap, ax=axes[0], label="Weighted L2 distance")
    axes[0].set(xlabel="Calibration run", ylabel="Calibration run",
                title="Pairwise distance between fitted factors")
    pairs = comparison["pairwise_distances"]
    if len(pairs):
        axes[1].hist(pairs, bins=min(25, max(1, int(np.sqrt(len(pairs))))),
                     color="steelblue", edgecolor="white")
    else:
        axes[1].text(.5, .5, "One run; no pairs", ha="center",
                     transform=axes[1].transAxes)
    axes[1].set(xlabel="Weighted L2 distance", ylabel="Run pairs",
                title="Distance distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "coefficient_similarity.png", dpi=150, bbox_inches="tight")

    spread = comparison["unit"].std(axis=0)
    top = np.argsort(spread)[::-1][:min(30, len(spread))]
    fig_spread, ax = plt.subplots(figsize=(max(10, .35 * len(top)), 5))
    image = ax.imshow(comparison["unit"][:, top], cmap="coolwarm", aspect="auto",
                      interpolation="nearest")
    fig_spread.colorbar(image, ax=ax, label="Normalized weighted coefficient")
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels([f"{j} (p{comparison['degrees'][j]})" for j in top],
                       rotation=90, fontsize=7)
    ax.set(xlabel="Coefficient index (chaos order)", ylabel="Calibration run",
           title="30 coefficients with the largest variation")
    fig_spread.tight_layout()
    fig_spread.savefig(output_dir / "figures" / "coefficient_spread.png", dpi=150,
                       bbox_inches="tight")
    return rows, (fig, fig_spread)


def coefficient_histogram_stats(comparison):
    """Summarize each aligned coefficient across calibration runs."""
    import pandas as pd

    values = comparison["unit"] * np.sqrt(comparison["factorials"])[None, :]
    return pd.DataFrame({
        "coefficient": np.arange(values.shape[1]),
        "chaos_order": comparison["degrees"],
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "p05": np.quantile(values, .05, axis=0),
        "median": np.median(values, axis=0),
        "p95": np.quantile(values, .95, axis=0),
        "fraction_positive": (values > 0).mean(axis=0),
        "weighted_std": comparison["unit"].std(axis=0),
    })


def plot_coefficient_histograms(comparison, indices, ncols=4):
    """Plot across-run histograms for any requested coefficient indices."""
    indices = [int(index) for index in indices]
    if not indices or any(index < 0 or index >= comparison["unit"].shape[1]
                          for index in indices):
        raise ValueError("Choose one or more valid coefficient indices.")
    values = comparison["unit"] * np.sqrt(comparison["factorials"])[None, :]
    columns = min(ncols, len(indices))
    rows = (len(indices) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 2.9 * rows),
                             squeeze=False)
    for ax, index in zip(axes.flat, indices):
        samples = values[:, index]
        ax.hist(samples, bins=min(20, max(5, int(np.sqrt(len(samples))))),
                color="steelblue", edgecolor="white")
        if samples.min() <= 0 <= samples.max():
            ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.axvline(samples.mean(), color="crimson", linewidth=1.5)
        ax.set(title=(f"Coefficient {index} (order {comparison['degrees'][index]})\n"
                      f"mean={samples.mean():.3g}, SD={samples.std():.3g}"),
               xlabel="Aligned, normalized coefficient", ylabel="Runs")
    for ax in list(axes.flat)[len(indices):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def save_coefficient_histograms(comparison, output_dir, top_n=12):
    """Save all coefficient statistics and a constant-versus-variable histogram grid."""
    output_dir = Path(output_dir)
    for folder in ("coefficients", "tables", "figures", "diagnostics"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    stats = coefficient_histogram_stats(comparison)
    stats.to_csv(output_dir / "tables" / "coefficient_histogram_stats.csv", index=False)
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    import pandas as pd

    selected = pd.concat([
        stats.iloc[[0]],
        stats.iloc[1:].nlargest(min(top_n - 1, len(stats) - 1), "weighted_std"),
    ])
    fig = plot_coefficient_histograms(comparison, selected["coefficient"])
    fig.savefig(output_dir / "figures" / "coefficient_histograms.png", dpi=150,
                bbox_inches="tight")
    return stats, fig
