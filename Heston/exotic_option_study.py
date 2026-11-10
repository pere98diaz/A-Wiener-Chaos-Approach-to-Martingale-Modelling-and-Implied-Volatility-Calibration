"""Compare Heston and calibrated chaos models on exotic option prices.

Loads the saved piecewise-constant and Legendre fits, then prices lookback,
barrier, and forward-start options using shared Monte Carlo paths per model.
Batch checkpoints allow pricing to resume after checking coefficients, settings,
and source provenance. Cached prices can be converted to implied volatilities
and plotted without repeating the simulation.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from tqdm import tqdm

from wiener_chaos.experiments.calibration import build_model, json_safe, save_json
from wiener_chaos.experiments.provenance import source_fingerprint
from wiener_chaos.pricing.heston import sim_qe_heston_torch
from wiener_chaos.pricing.options import (
    barrier_implied_vol,
    floating_strike_lookback_implied_vol,
    forward_start_implied_vol,
)

DEFAULT_RESULTS = Path("Heston/Exotic option results/individual_calibrations_1M_current_cpu")

RUN_FILES = {
    "Piecewise constant": Path(
        "Heston/Piecewise constant/Statistics calibration/"
        "single/result.json"
    ),
    "Legendre": Path(
        "Heston/Legendre/Statistics calibration/"
        "single/result.json"
    ),
}
MATURITIES = [round(i / 10, 1) for i in range(1, 11)]
BARRIER_MATURITIES = [0.1, 0.5, 1.0]
STRIKES = [95, 100, 105, 110, 115, 120]
BARRIERS = [80, 90]
START_TIMES = [0.1, 0.2, 0.3]
END_TIME = 1.0
MONEYNESS = [0.9, 0.95, 1.0, 1.05, 1.1]
LABELS = ["Heston", *RUN_FILES]
MONITORING_STEP = 0.001
PRICING_SOURCES = (
    'wiener_chaos/pricing/heston.py',
    'wiener_chaos/core/model.py',
    'wiener_chaos/pricing/options.py',
    'wiener_chaos/core/chaos.py',
    'wiener_chaos/core/basis.py',
    'wiener_chaos/core/utils.py',
    'wiener_chaos/core/basis_integrals.py',
    'wiener_chaos/experiments/provenance.py',
)


def pricing_source_hashes(root):
    return {name: source_fingerprint(Path(root)/name)
            for name in PRICING_SOURCES}


def _matching_sources(manifest, root):
    """Accept current fingerprints or an explicit audit of historical sources."""
    recorded = manifest.get("verified_provenance", manifest)
    return (recorded.get("pricing_sources") == pricing_source_hashes(root)
            and recorded.get("source_sha256") == source_fingerprint(__file__, stop_before="_read"))


def monitoring_grid(maturity, extra_dates=()):
    dates = np.asarray(extra_dates, dtype=float)
    if (not np.isfinite(maturity) or maturity <= 0 or not np.isfinite(dates).all()
            or np.any(dates < 0) or np.any(dates > maturity)):
        raise ValueError("Monitoring/reset dates must lie between zero and maturity")
    return np.unique(np.round(np.r_[np.arange(0, maturity, MONITORING_STEP),
                                    0., dates, maturity], 6)).tolist()


def experiment_grid():
    return {"monitoring_step": MONITORING_STEP, "maturities": MATURITIES,
            "barrier_maturities": BARRIER_MATURITIES,
            "strikes": STRIKES, "barriers": BARRIERS, "start_times": START_TIMES,
            "end_time": END_TIME, "moneyness": MONEYNESS}


def _sha(payload):
    return hashlib.sha256(
        json.dumps(json_safe(payload), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _as_prices(value, expected_shape):
    result = np.asarray(value.detach().cpu(), dtype=float).reshape(expected_shape)
    if not np.isfinite(result).all():
        raise FloatingPointError("Exotic pricing produced a nonfinite price.")
    if (result < 0).any():
        raise FloatingPointError("The exotic call payoffs must have nonnegative prices.")
    return result.tolist()


def prepare_calibrations(root, data, device, budget, seed=42, output=None):
    """Load both saved individual calibrations and verify cached provenance."""
    root = Path(root)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Exotic pricing requires a CUDA device visible to this Python process; "
            f"PyTorch currently sees {torch.cuda.device_count()} devices."
        )
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} is unavailable.")

    models = {}
    coefficients = {}
    model_identity = {}
    for label, relative in RUN_FILES.items():
        path = root / relative
        run = json.loads(path.read_text())
        dataset = root / "Heston/heston_data.pkl"
        if (run.get("data_sha256") is not None and dataset.is_file()
                and run["data_sha256"] != hashlib.sha256(dataset.read_bytes()).hexdigest()):
            raise ValueError(f"{path} was calibrated against a different dataset.")
        config = run["config"].copy()
        config["device"] = str(device)
        model = build_model(data, config)
        coeff = torch.as_tensor(run["d_a"], dtype=model.type_float, device=device)
        expected = len(model.wienerchaos.Lambda) + 1
        if coeff.ndim != 1 or coeff.numel() != expected or not torch.isfinite(coeff).all():
            raise ValueError(f"Invalid coefficients in {path}.")
        norm = model.chaos.normalizer(coeff)
        if not torch.isfinite(norm) or abs(float(norm) - 1.) > 1e-4:
            raise ValueError(f"Saved coefficients in {path} are not normalized.")
        models[label] = model
        coefficients[label] = coeff
        model_identity[label] = {
            "calibration_file": str(relative),
            "saved_coefficient_sha256": _sha(run["d_a"]),
            "coefficient_sha256": _sha(coeff),
            "saved_factor_norm": float(norm),
            "config": config,
        }

    identity = {
        "budget": budget,
        "seed": seed,
        "device": str(device),
        "models": model_identity,
        "reference_parameters": data["params"],
        "pricing_data_sha256": _sha(data),
        "pricing_sources": pricing_source_hashes(root),
        "grid": experiment_grid(),
        # Plot-only changes must not invalidate completed Monte Carlo checkpoints.
        "source_sha256": source_fingerprint(__file__, stop_before="_read"),
    }
    output = (root / DEFAULT_RESULTS
              if output is None else Path(output))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        numerical_fields = identity.keys() - {"pricing_sources", "source_sha256"}
        if (any(previous.get(key) != json_safe(identity[key]) for key in numerical_fields)
                or not _matching_sources(previous, root)):
            raise ValueError(
                f"{output} belongs to different coefficients, settings, or code. "
                "Use a new run directory or remove that output after reviewing it."
            )
    else:
        if any(output.rglob("*.json")):
            raise ValueError(f"{output} contains results but has no manifest.")
        save_json(manifest_path, identity)
    return models, coefficients, output


def _shared_payoff_means(paths, grid, tasks, data):
    """Price all requested calls from one batch; minima stop at each expiry."""
    minima = paths.cummin(dim=1).values
    strikes = paths.new_tensor(STRIKES)
    barriers = paths.new_tensor(BARRIERS)
    moneyness = paths.new_tensor(MONEYNESS)
    end = int(np.searchsorted(grid, round(END_TIME, 6)))
    means = {}
    for kind, key in tasks:
        i = int(np.searchsorted(grid, round(float(key), 6)))
        if kind == "lookback":
            payoff = (paths[:, i]-minima[:, i]).clamp_min(0)[:, None]
        elif kind == "barrier":
            call = (paths[:, i, None]-strikes).clamp_min(0)
            alive = minima[:, i, None] > barriers
            payoff = call[:, :, None]*alive[:, None, :]
        else:
            payoff = (paths[:, end, None]-paths[:, i, None]*moneyness).clamp_min(0)
        maturity = END_TIME if kind == "forward_start" else key
        means[kind, key] = payoff.mean(dim=0, dtype=torch.float64)*np.exp(-data["r"]*maturity)
    return means


def _implied_vols(kind, key, prices, data):
    r, q, S0 = data["r"], data["q"], data["S0"]
    if kind == "lookback":
        return floating_strike_lookback_implied_vol(
            prices[0], S0, float(key), r, q, flag="call"
        )
    if kind == "barrier":
        return [
            [barrier_implied_vol(prices[i][j], S0, strike, float(key), r, q, barrier)
             for j, barrier in enumerate(BARRIERS)]
            for i, strike in enumerate(STRIKES)
        ]
    if kind != "forward_start":
        raise ValueError(f"Unknown exotic payoff: {kind}")
    return [
        forward_start_implied_vol(price, S0, float(key), END_TIME, r, q, m, option_type="call")
        for price, m in zip(prices, MONEYNESS)
    ]


@torch.no_grad()
def run_pricing(data, models, coefficients, output, device, budget, seed=42):
    """One shared path batch per model prices every maturity and payoff."""
    output = Path(output)
    device = torch.device(device)
    manifest = json.loads((output / "manifest.json").read_text())
    if (not _matching_sources(manifest, Path(__file__).resolve().parent.parent)
            or manifest.get("pricing_data_sha256") != _sha(data)
            or manifest.get("budget") != json_safe(budget) or manifest.get("seed") != seed
            or manifest.get("grid") != json_safe(experiment_grid())
            or manifest.get("device") != str(device)
            or any(manifest.get("models", {}).get(label, {}).get("coefficient_sha256") != _sha(coefficients[label])
                   for label in RUN_FILES)):
        raise ValueError("Pricing provenance changed; call prepare_calibrations with a fresh output directory")
    tasks = ([("lookback", t) for t in MATURITIES]
             + [("barrier", t) for t in BARRIER_MATURITIES]
             + [("forward_start", t) for t in START_TIMES])
    dates = MATURITIES + BARRIER_MATURITIES + START_TIMES + [END_TIME]
    grid = monitoring_grid(max(dates), dates)
    records = {}
    for kind, key in tasks:
        path = output / "data" / f"{kind}_{float(key):.2f}.json"
        record = json.loads(path.read_text()) if path.exists() else {
            "kind": kind, "key": key, "prices": {}, "iv": {},
        }
        if record["kind"] != kind or float(record["key"]) != float(key):
            raise ValueError(f"Incorrect checkpoint at {path}.")
        records[kind, key] = record
    missing = {label: [task for task in tasks if label not in records[task]["prices"]]
               for label in LABELS}
    total = sum(budget["heston" if label == "Heston" else "chaos"][1]
                for label in LABELS if missing[label])
    with tqdm(total=total, desc="Shared exotic paths", unit="batch") as progress:
        for label_index, label in enumerate(LABELS):
            if missing[label]:
                count, repetitions = budget["heston" if label == "Heston" else "chaos"]
                if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (count, repetitions)):
                    raise ValueError("Monte Carlo batch sizes and counts must be positive integers")
                sums = {}
                for batch in range(repetitions):
                    progress.set_postfix_str(f"{label}: {batch+1}/{repetitions}")
                    _seed(seed + label_index*1_000_000 + batch, device)
                    if label == "Heston":
                        paths = sim_qe_heston_torch(
                            data["S0"], data["r"], data["q"], data["params"], grid, count, device,
                        )
                    else:
                        paths = models[label].sample_spot_price(
                            coefficients[label], grid, count, data["r"], data["q"], return_on_device=True,
                        )
                    if not torch.isfinite(paths).all() or (paths < 0).any():
                        raise FloatingPointError(f"Invalid simulated paths for {label}")
                    means = _shared_payoff_means(paths, grid, missing[label], data)
                    for task, value in means.items():
                        sums[task] = sums.get(task, 0.) + value/repetitions
                    del paths
                    progress.update(1)
                for task, value in sums.items():
                    records[task]["prices"][label] = _as_prices(value, value.shape)
            for kind, key in tasks:
                record = records[kind, key]
                record["iv"][label] = _implied_vols(kind, key, record["prices"][label], data)
                save_json(output / "data" / f"{kind}_{float(key):.2f}.json", record)
    return output


def _read(output, kind, key):
    path = Path(output) / "data" / f"{kind}_{float(key):.2f}.json"
    record = json.loads(path.read_text())
    if record.get("kind") != kind or float(record.get("key", -1)) != float(key):
        raise ValueError(f"Incorrect exotic pricing checkpoint: {path}")
    if any(label not in record["prices"] for label in LABELS):
        raise ValueError(f"Incomplete exotic pricing checkpoint: {path}")
    # Older checkpoints contain the larger barrier grid. Select the requested
    # contracts in memory so replotting them requires no new Monte Carlo run.
    cached_strikes, cached_barriers = STRIKES, BARRIERS
    if kind == "barrier":
        manifest_path = Path(output) / "manifest.json"
        if manifest_path.exists():
            grid = json.loads(manifest_path.read_text()).get("grid", {})
            cached_strikes = grid.get("strikes", STRIKES)
            cached_barriers = grid.get("barriers", BARRIERS)
        strike_indices = [cached_strikes.index(strike) for strike in STRIKES]
        barrier_indices = [cached_barriers.index(barrier) for barrier in BARRIERS]
    shape = {"lookback": (1,), "barrier": (len(cached_strikes), len(cached_barriers)),
             "forward_start": (len(MONEYNESS),)}[kind]
    for label in LABELS:
        values = np.asarray(record["prices"][label], dtype=float)
        if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid prices for {label} in {path}")
        if kind == "barrier":
            record["prices"][label] = values[np.ix_(strike_indices, barrier_indices)].tolist()
    record.setdefault("iv", {})
    return record


def _series(values):
    return np.asarray(values, dtype=float)


def _save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.show()
    plt.close(fig)
    return path


def plot_results(results_dir, plots_dir, data):
    """Recreate the paper's six exotic-option figures from squared-model results."""
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 14})
    paths = []
    manifest_path = results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not _matching_sources(manifest, Path(__file__).resolve().parent.parent):
        warnings.warn(
            "Saved prices are not verified against this pricing implementation. "
            "Verify their archived source provenance or recompute with RUN_PRICING=True.",
            RuntimeWarning, stacklevel=2,
        )
    series_styles = {
        "Heston": dict(linestyle="none", marker="x", markersize=7),
        "Piecewise constant": dict(linestyle="--", marker="o", markersize=4,
                                   markerfacecolor="none"),
        "Legendre": dict(linestyle=":", marker="s", markersize=4,
                         markerfacecolor="none"),
    }
    model_handles = [
        Line2D([], [], color="0.4", **series_styles["Heston"], label="Heston"),
        Line2D([], [], color="0.5", **series_styles["Piecewise constant"], label="Piecewise"),
        Line2D([], [], color="0.5", **series_styles["Legendre"], label="Legendre"),
    ]
    metrics = (("prices", "Price"), ("iv", "Implied Volatility"))

    # The paper uses one wide panel for each payoff and metric, with raw IV units.
    lookback_records = [_read(results_dir, "lookback", t) for t in MATURITIES]
    # IVs are inexpensive to recompute; never trust cached formula versions.
    for maturity, record in zip(MATURITIES, lookback_records):
        for label in LABELS:
            record["iv"][label] = _implied_vols(
                "lookback", maturity, record["prices"][label], data,
            )
    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(12, 4))
        for label, color in zip(LABELS, ("black", "C0", "C1")):
            values = [_series(record[metric][label]).reshape(-1)[0]
                      for record in lookback_records]
            ax.plot(MATURITIES, values, color=color, **series_styles[label])
        ax.set(xlabel="T", ylabel="Prices" if metric == "prices" else ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=model_handles, loc="upper left" if metric == "prices" else "upper right")
        filename = "price_lookback.pdf" if metric == "prices" else "iv_lookback.pdf"
        paths.append(_save_figure(fig, plots_dir / "Lookback" / filename))

    # The paper overlays six (maturity, barrier) combinations on each barrier plot.
    barrier_grid = [(t, level) for t in BARRIER_MATURITIES for level in BARRIERS]
    barrier_records = {t: _read(results_dir, "barrier", t) for t in BARRIER_MATURITIES}
    for maturity, record in barrier_records.items():
        for label in LABELS:
            record["iv"][label] = _implied_vols("barrier", maturity, record["prices"][label], data)
    x_barrier = np.log(np.asarray(STRIKES) / data["S0"])
    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(12, 4))
        color_handles = []
        for color_index, (maturity, barrier) in enumerate(barrier_grid):
            color = f"C{color_index}"
            color_handles.append(Line2D([], [], color=color, label=f"T={maturity:.1f}, L={barrier}"))
            record = barrier_records[maturity]
            j = BARRIERS.index(barrier)
            for label in LABELS:
                values = _series(record[metric][label])[:, j]
                ax.plot(x_barrier, values, color=color, **series_styles[label])
        ax.set(xlabel="Log-moneyness", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=model_handles + color_handles, ncol=3, loc="upper right")
        filename = "price_combined.pdf" if metric == "prices" else "iv_combined.pdf"
        paths.append(_save_figure(fig, plots_dir / "Barrier" / filename))

    # Forward-start curves use the same color per start time and the paper's compact legend.
    x_forward = np.log(MONEYNESS)
    forward_records = {t: _read(results_dir, "forward_start", t) for t in START_TIMES}
    for start, record in forward_records.items():
        for label in LABELS:
            record["iv"][label] = _implied_vols("forward_start", start, record["prices"][label], data)
    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(12, 4))
        time_handles = []
        for color_index, start in enumerate(START_TIMES):
            color = f"C{color_index}"
            time_handles.append(Line2D([], [], color=color, label=rf"$\tau$={start:.1f}"))
            record = forward_records[start]
            for label in LABELS:
                ax.plot(x_forward, _series(record[metric][label]),
                        color=color, **series_styles[label])
        ax.set(xlabel="Log-moneyness", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=model_handles + time_handles, ncol=2, loc="upper right")
        filename = "price_combined.pdf" if metric == "prices" else "iv_combined.pdf"
        paths.append(_save_figure(fig, plots_dir / "Forward_call" / filename))
    return paths
