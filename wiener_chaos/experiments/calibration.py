"""Run, resume, and analyse calibration studies across the experiment notebooks.

Combines model construction, independent initialization, calibration, and
evaluation for synthetic surfaces and SPX data. Manages seeds, Monte Carlo
batches, timing checks, checkpoint identities, and automatic study selection.
Shared exporters organize JSON results, coefficient tensors, tables, and figures;
the notebooks keep their own experiment settings.
"""

import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..core.model import ModelStock
from ..pricing.options import compute_ivs
from .provenance import source_fingerprint
from .reporting import (
    compare_coefficients,
    resolve_study_directory,
    save_coefficient_comparison,
    save_coefficient_histograms,
)


class CalibrationTimingError(RuntimeError):
    """The current calibration must be discarded because it became too slow."""


class CalibrationTimer:
    """Record 25-step windows, or check them against a healthy individual run.

    Training windows use the same start and end timestamps as speed reports:
    after setup, then after completed optimizer steps every 25 iterations.
    """

    def __init__(self, reference=None, tolerance=0.10, *, simulate_every=50,
                 clock=time.perf_counter):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Timing tolerance must be finite and nonnegative.")
        self.reference = reference
        self.tolerance = tolerance
        self.simulate_every = simulate_every
        self.clock = clock
        self.interval = 25
        self.windows = []
        self.init_seconds = None
        self._init_started = None
        self._last_time = None
        self._last_iteration = 0

    def _check(self, label, seconds, reference_seconds, *, steps=None):
        limit = reference_seconds * (1 + self.tolerance)
        if seconds > limit:
            detail = (f"{seconds:.2f}s exceeds {limit:.2f}s "
                      f"(reference {reference_seconds:.2f}s + {100*self.tolerance:g}%). ")
            if steps is not None:
                detail = (f"{steps/seconds:.2f} iterations/s is below {steps/limit:.2f} iterations/s "
                          f"(reference {steps/reference_seconds:.2f} iterations/s; "
                          f"{100*self.tolerance:g}% longer per iteration allowed). ")
            raise CalibrationTimingError(
                f"{label}: {detail}"
                "Current run discarded. Completed runs are saved; rerun the study to resume."
            )

    def start_initialization(self):
        self._init_started = self.clock()

    def check_initialization(self):
        if self.reference is not None:
            self._check("Initialization", self.clock() - self._init_started,
                        self.reference["initialization_seconds"])

    def finish_initialization(self, seconds):
        self.init_seconds = float(seconds)
        if self.reference is not None:
            self._check("Initialization", seconds, self.reference["initialization_seconds"])

    def _reference_window(self, end):
        windows = self.reference["windows"]
        if end == self.interval:
            return windows[0]["seconds"]  # Same first 25 completed steps as the printed rate.
        # Compare matching MC-refresh phases, excluding first-window warmup.
        candidates = [w["seconds"] for w in windows[1:]
                      if not self.simulate_every
                      or w["end_iteration"] % self.simulate_every == end % self.simulate_every]
        if not candidates:
            raise ValueError("Timing reference lacks a matching MC-refresh window; run a longer individual calibration.")
        return max(candidates)

    def training_step(self, iteration, *, now=None):
        if iteration == 0:
            self._last_time = self.clock() if now is None else now
            self._last_iteration = 0
        elif iteration % self.interval == 0:
            now = self.clock() if now is None else now
            seconds = now - self._last_time
            self.windows.append({"end_iteration": iteration, "seconds": seconds})
            if self.reference is not None:
                self._check(f"Iterations {self._last_iteration + 1}–{iteration}", seconds,
                            self._reference_window(iteration), steps=self.interval)
            self._last_time = now
            self._last_iteration = iteration

    def profile(self):
        return {"version": 2, "check_every": self.interval,
                "simulate_every": self.simulate_every,
                "initialization_seconds": self.init_seconds,
                "windows": self.windows}


def effective_config(config):
    """Ignore seed, GPU index and display options when checking the reference."""
    config = copy.deepcopy(config)
    for key in ("base_seed", "device"):
        config.pop(key, None)
    for key in ("print_loss_every_it", "print_plots"):
        config["calibration"].pop(key, None)
    return config


def load_timing_reference(path, config, data_path, source_root):
    """Require a fresh individual profile with the same effective experiment."""
    path = Path(path)
    instruction = "Run the SPX Legendre individual Calibration notebook once at normal speed first."
    if not path.is_file():
        raise ValueError(f"No timing reference at {path}. {instruction}")
    result = json.loads(path.read_text())
    profile = result.get("timing_profile", {})
    if (profile.get("version") != 2 or profile.get("check_every") != 25
            or len(profile.get("windows", [])) < 3):
        raise ValueError(f"Missing or incomplete individual timing profile. {instruction}")
    windows = profile["windows"]
    durations = [profile.get("initialization_seconds")] + [w.get("seconds") for w in windows]
    if (any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in durations)
            or [w.get("end_iteration") for w in windows] != list(range(25, 25*(len(windows)+1), 25))
            or profile.get("simulate_every") != config["calibration"].get("simulate_every")):
        raise ValueError(f"Invalid timing reference. {instruction}")
    if effective_config(result["config"]) != effective_config(config):
        raise ValueError(f"Individual timing reference uses different hyperparameters. {instruction}")
    if result.get("data_sha256") != hashlib.sha256(Path(data_path).read_bytes()).hexdigest():
        raise ValueError(f"Individual timing reference uses different data. {instruction}")
    names = ("wiener_chaos/core/model.py", "wiener_chaos/pricing/heston.py", "wiener_chaos/core/chaos.py", "wiener_chaos/core/basis.py", "wiener_chaos/core/utils.py",
             "wiener_chaos/core/basis_integrals.py", "wiener_chaos/pricing/options.py", "wiener_chaos/experiments/calibration.py",
             "wiener_chaos/experiments/provenance.py")
    recorded = result.get("verified_provenance", {}).get("sources", result.get("source_sha256", {}))
    for name in names:
        if recorded.get(name) != source_fingerprint(Path(source_root)/name):
            raise ValueError(f"Individual timing reference predates changes to {name}. {instruction}")
    return profile


ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value):
    """Detach nested tensors and use JSON null for nonfinite values."""
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_artifacts(output_dir, *, coefficients=None, tables=None, figures=None,
                   figure_formats=("png", "pdf")):
    """Use the same coefficient, table and figure folders for every experiment."""
    output_dir = Path(output_dir)
    for folder in ("coefficients", "tables", "figures"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    for name, values in (coefficients or {}).items():
        tensor = torch.as_tensor(values).detach().cpu()
        torch.save(tensor, output_dir / "coefficients" / f"{name}.pt")
    for name, table in (tables or {}).items():
        table.to_csv(output_dir / "tables" / f"{name}.csv")
    for name, figure in (figures or {}).items():
        for extension in figure_formats:
            figure.savefig(output_dir / "figures" / f"{name}.{extension}",
                           dpi=150, bbox_inches="tight")


def surface_metrics(actual, expected):
    """Do not silently discard invalid IVs or mismatched surface entries."""
    if len(actual) != len(expected):
        raise ValueError("Different numbers of maturity rows.")
    errors = []
    invalid = 0
    for row, reference in zip(actual, expected):
        if len(row) != len(reference):
            raise ValueError("Different numbers of strikes.")
        for value, target in zip(row, reference):
            if value is None or not np.isfinite(value) or not np.isfinite(target):
                invalid += 1
            else:
                errors.append(abs(float(value) - float(target)))
    return {
        "mae": float(np.mean(errors)) if errors and not invalid else None,
        "invalid_quotes": invalid,
    }


def format_iv_mae(metrics):
    """Show an actual post-fit IV error, including invalid-quote status."""
    if metrics["mae"] is None:
        count = metrics["invalid_quotes"]
        return f"unavailable ({count} invalid {'IV' if count == 1 else 'IVs'})"
    return f"{metrics['mae'] * 1e4:.2f} bp"


def calibration_monitor_metrics(model, calibration):
    """Name the monitored objective by its actual units and sample."""
    result = {"calibration_iterations": model.last_calibration_iterations}
    interval = calibration.get("check_validation_set_MC")
    if interval is None:
        result["best_training_price_mse"] = model.last_calibration_monitor_loss
    elif calibration.get("validation_metric", "price_mse") == "iv_mae":
        result["best_validation_iv_mae_bp"] = model.last_calibration_monitor_loss * 1e4
    else:
        result["best_validation_price_mse"] = model.last_calibration_monitor_loss
    return result


def normalize_legacy_monitor_record(record, calibration):
    """Correct old records that stored price MSE under an IV-error name."""
    if ("best_validation_iv_mae_bp" not in record
            or (calibration.get("check_validation_set_MC") is not None
                and calibration.get("validation_metric", "price_mse") == "iv_mae")):
        return record
    corrected = record.copy()
    value = corrected.pop("best_validation_iv_mae_bp") / 1e4
    key = ("best_training_price_mse" if calibration.get("check_validation_set_MC") is None
           else "best_validation_price_mse")
    corrected[key] = value
    return corrected


def build_model(data, config):
    model = ModelStock(data["S0"], torch.device(config["device"]))
    if config.get("basis", "piecewise_constant") == "piecewise_constant":
        grid = config["pi_M"]
        model.initialize_chaos(
            "piecewise_constant", grid[-1], config["d"], config["P"],
            len(grid) - 1, pi_M=grid, backend=config.get("backend", "tensor"),
        )
    else:
        horizon = max(data["maturities_calibr"] + data["maturities_test"])
        model.initialize_chaos(
            config["basis"], horizon, config["d"], config["P"], config["M"],
            base_jitter=config["base_jitter"], backend=config.get("backend", "tensor"),
        )
    if "pricing_chunk_size" in config:
        model.chaos.chunk_size = int(config["pricing_chunk_size"])
        if model.chaos.chunk_size < 1:
            raise ValueError("pricing_chunk_size must be positive.")
    model.set_up_for_Ts(data["maturities_calibr"], data["r"], data["q"])
    model.set_up_for_Ts(data["maturities_test"], data["r"], data["q"])
    if config.get("compact_past_eigenvalue_cutoff") is not None:
        model.chaos.enable_compact_past(config["compact_past_eigenvalue_cutoff"])
        model.chaos.chunk_size = config.get("p2_chunk_size", model.chaos.chunk_size)
    if "comparison_N_per_mat" in config:
        counts = config["comparison_N_per_mat"]
        if len(counts) != len(data["maturities_calibr"]):
            raise ValueError("comparison_N_per_mat must cover every calibration maturity.")
        comparison_N = {round(float(t), 6): n for t, n in zip(data["maturities_calibr"], counts)}
        original_seeded_sample = model._simulate_X_block_with_seed

        def fixed_comparison_sample(maturities_block, method_block, N_block, Ns_block, seed=None):
            if seed is not None:
                N_block = [comparison_N.get(round(float(t), 6), n)
                           for t, n in zip(maturities_block, N_block)]
            return original_seeded_sample(maturities_block, method_block, N_block, Ns_block, seed)

        model._simulate_X_block_with_seed = fixed_comparison_sample
    return model


@torch.no_grad()
def evaluate_surface(model, coefficients, data, pricing, calibration, seed=None):
    """Price a surface, optionally limiting MC paths in memory with batch_size.

    N times Ns is the total sample count at each maturity. Control variates are
    fitted within each batch; prices are weighted by their pricing sample counts.
    Fixed seeds depend on maturity and batch indices and leave the RNG unchanged.
    """
    batch_size = pricing.get("batch_size")
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be a positive integer.")
    if batch_size is None and seed is not None:
        draws = model._simulate_X_block_with_seed(
            data["maturities"], pricing["method"], pricing["N"], pricing["Ns"], seed,
        )
        prices = model._prices_model_block(
            coefficients, draws, data["maturities"], data["strikes"],
            pricing["method"], pricing["nodes"],
            poly_cv_degree=calibration["poly_cv_degree"],
            n_samples_cv=calibration.get("n_samples_cv"),
        )
        prices = [row.detach().cpu().tolist() for row in prices]
    else:
        prices = []
        for index, (t, strikes, method, count, repetitions, nodes) in enumerate(zip(
            data["maturities"], data["strikes"], pricing["method"],
            pricing["N"], pricing["Ns"], pricing["nodes"],
        )):
            if batch_size is None:
                draws = (model._simulate_X_block([t], ["MC"], [count], [repetitions])[0]
                         if method == "MC" else None)
                price = model.compute_model_prices(
                    coefficients, t, strikes, method, X=draws, n=nodes,
                    poly_cv_degree=calibration["poly_cv_degree"],
                )
                prices.append(price.detach().cpu().tolist())
                del draws, price
                continue
            if method != "MC":
                price = model.compute_model_prices(coefficients, t, strikes, method, n=nodes)
                prices.append(price.detach().cpu().tolist())
                continue
            total = int(count) * int(repetitions)
            if total < 1:
                raise ValueError("MC sample counts must be positive.")
            limit = batch_size
            weighted_price = torch.zeros(len(strikes), dtype=torch.float64)
            weight = 0
            for batch, offset in enumerate(range(0, total, limit)):
                size = min(limit, total - offset)
                reserved = 0
                if calibration["poly_cv_degree"] is not None:
                    reserved = calibration.get("n_samples_cv") or 0
                if size <= reserved or (calibration["poly_cv_degree"] is not None
                                        and (reserved or size) < 2):
                    raise ValueError("Each MC batch needs enough control and pricing samples.")
                batch_seed = None if seed is None else int(seed) + index + batch * 1_000_000
                draws = model._simulate_X_block_with_seed([t], ["MC"], [size], [1], batch_seed)
                price = model._prices_model_block(
                    coefficients, draws, [t], [strikes], ["MC"], [None],
                    poly_cv_degree=calibration["poly_cv_degree"],
                    n_samples_cv=calibration.get("n_samples_cv"),
                )[0]
                weighted_price += price.detach().cpu().to(torch.float64) * (size - reserved)
                weight += size - reserved
                del draws, price
            prices.append((weighted_price / weight).tolist())
    ivs = compute_ivs(prices, data["maturities"], data["strikes"],
                      data["S0"], data["r_s"], data["q_s"])
    return prices, ivs


def test_surface(model, coefficients, data, pricing, calibration, seed=None):
    surface = {
        "maturities": data["maturities_test"], "strikes": data["strikes_test"],
        "S0": data["S0"], "r_s": data["r"], "q_s": data["q"],
    }
    return evaluate_surface(model, coefficients, surface, pricing, calibration, seed)


def validate_config(data, config):
    if config["n_calibrations"] < 1:
        raise ValueError("At least one calibration is required.")
    initialization = config["initialization"]
    if (initialization["type"] != "calibrate_heston"
            or initialization.get("initial_guess") not in ("surface", "random")):
        raise ValueError("This study requires per-run Heston-based initialization.")
    if config.get("basis", "piecewise_constant") != "piecewise_constant":
        if (initialization.get("projection_control") != "none"
                or initialization.get("projection_antithetic") is not False):
            raise ValueError("General-basis Heston projection requires the plain estimator.")
    horizon = (config["pi_M"][-1] if config.get("basis", "piecewise_constant") == "piecewise_constant"
               else max(data["maturities_calibr"] + data["maturities_test"]))
    for split in ("calibr", "test"):
        maturities = data[f"maturities_{split}"]
        pricing = config[f"pricing_{split}"]
        surfaces = [data[f"{name}_{split}"] for name in ("strikes", "prices", "iv")]
        allocations = [pricing[key] for key in ("method", "N", "Ns", "nodes")]
        if any(len(value) != len(maturities) for value in [*surfaces, *allocations]):
            raise ValueError(f"Misaligned {split} maturity inputs.")
        batch_size = pricing.get("batch_size")
        if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
            raise ValueError("batch_size must be a positive integer.")
        if any(not 0 < t <= horizon for t in maturities):
            raise ValueError("Maturities must lie within the chaos horizon.")
        for strikes, prices, ivs in zip(*surfaces):
            if not len(strikes) == len(prices) == len(ivs):
                raise ValueError("Misaligned strike inputs.")


def results_tables(records):
    calibration_rows = []
    for record in records:
        row = {key: record[key] for key in (
            "run", "seed", "projection_seed", "calibration_seed", "loss_ins", "loss_oos", "elapsed_seconds_init",
            "elapsed_seconds_chaos_calibration", "elapsed_seconds_total",
        )}
        for key in ("calibration_iterations", "best_training_price_mse",
                    "best_validation_price_mse", "best_validation_iv_mae_bp"):
            if key in record:
                row[key] = record[key]
        row["heston_objective"] = (record.get("heston_fit") or {}).get("objective")
        for label, key in (("calibration", "iv_calibration_metrics"), ("test", "iv_test_metrics")):
            mae = record[key]["mae"]
            row[f"iv_mae_{label}_bp"] = mae * 1e4 if mae is not None else np.nan
            row[f"invalid_iv_{label}"] = record[key]["invalid_quotes"]
        calibration_rows.append(row)
    runs = pd.DataFrame(calibration_rows)
    return runs


def tqdm_progress_callback(progress_bar):
    """Show the current phase and advance once for each completed or resumed run."""
    def update(message):
        progress_bar.set_postfix_str(message)
        if message.endswith(("complete", "loaded checkpoint")):
            progress_bar.update(1)

    return update


def export_results(records, output_dir, config):
    output_dir = Path(output_dir)
    runs = results_tables(records)
    # Keep familiar field names for downstream analysis, in a separate study file.
    analysis = {key: [record[key] for record in records] for key in (
        "d_a", "initial_coefficients", "heston_init_params", "heston_initial_guess", "heston_fit",
        "loss_ins", "loss_oos", "iv_model", "iv_model_test",
        "elapsed_seconds_chaos_calibration", "elapsed_seconds_init",
    )}
    analysis.update({
        "config": config, "n_completed": len(records), "runs": records,
        "loss_iv_oos": [r["iv_calibration_metrics"]["mae"] for r in records],
        "loss_iv_test": [r["iv_test_metrics"]["mae"] for r in records],
    })
    save_json(output_dir / "results.json", analysis)
    save_artifacts(output_dir, coefficients={
        "initial_coefficients": torch.tensor(analysis["initial_coefficients"], dtype=torch.float32),
        "calibrated_coefficients": torch.tensor(analysis["d_a"], dtype=torch.float32),
    }, tables={"calibrations": runs.set_index("run")})


def load_synthetic_results(directory):
    """Inspect a completed synthetic-data study without repeating calibrations."""
    directory = resolve_study_directory(directory)
    path = directory / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"No saved study results at {path}")
    saved = json.loads(path.read_text())
    records = [normalize_legacy_monitor_record(record, saved.get("config", {}).get("calibration", {}))
               for record in saved.get("runs", [])]
    if (not records or saved.get("n_completed") != len(records)
            or [record.get("run") for record in records] != list(range(1, len(records) + 1))):
        raise ValueError(f"Incomplete or inconsistent saved study: {path}")
    return records, directory, saved["config"]


def select_study_directory(base_dir, identity):
    """Resume matching checkpoints; preserve incompatible studies in place."""
    base_dir = Path(base_dir)

    def saved_identity(directory):
        path = directory / "manifest.json"
        return json.loads(path.read_text())["identity"] if path.exists() else None

    previous = saved_identity(base_dir)
    if previous == identity:
        return base_dir
    if previous is None and not any(base_dir.glob("runs/run_*/result.json")):
        return base_dir

    # A stable folder name resumes the same variant on later executions.
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output_dir = base_dir.with_name(f"{base_dir.name}_{fingerprint}")
    existing = saved_identity(output_dir)
    if existing is not None and existing != identity:
        raise ValueError(f"Unexpected checkpoint identity in {output_dir}.")
    if existing is None and any(output_dir.glob("runs/run_*/result.json")):
        raise ValueError(f"Checkpoints have no manifest in {output_dir}.")

    changes = []
    if previous is None:
        changes.append("existing checkpoints have no manifest")
    else:
        old_config = previous.get("config", {})
        new_config = identity["config"]
        changes.extend(
            f"config.{key}: {old_config.get(key)!r} -> {new_config.get(key)!r}"
            for key in sorted(set(old_config) | set(new_config))
            if old_config.get(key) != new_config.get(key)
        )
        if previous.get("data_sha256") != identity.get("data_sha256"):
            changes.append("input data changed")
        if previous.get("source_sha256") != identity.get("source_sha256"):
            changes.append("study or pricing code changed")
        if previous.get("experiment") != identity.get("experiment"):
            changes.append("comparison seed protocol changed")
    print("Using a separate results folder because " + "; ".join(changes) + ".")
    return output_dir


def run_study(data, config, output_dir, data_path, progress=print, n_calibrations=None, *,
              fixed_validation_seed=None, fixed_evaluation_seed=None, fixed_test_seed=None):
    """Run/resume the configured independent fits; checkpoint after every fit."""
    validate_config(data, config)
    run_count = config["n_calibrations"] if n_calibrations is None else int(n_calibrations)
    if not 1 <= run_count <= config["n_calibrations"]:
        raise ValueError("n_calibrations must be between 1 and the configured study total.")
    base_dir = Path(output_dir)
    root = ROOT
    source_paths = [root / name for name in (
        'wiener_chaos/core/model.py',
        'wiener_chaos/core/chaos.py',
        'wiener_chaos/core/basis.py',
        'wiener_chaos/core/utils.py',
        'wiener_chaos/pricing/options.py',
        'wiener_chaos/core/basis_integrals.py',
        'wiener_chaos/pricing/heston.py',
        'wiener_chaos/experiments/calibration.py',
        'wiener_chaos/experiments/provenance.py',
    )]
    source_hashes = {p.relative_to(root).as_posix(): source_fingerprint(p) for p in source_paths}
    identity = json_safe({
        "config": config,
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
    })
    if any(seed is not None for seed in (
            fixed_validation_seed, fixed_evaluation_seed, fixed_test_seed)):
        identity["experiment"] = {
            "fixed_validation_seed": fixed_validation_seed,
            "fixed_evaluation_seed": fixed_evaluation_seed,
            "fixed_test_seed": fixed_test_seed,
        }
    output_dir = select_study_directory(base_dir, identity)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        save_json(manifest_path, {
            "identity": identity,
            "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                        "device": config["device"], "dtype": "float32"},
        })
    print(f"Study results: {output_dir}")
    pricing = config["pricing_calibr"]
    records = []
    for index in range(run_count):
        run_id = index + 1
        path = output_dir / "runs" / f"run_{run_id:03d}" / "result.json"
        if path.exists():
            record = normalize_legacy_monitor_record(
                json.loads(path.read_text()), config["calibration"]
            )
            if record["run"] != run_id:
                raise ValueError(f"Unexpected run ID in {path}")
            records.append(record)
            progress(
                f"Run {run_id}/{run_count}: IV MAE calibration "
                f"{format_iv_mae(record['iv_calibration_metrics'])}, test "
                f"{format_iv_mae(record['iv_test_metrics'])}; loaded checkpoint"
            )
            continue
        seed = config["base_seed"] + index
        test_seed = (fixed_test_seed if fixed_test_seed is not None
                     else config["base_seed"] + 1_000_000 + index)
        heston_stage = "Heston fit" if config["initialization"].get("fit_heston", True) else "Heston heuristic"
        progress(f"Run {run_id}/{run_count}: {heston_stage} and projection")
        seed_everything(seed)
        model = build_model(data, config)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        init_config = config["initialization"]
        initial = model.initialize_params(
            type="calibrate_heston", maturities=data["maturities_calibr"],
            strikes=data["strikes_calibr"], prices_market=data["prices_calibr"],
            r=data["r"], q=data["q"], seed=seed,
            **{key: value for key, value in init_config.items() if key != "type"},
        )
        fit = model.heston_fit_result
        if fit is not None and not fit.success:
            raise RuntimeError(f"Run {run_id}: Heston fit did not converge: {fit.message}")
        heston_fit = ({"success": bool(fit.success), "iterations": int(fit.nit),
                       "objective": float(fit.fun), "message": str(fit.message)} if fit is not None else None)
        initial_coefficients = initial.detach().cpu().clone()
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        init_seconds = time.perf_counter() - started
        calibration_seed = seed + 3_000_000
        progress(f"Run {run_id}/{run_count}: chaos calibration (init {init_seconds:.1f}s)")
        seed_everything(calibration_seed)
        coefficients, prices, loss_in, loss_out, fit_seconds = model.calibrate(
            data["iv_calibr"], data["prices_calibr"], [data["maturities_calibr"]],
            data["strikes_calibr"], pricing["method"], pricing["N"],
            pricing["Ns"], pricing["nodes"], d_a=initial,
            validation_seed=fixed_validation_seed, evaluation_seed=fixed_evaluation_seed,
            **{**config["calibration"], "print_plots": False},
        )
        if not torch.isfinite(coefficients).all():
            raise FloatingPointError(f"Run {run_id}: nonfinite fitted coefficients")
        ivs = compute_ivs(prices, data["maturities_calibr"], data["strikes_calibr"],
                          data["S0"], data["r"], data["q"])
        progress(f"Run {run_id}/{run_count}: held-out test (cal {fit_seconds:.1f}s)")
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        test_started = time.perf_counter()
        if fixed_test_seed is None:
            seed_everything(test_seed)
        prices_test, ivs_test = test_surface(
            model, coefficients, data, config["pricing_test"], config["calibration"],
            seed=fixed_test_seed,
        )
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        test_seconds = time.perf_counter() - test_started
        record = json_safe({
            "run": run_id, "seed": seed, "projection_seed": seed,
            "calibration_seed": calibration_seed, "test_seed": test_seed,
            **calibration_monitor_metrics(model, config["calibration"]),
            "validation_seed": (fixed_validation_seed
                                if config["calibration"].get("check_validation_set_MC") is not None else None),
            "evaluation_seed": fixed_evaluation_seed,
            "reached_iteration_cap": model.last_calibration_iterations >= config["calibration"]["n_iter"],
            "d_a": coefficients,
            "initial_coefficients": initial_coefficients,
            "heston_init_params": model.heston_params,
            "heston_initial_guess": model.heston_initial_guess,
            "heston_fit": heston_fit,
            "loss_ins": loss_in, "loss_oos": loss_out,
            "prices_model": prices, "prices_model_test": prices_test,
            "iv_model": ivs, "iv_model_test": ivs_test,
            "iv_calibration_metrics": surface_metrics(ivs, data["iv_calibr"]),
            "iv_test_metrics": surface_metrics(ivs_test, data["iv_test"]),
            "elapsed_seconds_init": init_seconds,
            "elapsed_seconds_chaos_calibration": fit_seconds,
            "elapsed_seconds_total": init_seconds + fit_seconds,
            "elapsed_seconds_test_evaluation": test_seconds,
        })
        save_json(path, record)
        records.append(record)
        progress(
            f"Run {run_id}/{run_count}: IV MAE calibration "
            f"{format_iv_mae(record['iv_calibration_metrics'])}, test "
            f"{format_iv_mae(record['iv_test_metrics'])}; complete"
        )
        del initial, coefficients
    export_results(records, output_dir, config)
    if run_count == config["n_calibrations"]:
        save_json(base_dir.parent / f"{base_dir.name}_latest.json",
                  {"directory": output_dir.name})
    return records, output_dir


def initialize_parameters_timed(model, *, timing=None, **kwargs):
    """Time parameter initialization only, including completed GPU work."""
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    init_started = time.perf_counter()
    if timing is not None:
        timing.start_initialization()
        kwargs["timing_check"] = timing.check_initialization
    initial = model.initialize_params(**kwargs)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    seconds = time.perf_counter() - init_started
    if timing is not None:
        timing.finish_initialization(seconds)
    return initial, seconds


def build_spx_model(data, config):
    model = ModelStock(data["S0"], torch.device(config["device"]), torch.float32)
    if config.get("basis", "piecewise_constant") == "piecewise_constant":
        grid = config["pi_M"]
        model.initialize_chaos(
            "piecewise_constant", grid[-1], config["d"], config["P"],
            len(grid) - 1, pi_M=grid, backend=config["backend"],
        )
    else:
        model.initialize_chaos(
            config["basis"], max(data["maturities"]), config["d"], config["P"], config["M"],
            base_jitter=config["base_jitter"], backend=config.get("backend", "tensor"),
        )
    if config.get("pricing_chunk_size") is not None:
        chunk_size = config["pricing_chunk_size"]
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError("pricing_chunk_size must be a positive integer.")
        model.chaos.chunk_size = int(chunk_size)
    # SPX rates are term structures: retain the rate paired with each maturity.
    model.set_up_for_Ts(data["maturities"], data["r_s"], data["q_s"])
    return model


def plot_spx_smiles(data, fitted_ivs):
    columns = 3
    rows = (len(data["maturities"]) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.3 * rows), squeeze=False)
    for ax, T, strikes, market, fitted, bid, ask, r, q in zip(
        axes.flat, data["maturities"], data["strikes"], data["iv"], fitted_ivs,
        data["iv_bid"], data["iv_ask"], data["r_s"], data["q_s"],
    ):
        forward = data["S0"] * np.exp((r - q) * T)
        x = np.log(np.asarray(strikes) / forward)
        ax.fill_between(x, 100 * np.asarray(bid), 100 * np.asarray(ask),
                        alpha=.25, label="Market bid-ask")
        ax.plot(x, 100 * np.asarray(market), ".", label="Market IV")
        ax.plot(x, 100 * np.asarray(fitted, dtype=float), "-", label="Calibrated chaos")
        ax.set(title=f"T = {T:g}", xlabel="log(K / forward)", ylabel="Implied volatility (%)")
    for ax in list(axes.flat)[len(data["maturities"]):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("SPX calibration against Market bid-ask")
    fig.tight_layout()
    plt.show()
    return fig


def quiet_config(config, seed):
    result = copy.deepcopy(config)
    result["base_seed"] = int(seed)
    result["calibration"]["print_loss_every_it"] = None
    result["calibration"]["print_plots"] = False
    return result


def spx_initialization_mode(config):
    """Validate the study's Heston projection for the selected basis."""
    initialization = config.get("initialization", {})
    if (initialization.get("type") != "calibrate_heston"
            or initialization.get("initial_guess") != "surface"):
        raise ValueError("This study requires surface-based Heston initialization in every run.")
    fit_heston = initialization.get("fit_heston", True)
    if not isinstance(fit_heston, bool):
        raise ValueError("fit_heston must be a bool")
    prefix = "per_run_surface_heston" if fit_heston else "per_run_heuristic_heston"
    if (initialization.get("projection_control", "none") == "none"
            and initialization.get("projection_antithetic", False) is False):
        return f"{prefix}_plain_projection"
    if config.get("basis", "piecewise_constant") == "piecewise_constant":
        if (initialization.get("projection_control") != "linear_brownian"
                or initialization.get("projection_antithetic") is not True):
            raise ValueError("The piecewise study requires plain or controlled antithetic Heston projection.")
        return f"{prefix}_linear_brownian_antithetic"
    raise ValueError("The general-basis study requires plain Heston projection.")


def spx_study_identity(config, data_path, experiment=None):
    source_names = [
        'wiener_chaos/core/model.py',
        'wiener_chaos/pricing/heston.py',
        'wiener_chaos/pricing/options.py',
        'wiener_chaos/core/utils.py',
        'wiener_chaos/core/basis_integrals.py',
        'wiener_chaos/core/basis.py',
        'wiener_chaos/core/chaos.py',
        'wiener_chaos/experiments/calibration.py',
        'wiener_chaos/experiments/provenance.py',
    ]
    payload = json_safe({
        "config": config,
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "sources": {name: source_fingerprint(ROOT / name)
                    for name in source_names},
    })
    if experiment is not None:
        payload["experiment"] = experiment
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return signature, payload


def run_spx_calibration(data, config, seed, data_path, *,
                        fixed_validation_seed=None, fixed_evaluation_seed=None,
                        timing=None,
):
    """Run one independent SPX fit using the supplied notebook configuration."""
    run_config = quiet_config(config, seed)
    calibration_seed = seed + 3_000_000
    seed_everything(seed)  # A fresh projection sample for this run.
    model = build_spx_model(data, run_config)
    pricing = run_config["pricing_calibr"]
    comparison_N = run_config["comparison_N_per_mat"]
    if len(comparison_N) != len(data["maturities"]):
        raise ValueError("comparison_N_per_mat must cover every maturity.")
    original_seeded_sample = model._simulate_X_block_with_seed

    def fixed_comparison_sample(maturities_block, method_block, N_block, Ns_block, seed=None):
        """Keep validation/evaluation samples identical across training budgets."""
        if seed is None:
            return original_seeded_sample(
                maturities_block, method_block, N_block, Ns_block, seed
            )
        if len(maturities_block) != len(comparison_N):
            raise ValueError("This study calibrates all maturities in one block.")
        return original_seeded_sample(
            maturities_block, method_block, comparison_N, pricing["Ns"], seed
        )

    model._simulate_X_block_with_seed = fixed_comparison_sample
    init_config = run_config["initialization"]
    if init_config["type"] != "calibrate_heston":
        raise ValueError("This study requires Heston-based initialization in every run.")
    if init_config.get("initial_guess") != "surface":
        raise ValueError("This study requires the surface-derived Heston starting guess.")
    initial, init_seconds = initialize_parameters_timed(
        model, timing=timing, type="calibrate_heston", maturities=data["maturities"],
        strikes=data["strikes"], prices_market=data["prices"],
        r=data["r_s"], q=data["q_s"], seed=seed,
        **{key: value for key, value in init_config.items() if key != "type"},
    )
    fit = model.heston_fit_result
    if fit is not None and not fit.success:
        raise RuntimeError(f"Heston fit did not converge for seed {seed}: {fit.message}")
    heston_fit = ({"success": bool(fit.success), "iterations": int(fit.nit),
                   "objective": float(fit.fun), "message": str(fit.message)} if fit is not None else None)
    heston_params = model.heston_params
    heston_initial_guess = model.heston_initial_guess
    initial_coefficients = initial.detach().cpu().clone()
    seed_everything(calibration_seed)
    d_a, prices_model, loss_ins, loss_oos, fit_seconds = model.calibrate(
        data["iv"], data["prices"], [data["maturities"]], data["strikes"],
        pricing["method"], pricing["N"], pricing["Ns"], pricing["nodes"],
        d_a=initial, validation_seed=fixed_validation_seed,
        evaluation_seed=fixed_evaluation_seed, **run_config["calibration"],
        **({"timing_callback": timing.training_step} if timing is not None else {}),
    )
    if not torch.isfinite(d_a).all():
        raise FloatingPointError("Nonfinite calibrated coefficients.")
    iv_model = compute_ivs(prices_model, data["maturities"], data["strikes"],
                           data["S0"], data["r_s"], data["q_s"])
    chaos_source = "wiener_chaos/core/basis.py"
    return json_safe({
        "config": run_config, "seed": seed, "projection_seed": seed,
        "calibration_seed": calibration_seed,
        "validation_seed": (fixed_validation_seed
                            if run_config["calibration"].get("check_validation_set_MC") is not None else None),
        "evaluation_seed": fixed_evaluation_seed,
        "training_N_per_mat": pricing["N"],
        "comparison_N_per_mat": comparison_N,
        "d_a": d_a,
        "initial_coefficients": initial_coefficients,
        "heston_init_params": heston_params,
        "heston_initial_guess": heston_initial_guess,
        "heston_fit": heston_fit,
        "maturities": data["maturities"], "r_s": data["r_s"], "q_s": data["q_s"],
        "loss_ins": loss_ins, "loss_oos": loss_oos,
        **calibration_monitor_metrics(model, run_config["calibration"]),
        "reached_iteration_cap": model.last_calibration_iterations >= run_config["calibration"]["n_iter"],
        "prices_model": prices_model, "iv_model": iv_model,
        "iv_calibration_metrics": surface_metrics(iv_model, data["iv"]),
        "elapsed_seconds_init": init_seconds,
        "elapsed_seconds_chaos_calibration": fit_seconds,
        "elapsed_seconds_total": init_seconds + fit_seconds,
        **({"timing_profile": timing.profile()} if timing is not None else {}),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "heston_source_sha256": source_fingerprint(ROOT / "wiener_chaos/pricing/heston.py"),
        "model_source_sha256": source_fingerprint(ROOT / "wiener_chaos/core/model.py"),
        "chaos_source_sha256": source_fingerprint(ROOT / chaos_source),
        "study_source_sha256": source_fingerprint(__file__),
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": run_config["device"]},
    })


def resume_spx_identity(directory, identity, timing_tolerance):
    """Keep a study's checkpoints when explicitly relaxing its timing threshold."""
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Cannot resume: no study manifest at {path}")
    manifest = json.loads(path.read_text())
    previous = {key: manifest[key] for key in ("config", "data_sha256", "sources", "experiment")}
    signature = hashlib.sha256(json.dumps(previous, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if signature != manifest.get("study_signature"):
        raise ValueError("Saved study manifest has an inconsistent identity.")
    old_guard = previous["experiment"].get("timing_guard")
    new_guard = identity["experiment"].get("timing_guard")
    if old_guard is None or new_guard is None:
        raise ValueError("Explicit timing-tolerance resume requires an existing timing reference.")
    old_tolerance = manifest.get("active_timing_tolerance", old_guard["tolerance"])
    if timing_tolerance < old_tolerance:
        raise ValueError("This resume option only allows keeping or increasing the timing tolerance.")
    expected, actual = copy.deepcopy(previous), copy.deepcopy(identity)
    for item in (expected, actual):
        item["experiment"]["timing_guard"].pop("tolerance")
        # This workflow adds the resume option; all numerical model sources must match.
        item["sources"].pop("wiener_chaos/experiments/calibration.py")
    if expected != actual:
        raise ValueError("Cannot reuse checkpoints: calibration settings, data, numerical sources, "
                         "seeds or timing reference differ from the saved study.")
    # Preserve the original signature and provenance for the already accepted runs.
    if timing_tolerance != old_tolerance:
        manifest.setdefault("timing_tolerance_changes", []).append({
            "from": old_tolerance, "to": timing_tolerance,
            "completed_runs": len(list(Path(directory).glob("runs/run_*/result.json"))),
        })
    manifest["active_timing_tolerance"] = timing_tolerance
    manifest["resume_workflow_sha256"] = identity["sources"]["wiener_chaos/experiments/calibration.py"]
    return signature, manifest


def run_spx_study(data, config, data_path, *, n_calibrations=100, resume=True, output_root, progress_factory=tqdm,
              fixed_validation_seed=None, fixed_evaluation_seed=None,
              timing_reference=None, timing_tolerance=0.10,
              resume_directory=None,
):
    if isinstance(n_calibrations, bool) or not isinstance(n_calibrations, int) or n_calibrations < 1:
        raise ValueError("n_calibrations must be a positive integer.")
    simulate_every = config["calibration"].get("simulate_every")
    if simulate_every is not None and (
            isinstance(simulate_every, bool) or not isinstance(simulate_every, (int, np.integer))
            or simulate_every < 1):
        raise ValueError("simulate_every must be None or a positive integer.")
    initialization_mode = spx_initialization_mode(config)
    experiment = {
        "initialization_mode": initialization_mode,
        "projection_seed_mode": "per_run",
        "training_seed_mode": "per_run",
        "fixed_validation_seed": fixed_validation_seed,
        "fixed_evaluation_seed": fixed_evaluation_seed,
        "simulate_every": simulate_every,
    }
    reference = None
    if timing_reference is not None:
        reference = load_timing_reference(timing_reference, config, data_path, ROOT)
        CalibrationTimer(reference, timing_tolerance)  # Validate before writing any checkpoint.
        experiment["timing_guard"] = {
            "reference": reference, "tolerance": timing_tolerance,
            "reference_sha256": hashlib.sha256(Path(timing_reference).read_bytes()).hexdigest(),
        }
        print(f"Timing guard: individual reference + {100*timing_tolerance:g}%; "
              f"initialization limit {reference['initialization_seconds']*(1+timing_tolerance):.2f}s; "
              "checking every 25 iterations.")
    signature, identity = spx_study_identity(config, data_path, experiment)
    output_root = Path(output_root)
    if resume_directory is not None:
        if not resume:
            raise ValueError("Set resume=True when selecting an existing study directory.")
        output_dir = Path(resume_directory)
        signature, manifest = resume_spx_identity(output_dir, identity, timing_tolerance)
    else:
        output_dir = output_root / "study"
        existing_manifest = output_dir / "manifest.json"
        if existing_manifest.is_file():
            previous = json.loads(existing_manifest.read_text())
            if previous.get("study_signature") != signature:
                output_dir = output_root / f"study_{signature[:16]}"
        manifest = {**identity, "study_signature": signature}
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [None] * n_calibrations
    for index in range(n_calibrations):
        path = output_dir / "runs" / f"run_{index + 1:03d}" / "result.json"
        if resume and path.exists():
            cached = json.loads(path.read_text())
            # An interrupted run may have saved the source result before adding
            # study metadata. Rerun that incomplete checkpoint.
            if "study_signature" not in cached:
                continue
            expected_seed = int(config["base_seed"]) + index
            if (cached["study_signature"] != signature or cached.get("seed") != expected_seed
                    or cached.get("run") != index + 1
                    or cached.get("config") != json_safe(quiet_config(config, expected_seed))):
                raise ValueError(f"Checkpoint does not match this study: {path}")
            records[index] = cached

    save_json(output_dir / "manifest.json", manifest)
    if resume_directory is not None:
        print(f"Resuming {output_dir}: {sum(r is not None for r in records)} accepted, "
              f"{sum(r is None for r in records)} remaining.")

    with progress_factory(total=n_calibrations, initial=sum(r is not None for r in records),
                          desc="SPX calibrations", unit="calibration") as progress:
        for index in range(n_calibrations):
            if records[index] is not None:
                continue
            run_dir = output_dir / "runs" / f"run_{index + 1:03d}"
            timing_options = ({"timing": CalibrationTimer(reference, timing_tolerance,
                                                       simulate_every=simulate_every)}
                              if reference is not None else {})
            try:
                record = run_spx_calibration(
                    data, config, int(config["base_seed"]) + index, data_path,
                    fixed_validation_seed=fixed_validation_seed,
                    fixed_evaluation_seed=fixed_evaluation_seed,
                    **timing_options,
                )
            except CalibrationTimingError as error:
                accepted = [r for r in records if r is not None]
                save_json(output_dir / "results.json", {
                    "config": config, "n_completed": len(accepted), "study_signature": signature,
                    "runs": accepted, "stopped_run": index + 1, "stop_reason": str(error),
                })
                # Do not let Jupyter's saved traceback retain the rejected GPU samples.
                error.__traceback__ = None
                raise CalibrationTimingError(f"{error}\nCheckpoints: {output_dir}") from None
            record.update(run=index + 1, study_signature=signature)
            if reference is not None:
                record["timing_tolerance"] = timing_tolerance
            save_json(run_dir / "result.json", record)
            records[index] = record
            set_postfix = getattr(progress, "set_postfix_str", None)
            if set_postfix is not None:
                set_postfix(f"IV MAE {format_iv_mae(record['iv_calibration_metrics'])}")
            progress.update(1)

    save_json(output_dir / "results.json", {
        "config": config, "n_completed": len(records), "study_signature": signature,
        "runs": records,
    })
    save_artifacts(output_dir, coefficients={
        "initial_coefficients": torch.tensor([r["initial_coefficients"] for r in records], dtype=torch.float32),
        "calibrated_coefficients": torch.tensor([r["d_a"] for r in records], dtype=torch.float32),
    }, tables={"calibrations": pd.DataFrame([summarize_spx_run(r, data) for r in records]).set_index("run")})
    save_json(output_root / "study_latest.json", {"directory": os.path.relpath(output_dir, output_root)})
    return records, output_dir


def load_saved_results(directory):
    """Load complete study records without constructing or calibrating a model."""
    directory = resolve_study_directory(directory)
    path = directory / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"No saved study results at {path}")
    saved = json.loads(path.read_text())
    records = saved.get("runs", [])
    try:
        spx_initialization_mode(saved.get("config", {}))
    except ValueError as error:
        raise ValueError(f"Saved results are not from a supported per-run Heston projection study: {path}") from error
    if any("projection_seed" not in record for record in records):
        raise ValueError(f"Saved results are not from this per-run Heston projection study: {path}")
    if (not records or saved.get("n_completed") != len(records)
            or [record.get("run") for record in records] != list(range(1, len(records) + 1))):
        raise ValueError(f"Incomplete or inconsistent saved study: {path}")
    return records, directory, saved["config"]


def summarize_spx_run(record, data):
    inside = count = 0
    for ivs, bid, ask in zip(record["iv_model"], data["iv_bid"], data["iv_ask"]):
        values = np.asarray(ivs, dtype=float)
        inside += int((np.isfinite(values) & (values >= np.asarray(bid)) & (values <= np.asarray(ask))).sum())
        count += len(values)
    mae = record["iv_calibration_metrics"]["mae"]
    return {
        "run": record["run"], "seed": record["seed"],
        "calibration_seed": record.get("calibration_seed"),
        "projection_seed": record.get("projection_seed"),
        "heston_objective": (record.get("heston_fit") or {}).get("objective"),
        "calibration_iterations": record.get("calibration_iterations"),
        "reached_iteration_cap": record.get("reached_iteration_cap"),
        "iv_mae_bp": None if mae is None else 1e4 * mae,
        "invalid_iv_quotes": record["iv_calibration_metrics"]["invalid_quotes"],
        "within_bid_ask_pct": 100 * inside / count,
        "loss_ins": record["loss_ins"], "loss_oos": record["loss_oos"],
        "initialization_s": record["elapsed_seconds_init"],
        "calibration_s": record["elapsed_seconds_chaos_calibration"],
        "total_s": record["elapsed_seconds_total"],
    }


def relative_nonconstant_l2(comparison):
    values = comparison["unit"][:, 1:]
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise ValueError("A factor has zero nonconstant weighted L2 norm.")
    return np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1) / (
        (norms[:, None] + norms[None, :]) / 2
    )


def coefficient_report(records, config, output_dir, run_table, display=print):
    """Save and show the same weighted-L2 and histogram report for every study."""
    output_dir = Path(output_dir)
    for folder in ("coefficients", "tables", "figures", "diagnostics"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    ids = [record["run"] for record in records]
    initial = compare_coefficients([r["initial_coefficients"] for r in records], config, ids)
    final = compare_coefficients([r["d_a"] for r in records], config, ids)
    np.save(output_dir / "coefficients" / "initial_coefficients.npy", initial["raw"])
    save_json(output_dir / "diagnostics" / "initial_coefficient_comparison.json", initial["summary"])
    save_coefficient_comparison(final, output_dir, save_json)
    stats, histograms = save_coefficient_histograms(final, output_dir)

    initial_distances = relative_nonconstant_l2(initial)
    final_distances = relative_nonconstant_l2(final)
    pairs = np.triu_indices(len(records), 1)
    initial_pairs = initial_distances[pairs]
    final_pairs = final_distances[pairs]
    summary = {
        "initial_mean_pairwise": float(initial_pairs.mean()) if len(initial_pairs) else 0.0,
        "initial_max_pairwise": float(initial_pairs.max()) if len(initial_pairs) else 0.0,
        "final_mean_pairwise": float(final_pairs.mean()) if len(final_pairs) else 0.0,
        "final_max_pairwise": float(final_pairs.max()) if len(final_pairs) else 0.0,
    }
    save_json(output_dir / "diagnostics" / "nonconstant_relative_l2_summary.json", summary)
    np.savez(output_dir / "coefficients" / "nonconstant_relative_l2_matrices.npz",
             initial=initial_distances, final=final_distances)
    heston_parameters = np.asarray([r["heston_init_params"] for r in records], dtype=float)
    print(f"Initial mean/max pairwise relative weighted L2: "
          f"{summary['initial_mean_pairwise']:.3e} / {summary['initial_max_pairwise']:.3e}")
    print(f"Final mean/max pairwise relative weighted L2: "
          f"{summary['final_mean_pairwise']:.3e} / {summary['final_max_pairwise']:.3e}")
    print(f"Maximum across-run difference in fitted Heston parameters: "
          f"{np.ptp(heston_parameters, axis=0).max():.3e}")

    columns = ["projection_seed", "calibration_seed"]
    if "calibration_iterations" in run_table:
        columns.append("calibration_iterations")
    columns += ["heston_objective", "loss_ins", "loss_oos"]
    columns += (["iv_mae_bp"] if "iv_mae_bp" in run_table
                else ["iv_mae_calibration_bp", "iv_mae_test_bp"])
    sampling = run_table[columns].copy()
    sampling["initial_relative_weighted_l2_to_run1"] = initial_distances[0]
    sampling["final_relative_weighted_l2_to_run1"] = final_distances[0]
    display(sampling)
    sampling.to_csv(output_dir / "tables" / "sampling_variability.csv")
    if "iv_mae_bp" in run_table:
        raw = np.asarray([r["d_a"] for r in records], dtype=float)
        directions = final["unit"][:, 1:]
        lengths = np.linalg.norm(directions, axis=1)
        if np.all(lengths > 0):
            unit_directions = directions / lengths[:, None]
            similarity = float((unit_directions @ unit_directions.T)[pairs].mean()) if len(pairs[0]) else 1.0
        else:
            similarity = float("nan")
        print(f"Maximum raw coefficient difference: {np.ptp(raw, axis=0).max():.3e}")
        print(f"Mean similarity of nonconstant coefficients: {similarity:.3f} "
              "(1 = same direction)")
    print("Final-coefficient histograms: constant coefficient and the 11 most variable nonconstant coefficients.")
    display(stats.iloc[1:].nlargest(11, "weighted_std")[[
        "coefficient", "chaos_order", "mean", "std", "p05", "p95"
    ]])
    display(histograms)
    return final
