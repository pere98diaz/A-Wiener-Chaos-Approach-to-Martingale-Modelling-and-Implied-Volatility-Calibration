# Wiener chaos pricing

Stock-price models based on Wiener chaos expansions, with calibration and option-pricing experiments for Heston, rough Heston, and SPX data. The experiments compare piecewise-constant and Legendre bases.

Code accompanying [A Wiener Chaos Approach to Martingale Modelling and Implied Volatility Calibration](https://arxiv.org/abs/2602.16232), by Pere Diaz-Lozano and Thomas K. Kloster.

## Installation

The environment in [requirements.txt](requirements.txt) was tested on Linux with Python 3.11.14. From the repository root, create a Python 3.11 environment and install the CPU version of PyTorch followed by the remaining dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU experiments, install the appropriate [PyTorch 2.5.1 CUDA build](https://pytorch.org/get-started/previous-versions/#v251) instead of the CPU build before installing the requirements. The experiment notebooks expose their device and Monte Carlo budgets in their configuration cells.

## Experiments

| Folder | Experiments |
| --- | --- |
| [Heston](Heston/) | Synthetic vanilla surfaces, repeated calibrations, and exotic option pricing |
| [Rough Heston](Rough%20Heston/) | Synthetic calibration surfaces and evaluation at held-out maturities |
| [Real data](Real%20data/) | SPX calibration, bid/ask comparisons, and timing analysis |

Each folder contains data preparation and plotting notebooks. Its `Piecewise constant/` and `Legendre/` subfolders contain:

- `Calibration.ipynb`: fit and evaluate one model.
- `Calibration study.ipynb`: run or resume 100 independent calibrations, or analyse saved runs. Reports cover fit quality, optimizer iterations, and runtime.

## Running the notebooks

Start `jupyter lab` from the repository root with the environment above activated, and select its Python kernel. Calibration and plotting notebooks locate the repository from their working directory and import `wiener_chaos` directly.

1. Prepare the input data if needed. Run `Heston/Heston data.ipynb`, `Rough Heston/Rough Heston data.ipynb`, or `Real data/Data.ipynb` from its containing dataset folder. SPX preparation requires `Real data/data_pere.csv`.
2. Open a calibration notebook in the chosen basis folder. Review its configuration, including Monte Carlo budgets, optimizer settings, and device selection, then run its cells in order.
3. For repeated fits, use `Calibration study.ipynb`. Its run/load settings control whether to analyse saved results or run the configured study. The study notebooks load completed results automatically; set `LOAD_SAVED_RESULTS = False` explicitly to start a new study.
4. Run the dataset's `Plots.ipynb` to recreate the paper figures from saved results. Heston's `Exotic options.ipynb` loads cached prices by default; set `RUN_PRICING = True` to create or resume exotic pricing.

Generated datasets and experiment results stay local and are excluded by `.gitignore`. SPX inputs and numerical results are not distributed.

## Data and reproducibility

### Heston and rough Heston

Run the respective data notebook to generate synthetic inputs, then use the calibration notebooks to produce results. When saved results are already available locally, the study notebooks load them automatically and the plotting notebooks recreate the figures without recalibration.

### SPX: bring your own input

Only the preprocessing, calibration, and analysis code is public. The original quotes, processed surfaces, calibrated coefficients, fitted prices/volatilities, tables, timings, and figures are excluded, including notebook outputs and recorded numerical results in notebook text. The original SPX numerical experiments cannot be reproduced from the public files alone.

To run the workflow with your own data, place a CSV at `Real data/data_pere.csv`. It must contain one European call-option quote per row with these columns:

| Column | Meaning and units |
| --- | --- |
| `strike` | Option strike, in the same currency units as the index |
| `bid`, `ask`, `mid` | Call prices in the same units; `mid` is the supplied midpoint |
| `idx_price` | Positive underlying index level for that quote |
| `ttm` | Positive time to maturity, in years |
| `rfr` | Annualized continuously compounded risk-free rate, as a decimal |
| `dvd` | Annualized continuously compounded dividend yield, as a decimal |
| `impl_vol` | Supplied annualized implied volatility, as a decimal |

Use a consistent index observation and a single rate/dividend pair per maturity, with one row per maturity/strike pair. The preprocessing normalizes strikes and option prices to an index level of 100 and calculates implied volatilities. Run `Real data/Data.ipynb` to create the local `real_data.pkl`, then run either basis's `Calibration.ipynb`. Adjust the per-maturity sampling allocations and device settings to match your input and machine. For the Legendre repeated study, first run the individual calibration to create its local timing reference. Studies and analysis notebooks then use the results generated from your own input.

All SPX inputs and generated artifacts stay local: the folder's Git rules allow only the seven workflow notebooks and its `.gitignore`. Keep notebook outputs and attachments cleared before sharing the notebooks.

## Saved results

Each basis folder uses the same layout under `Statistics calibration/`:

```text
Statistics calibration/
├── single/
│   ├── result.json
│   ├── coefficients/
│   ├── tables/
│   └── figures/
├── study/
│   ├── manifest.json
│   ├── results.json
│   ├── runs/run_001/result.json
│   ├── coefficients/
│   ├── tables/
│   ├── figures/
│   └── diagnostics/
└── study_latest.json
```

`result.json` stores one fit; `results.json` stores the aggregate study. Both include fitted coefficients, prices, implied volatilities, seeds, and timings. Coefficient tensors use `initial_coefficients.pt` and `calibrated_coefficients.pt`. Tables use `fit_summary.csv`, `fit_by_maturity.csv` for single fits, and `calibrations.csv` and `timing_summary.csv` for studies. Figures use `calibration`, `test`, and `study_summary` as applicable, in PNG and PDF formats. Additional coefficient diagnostics remain grouped with the study artifacts.

Study manifests retain source and data fingerprints. Incompatible runs use a separate `study_<fingerprint>/` directory. Completed studies update the relative `study_latest.json` pointer, which loading and plotting follow automatically. No result paths need copying between notebooks. Local backups are stored in `.project-backups/`, which is excluded from Git.

New source fingerprints ignore comments, formatting, and docstrings while retaining executable code. The existing exotic-price cache and SPX timing reference retain their original hashes alongside an explicit archived-source audit of the reorganized code. Changes to numerical code still invalidate those checks; data and experiment settings are checked separately.

The paper plots use study run 76 for Heston and Rough Heston, and run 5 for SPX. The Rough Heston Legendre study and matching single fit include the 20-million-path evaluation at the shortest held-out maturity; evaluation batches contain at most 4 million paths.

Paper figures are saved under `Heston/Plots/`, `Rough Heston/Plots/`, and `Real data/Plots/`. `Real data/Legendre/Calibration time analysis.ipynb` analyses the saved timing windows. Individual SPX piecewise bid/ask smiles are saved under `single/figures/smiles/`.

## Code layout

The reusable code is grouped by purpose:

```text
wiener_chaos/
├── core/
│   ├── model.py             # Stock model, pricing, and calibration
│   ├── chaos.py             # Conditional-square models and evaluators
│   ├── basis.py             # Chaos bases and Heston projection
│   ├── basis_integrals.py   # Basis integrals and covariance matrices
│   └── utils.py             # Multi-indices, Hermite features, Brownian draws
├── pricing/
│   ├── heston.py            # Heston pricing, simulation, and fitting
│   ├── rough_heston.py      # Rough Heston pricing and rational approximations
│   └── options.py           # Payoffs, Black–Scholes formulas, implied vols
└── experiments/
    ├── calibration.py       # Study workflows, checkpoints, and result loading
    ├── reporting.py         # Paper figures and coefficient diagnostics
    └── provenance.py        # Executable-source fingerprints
```

For example, from the repository root:

```python
from wiener_chaos.core.model import ModelStock
from wiener_chaos.pricing.options import compute_ivs
```

## License and citation

The code is available under the [MIT license](LICENSE). Citation metadata is provided in [CITATION.cff](CITATION.cff). Please cite the accompanying paper when using this code:

```bibtex
@article{diazlozano2026wiener,
  title = {A Wiener Chaos Approach to Martingale Modelling and Implied Volatility Calibration},
  author = {Diaz-Lozano, Pere and Kloster, Thomas K.},
  journal = {arXiv preprint arXiv:2602.16232},
  year = {2026},
  doi = {10.48550/arXiv.2602.16232}
}
```
