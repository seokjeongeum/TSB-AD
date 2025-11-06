<h1 align="center">MAD: Multimodal Framework for Adaptive Time-Series Anomaly Detection</h1>
<h2 align="center">Adaptive head selection with selective dimensionality reduction for multivariate TSAD</h2>

<p align="center">
<img alt="License" src="https://img.shields.io/github/license/TheDatumOrg/tsb-ad">
<img alt="Conference" src="https://img.shields.io/badge/IEEE%20BigData-2025-blue">
<img alt="Python" src="https://img.shields.io/badge/python-3.8--3.12-brightgreen">
</p>

### TL;DR
- MAD augments TSPulse with a two-stage pipeline: (1) selective dimensionality reduction (best-channel or PCA), then (2) Large Multimodal Model (LMM)-guided per-channel head selection using visual few-shot prompts.
- On TSB-AD-M (17 datasets, 5,851 time series), MAD achieves mean VUS-PR ≈ 0.437, improving over TSPulse zero-shot (0.361) and xLSTMAD (0.37), while reducing runtime by ~93% via dimensionality reduction.
- Label-free, dataset-agnostic, with configurable prompts and accuracy–efficiency trade-offs.

<h2 id="overview"> 📄 Why MAD?</h2>

- TSPulse uses static head selection and ingests all channels, which can amplify high-dimensional noise.
- MAD adapts heads per channel without labels using LMM visual reasoning and reduces input dimensionality to speed up inference.
- Contributions:
  - Adaptive, few-shot visual head selection across forecast/time/fft/ensemble heads.
  - Selective dimensionality reduction to remove uninformative channels.
  - Reproducible evaluation on TSB-AD-M with ablations and efficiency analysis.

<h2 id="start"> ⚙️ Get Started </h2>

<h3 id="dataset">🗄️ Dataset</h3>

- Multivariate benchmark: download TSB-AD-M from `https://www.thedatum.org/datasets/TSB-AD-M.zip` and extract under `Datasets/TSB-AD-M/`.
- **File lists used in our experiments** (see `MAD/_d_full_experiment.sh`):
  - **Multivariate experiments**: `Datasets/File_List/MAD-M.csv` (for most MAD variants) and `Datasets/File_List/TSB-AD-M.csv` (for ablation studies and comparisons)
  - Additional splits: `TSB-AD-M-Tuning.csv`, `TSB-AD-M-Eva.csv`, and others under `Datasets/File_List/`
- Expected layout:

```
Datasets/
  TSB-AD-M/
    <dataset>.csv  # last column 'Label'
  File_List/
    MAD-M.csv
    TSB-AD-M.csv
    ...
```

<h3 id="tsad">💻 Installation</h3>

Using the provided conda environment file:

```bash
conda env create -f MAD/environment.yml
conda activate tsb-ad-env
pip install -e .
```

Notes:
- The env installs PyTorch (cu126 wheels) via pip. If you need CPU-only, remove the extra-index-url and install the CPU wheel.
- Extras for LMM selection (Gemini): set `GEMINI_API_KEY` in your environment (supports `.env`). If not set, disable LMM selection or fall back to ensemble.

<h3 id="usage">🧑‍💻 Basic Usage (MAD)</h3>

Minimal Python example using MAD via `TSB_AD.model_wrapper`:

```python
import pandas as pd
from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.evaluation.metrics import get_metrics

csv_path = 'Datasets/TSB-AD-M/149_SMAP_id_6_Sensor_tr_2128_1st_5000-0.csv'
df = pd.read_csv(csv_path).dropna()
data = df.iloc[:, 0:-1].values.astype(float)
label = df['Label'].astype(int).to_numpy()

# Run MAD zero-shot (selective reduction + LMM-guided head selection)
scores = run_Unsupervise_AD('MAD_ZS', data, use_llm_selection=True, llm_few_shot_config='default')
metrics = get_metrics(scores, label)
print(metrics)
```

<h2 id="overview-arch"> 🧩 MAD Overview</h2>

<p align="center">
<img width="800" src="assets/fig/framework_diagram.png" alt="MAD Framework Diagram"/>
</p>

Two-stage workflow:
- **Stage 1**: Selective dimensionality reduction via best-channel selection or PCA (configurable `n_components`) to reduce high-dimensional noise.
- **Stage 2**: Per-channel head selection guided by an LMM (Gemini 2.5 Pro) with few-shot visual prompts across heads: forecast, time, fft, ensemble.

<p align="center">
<img width="600" src="assets/fig/best_channel_selection_example.png" alt="Best Channel Selection Example"/>
</p>

High-level pseudocode:

```text
Input X (T x C)
if use_dimensionality_reduction:
  X' = PCA/best-channel(X)
scores = {h: head(h).score(X') for h in [forecast, time, fft]}
scores['ensemble'] = aggregate(scores)
if use_llm_selection:
  for channel in channels(X'):
    selected_h[channel] = LMM_select(scores, raw=channel)
  S = concat(scores[selected_h[channel]] for channel)
else:
  S = scores['ensemble']
return S
```

See `MAD/MADPipeline.py` and `MAD/MADDetector.py`.

<h2 id="results"> 📊 Experimental Results</h2>

**Performance Summary**:

<p align="center">
<img width="800" src="assets/fig/consolidated_summary.png" alt="Consolidated Summary"/>
</p>

<p align="center">
<img width="800" src="assets/fig/consolidated_strategy_comparison.png" alt="Strategy Comparison"/>
</p>

- **Mean VUS-PR**: MAD (default few-shot) achieves **0.437** vs TSPulse ZS (0.361), xLSTMAD (0.37), and static triangulation rule (0.442).
- **Efficiency**: Runtime reduction ~93% via dimensionality reduction (see inference analysis below).
- **Ablations**: Results available in `MAD/analysis_multi_summary.txt` and CSVs under `MAD/` comparing dimensionality reduction and LLM selection components.

<h2 id="repro"> 🔁 Reproduce Paper Results</h2>

We provide scripts used in the paper:
- `MAD/_c_reproduce_tspulse.sh`: reproduce TSPulse baselines.
- `MAD/_d_full_experiment.sh`: run full MAD experiments over TSB-AD-M.
- `MAD/_e_compare_results.py`: aggregate results and generate comparison tables/PDFs (saved to `MAD/comparison_results/`).

Typical flow:
1) Download and extract TSB-AD-M under `Datasets/TSB-AD-M/`.
2) Set `GEMINI_API_KEY` if using LMM selection.
3) Run the script(s); logs and results are saved under `MAD/comparison_results/` and CSV summaries (`MAD/comparison_df_multivariate.csv`, `MAD/consolidated_vus_pr_scores.csv`).

Hardware and runtime: dimensionality reduction reduces runtime; LMM selection adds overhead. You can disable LMM via `use_llm_selection=False`.

<h2 id="prompting"> 🧪 Prompting & Visual Input Details</h2>

- Few-shot configs: `default`, `forecast_biased`, `non_forecast_biased` (see `MAD/MADPipeline.py`).
- Visual layout: large figure width and font for robust OCR by Gemini; raw data on top, followed by per-head scores, anomalies marked with red x in examples.
- Environment: set `GEMINI_API_KEY`; model `gemini-2.5-pro` by default.

We summarize prompt sensitivity, example count, and layout notes in `docs/EXPERIMENTS.md`.

<h2 id="efficiency"> ⚖️ Efficiency vs Accuracy Trade-offs</h2>

<p align="center">
<img width="800" src="assets/fig/inference_summary_time_cost.png" alt="Inference Summary Time Cost"/>
</p>

**Configuration options**:
- `use_dimensionality_reduction`: enable to reduce channels before scoring; controls: `n_components` for PCA (default: 3).
- `use_llm_selection`: enable per-channel head selection; otherwise use `ensemble`.
- `prediction_mode`: choose heads subset (e.g., `forecast+time+fft`).

<!-- <h2 id="citation"> 📑 Citation</h2>

If you use MAD, please cite our BigData 2025 paper:

```bibtex
@inproceedings{seok2025mad,
  title={{MAD: Multimodal Framework for Adaptive Time-Series Anomaly Detection}},
  author={Seok, Jeongeum and Han, Wook-Shin},
  booktitle={IEEE International Conference on Big Data},
  year={2025}
}
``` -->

<h2 id="license"> 📜 License & Acknowledgements</h2>

- License: Apache 2.0 (see `LICENSE`).
- Built on and evaluates with `TSB_AD` and `granite-tsfm` utilities. 
