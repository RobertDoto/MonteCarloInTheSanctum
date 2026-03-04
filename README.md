# Sanctum Probabilities

Monte Carlo simulation of the Sanctum gatcha/loot system, estimating expected point accumulation over many rolls.

## Overview

The loot system awards items from a pool of 753 possibilities across three subsets. Each roll selects one item according to a probability distribution that changes over time: one-time items are removed after collection, and their probability is redistributed among remaining items within the same subset. Pity mechanics force outcomes after streaks of bad luck.

This project simulates the process 300,000 times in parallel to estimate expected cumulative points, percentile bands, and distribution shapes at every roll. GPU acceleration (via CuPy) is supported for NVIDIA hardware.

## Repository Structure

```
master/
├── analysis/
│   ├── expected_points.py          Main simulator (CPU/GPU, 300k sims x 1000 rolls)
│   ├── expected_points_cpu.py      CPU-only variant (100k sims x 3000 rolls)
│   ├── expected_points_gpu.py      GPU-only variant
│   ├── validity_testing.py         Statistical tests comparing CPU vs GPU output
│   └── variance_decomposition.py   Shapley-value variance attribution
├── plots/
│   ├── plot_generation.py          Generates all visualisations from saved results
│   └── *.png                       Pre-generated plots
├── results/
│   ├── simulation_results.csv      Summary statistics (one row per roll)
│   └── simulation_snapshots.npz    Full distributions at checkpoint rolls
├── tier data/
│   ├── items_all_normalised.csv    Complete item pool (755 entries)
│   ├── join_tiers.py               Merges per-tier CSVs into the master file
│   ├── A/                          A-tier source data and images
│   ├── B/                          B-tier source data, images, and OCR pipeline
│   └── S/                          S-tier source data and images
├── pity_collision.tex/.pdf         Markov chain analysis of simultaneous pity events
├── simulation_foundations.tex/.pdf  Mathematical foundations of the simulator
├── requirements.txt                Python dependencies
└── LICENSE                         CC BY-NC-ND 4.0
```

## Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib

Install dependencies:

```
pip install -r requirements.txt
```

### GPU Acceleration (Optional)

For NVIDIA GPUs, install [CuPy](https://cupy.dev/) matching your CUDA version:

```
pip install cupy-cuda12x
```

Replace `12x` with your CUDA version (e.g. `cupy-cuda11x`). The simulator auto-detects GPU availability and prompts for backend selection.

## Usage

### Run the simulation

```
python analysis/expected_points.py
```

This produces `results/simulation_results.csv` and `results/simulation_snapshots.npz`.

### Generate plots

```
python plots/plot_generation.py
```

Reads from `results/` and writes PNGs to `plots/`.

### Validate CPU vs GPU equivalence

```
python analysis/validity_testing.py
```

Runs both backends and applies t-test, F-test, and Kolmogorov-Smirnov tests.

## OCR Data Pipeline

The B-tier item probabilities were extracted from in-game screenshots using an OCR pipeline. This was necessary because the game does not expose drop rates in a machine-readable format — they are only visible as text rendered in the UI.

The pipeline lives in `tier data/B/ocr image pipeline/` and runs in two stages:

### Stage 1: Image to CSV (`ocr_process_images.py`)

Runs PaddleOCR on screenshot images to extract text, then parses each line into structured fields (percentage, item name, Mythic Essence value).

**How it works:**
1. Loads all `.png` screenshots from the input folder
2. Runs PaddleOCR on each image to detect text regions with bounding boxes
3. Filters out low-confidence detections (below 0.6 threshold)
4. Merges text boxes on the same visual row — OCR often splits a single line like `0.05343%  Got 'em  + 2 Mythic Essence` into separate detections for the percentage, name, and ME value. Boxes are merged when their vertical midpoints are within 15 pixels of each other
5. Sorts fragments left-to-right within each row, then parses the merged text using regex to extract the three fields
6. Writes raw OCR text to `ocr_raw_data.csv` and parsed results to `ocr_results.csv`

**Runs inside Docker** with GPU acceleration. The Dockerfile is based on `nvidia/cuda` and installs PaddleOCR + PaddlePaddle-GPU:

```
docker build -t ocr-app "tier data/B/ocr image pipeline/dockerised image processor"

docker run --gpus all \
  -v "/path/to/dockerised image processor:/app" \
  -v "/path/to/tier data/B/images:/app/images" \
  -v paddleocr-models:/root/.paddleocr \
  ocr-app
```

**Requirements:** Docker, NVIDIA GPU with CUDA, and the `nvidia-container-toolkit`.

### Stage 2: Quality assurance (`quality_assurance_b_tier.py`)

Post-processes the raw OCR output into a clean, normalised CSV matching the schema used by the rest of the project.

```
python "tier data/B/ocr image pipeline/quality_assurance_b_tier.py"
```

**Corrections applied:**
- Strips `%` symbols from percentages
- Fixes OCR artefacts in item names (trailing periods, incomplete ellipses)
- Corrects Roman numeral suffixes where OCR misreads characters (e.g. `l`/`i`/`1` → `I` in names like "Hangzhou 2022 VIII", "Pentakill III")
- Fixes word-initial `I` misread as `ll` (e.g. "llustration" → "Illustration")
- Deduplicates rows on the cleaned item name
- Corrects truncated display percentages: the game UI truncates to 4 significant figures (e.g. showing `0.05343%` instead of the true `14.053/263%`), so the script recomputes exact fractional values to ensure the tier sums to exactly 89.5%

**Output:** `tier data/B/b_tier_items_normalised.csv`, which feeds into `join_tiers.py` to produce the master `items_all_normalised.csv`.

## Mathematical Documentation

- **pity_collision.pdf** -- Derives the long-run probability of simultaneous S-tier and A-tier pity events using stationary distributions of Markov chains.
- **simulation_foundations.pdf** -- Explains why Monte Carlo simulation is used, and the theory behind two-stage sampling, inverse CDF sampling, probability redistribution, tier grouping, and percentile estimation.

LaTeX source files are included alongside each PDF.

## License

This work is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). You may share the material with attribution, but may not use it commercially or distribute modified versions.
