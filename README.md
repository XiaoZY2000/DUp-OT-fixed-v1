# DUP-OT

Official implementation of **DUP-OT**, accepted at the **PAKDD 2026 DSFA Special Session**.

DUP-OT is a distribution-based framework for non-overlapping cross-domain recommendation,
which models user preferences as Gaussian mixture distributions and aligns domains
via Optimal Transport.

## Quick Start

### 1. Setup Environment
```bash
conda env create -f environment.yaml
conda activate dup-ot
```

### 2. Prepare Data
Place raw Amazon review JSON files in `data/raw/`:
```
data/raw/Digital_Music.json.gz
data/raw/Electronics.json.gz
```

### 3. Run Full Pipeline
```bash
python run_pipeline.py --config config.yaml --dataset_pair Digital_Music,Electronics --stages all --mode rating
```

### 4. Run Specific Stages
```bash
# Data processing only
python run_pipeline.py --stages gen_data

# Train + eval only (assumes preprocessed data exists)
python run_pipeline.py --stages train,eval

# Ablation study
python run_pipeline.py --stages train_ablation,eval_ablation
```

## Project Structure

```
DUP-OT/
├── config.yaml              # All configurable parameters
├── environment.yaml          # Conda environment
├── run_pipeline.py          # Entry point (raw data → eval)
├── data/
│   └── raw/                 # Place raw data here
├── src/
│   ├── data/                # Data loading, filtering, splitting
│   │   ├── filter_dataset.py
│   │   ├── gen_data.py
│   │   ├── split_dataset.py
│   │   └── choose_source_target.py
│   ├── preprocess/          # Embedding generation, autoencoder
│   │   ├── data_processing.py
│   │   └── auto_encoder.py
│   ├── model/               # Core models
│   │   ├── gmm.py           # GMM (trainable + frozen modes)
│   │   ├── components.py    # Shared NN components & datasets
│   │   ├── rating_model.py  # Rating prediction (cross-domain + ablation)
│   │   └── ranking_model.py # BPR ranking (cross-domain + ablation)
│   ├── transport/           # Optimal transport
│   │   └── ot_plan.py       # Cost matrix, EMD, MMD
│   ├── eval/                # Evaluation metrics
│   │   └── metrics.py       # RMSE, MAE, HR@K, NDCG@K
│   └── utils/               # Utilities
│       ├── seed.py
│       ├── device.py
│       └── io.py
├── stored/                  # Auto-created: cached embeddings, models, results
└── model/                   # Auto-created: autoencoder checkpoints
```

## Key Configuration Options

### GMM Modes (`config.yaml → gmm`)
- **`trainable: true`** — GMM means/variances are `nn.Parameter`, fine-tuned during training with lower LR (`gmm_lr_scale`)
- **`trainable: false`** — GMM parameters are frozen (registered buffers), only the weight learner and predictor are trained

### Task Modes (`config.yaml → mode`)
- **`rating`** — Rating prediction (RMSE/MAE)
- **`ranking`** — BPR ranking (HR@K/NDCG@K)

### Pipeline Stages
| Stage | Description |
|-------|-------------|
| `gen_data` | Filter raw data → build interaction JSON |
| `split` | Time-based train/val/test split |
| `preprocess` | Review encoding → autoencoder → GMM → OT plan |
| `train` | Cross-domain model training |
| `eval` | Test set evaluation |
| `train_ablation` | Target-only training (ablation) |
| `eval_ablation` | Target-only evaluation |
