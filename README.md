# Power Consumption Classifier — MLOps Pipeline

Classifies Zone 1 power consumption (Tetouan City dataset) into
**Low / Medium / High** using an automated MLOps pipeline.

## Pipeline stages

| Stage | Script | Description |
|-------|--------|-------------|
| Preprocess | `src/preprocess.py` | Feature engineering, scaling, train/test split |
| Train | `src/model.py` | Random Forest vs KNN — best saved as deployment bundle |
| Evaluate | `src/evaluate.py` | Accuracy, F1, confusion matrix, feature importance |
| Monitor | `src/monitor.py` | KS-test drift detection + performance threshold check |

## Running locally

```bash
pip install -r requirements.txt

# Run all stages
dvc repro

# Or run individually
python src/preprocess.py
python src/model.py
python src/evaluate.py
python src/monitor.py

# Launch the Streamlit app
streamlit run app.py
```

## Configuration

All hyperparameters and file paths are in `params.yaml` — no need
to edit the scripts directly.

## Automation

GitHub Actions triggers the pipeline when `data/new_data.csv` is pushed.
New data is validated against a minimum row threshold before retraining.
Drift is detected using the Kolmogorov-Smirnov test on feature distributions.
Artifacts are versioned with DVC and stored on DagShub.
