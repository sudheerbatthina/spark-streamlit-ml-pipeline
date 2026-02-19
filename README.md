# DIC Phase 3 — PySpark + Streamlit ML

This project contains:
- A **PySpark notebook** for data preprocessing + feature engineering (`src/notebooks/DIC_PHASE_3.ipynb`)
- A **Streamlit app** for interactive model training and evaluation (`src/app/streamlit_app.py`)
- Supporting datasets in `data/` and a report in `reports/`

## Repo structure
```
dic-phase3-spark-streamlit-ml/
  src/
    notebooks/
      DIC_PHASE_3.ipynb
    app/
      streamlit_app.py
  data/
    dataset.csv
    cleaned_data.csv
    bonus_dataset.csv
  reports/
    Phase3report.pdf
  media/
    bonus_video.mp4
```

## Run the Streamlit app
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

## Run the notebook (PySpark)
Install notebook deps:
```bash
pip install -r requirements-notebook.txt
```

In the notebook, use the dataset path (recommended):
```python
file_path = "data/dataset.csv"
```
Tip: launch Jupyter from the **repo root** so relative paths work consistently.

## Notes
- If you plan to keep large files out of git later, consider Git LFS for videos/datasets.
- Remove or anonymize any personal identifiers before making the repo public.
