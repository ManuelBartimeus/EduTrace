# EduTrace Research Project

EduTrace is a research workspace for student dropout risk modeling and explainable intervention support. The repository is now split into a clear research layout so the notebook, generated artifacts, and runnable interface are separated without changing the contents of the original files.

## Repository layout

```text
EduTrace_Project-20260714T085048Z-1-001/
├── app/           # Flask app, frontend assets, and runtime model files
├── notebooks/     # Research notebook and notebook helper module
├── data/          # Source datasets used for modeling
├── models/        # Trained models and preprocessing artifacts
├── results/       # Evaluation outputs, plots, and metrics
├── logs/          # Run logs and experiment logs
├── requirements.txt
└── README.md
```

## What is where

The `app/` folder contains the runnable Flask interface together with the files it expects at runtime, including the HTML frontend and serialized model artifacts. The original app-specific README and requirements file were preserved inside that folder.

The `notebooks/` folder contains the main notebook used for model engineering and the local `shap.py` helper used by the notebook.

The `data/`, `models/`, `results/`, and `logs/` folders now hold the project outputs in a standard research-project layout.

## Setup

1. Create and activate a Python environment.
2. Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Run the web app

From the repository root, start the Flask app from the `app/` folder so its static files are served from the correct location:

```bash
cd app
python app.py
```

Then open the local address shown in the terminal, usually `http://localhost:5000`.

## Open the notebook

Open `notebooks/EduTrace_Main_2.ipynb` in Jupyter or VS Code to inspect the modeling workflow and reproduce the training logic.

## Notes

- The contents of the original files were not edited.
- The `app/` folder is the runtime bundle for the interface, so its files should stay together.
- The root `requirements.txt` is a consolidated environment file for both the notebook and the app.
