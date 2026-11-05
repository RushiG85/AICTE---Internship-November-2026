# AICTE-Internship-November-2026

NSAP Scheme Prediction
A simple machine learning project that classifies district-level NSAP records into three scheme categories: IGNDPS, IGNOAPS and IGNWPS.
Note: This is an educational/research project using district-level aggregate data. It does not check an individual's eligibility for an NSAP benefit.

1. Project files
- app.py — Streamlit web application
- eda.py — data exploration and cleaning
- train_model.py — trains and saves the machine learning model
- validate_model.py — evaluates the model
- inspect_data.py — checks the dataset
- data/ — input dataset
- models/ — saved model files
- outputs/ — reports and charts
- requirements.txt — required Python packages
2. Setup (Windows)
Open PowerShell in the project folder and run:
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
If PowerShell blocks activation, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
3. Run the project
If the saved model is not present, train it first:
python train_model.py
Start the web application:
python -m streamlit run app.py
Open the local URL shown in the terminal (usually http://localhost:8501).
4. Evaluate the model
Run:
python validate_model.py
The Random Forest model achieved about 83% accuracy in grouped cross-validation on this dataset. Results are exploratory and depend on the small aggregate dataset.
5. Dataset
The project uses district-wise NSAP pension aggregate data from AI Kosh. Keep the dataset in the data folder and make sure its filename matches the path configured in the Python scripts.
