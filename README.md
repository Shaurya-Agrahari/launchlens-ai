# LaunchLens AI — Startup Idea & Success Planning Studio

A beginner-friendly Flask + scikit-learn final-year project for startup idea discovery, planning and data exploration.

## Features
- User registration/login with hashed passwords and SQLite.
- Startup idea cards with customer, problem, MVP, revenue model, pros and cons.
- Interactive startup-planning score form and actionable recommendations.
- Model comparison: Random Forest, Extra Trees and Logistic Regression.
- Holdout evaluation metrics: accuracy, precision, recall, F1 and ROC-AUC where available.
- Dataset exploration charts and category distribution.
- FAQ-style AI assistant that works locally without an API key.
- Links to startup discovery and research platforms (external websites).
- Responsive colorful dashboard, saved plan history and health endpoint.
- Includes the supplied `startup data.csv` as `data/startup_data.csv`.

## Quick start — Windows
1. Extract the ZIP file to a folder.
2. Install Python 3.10 or newer from https://www.python.org/downloads/ (enable “Add Python to PATH”).
3. Double-click `run_windows.bat`. The first run installs required packages and may take a few minutes.
4. Open `http://127.0.0.1:5000` in your browser.
5. Register an account, log in, and open Dashboard.

If the batch file cannot find Python, open Command Prompt in the project folder and run:
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

## macOS / Linux
Run `bash run_mac_linux.sh`, then open `http://127.0.0.1:5000`.

## Dataset and model notes
The supplied CSV contains 923 rows and 49 columns. The app uses the binary `labels` column as a historical outcome proxy and a small set of available startup attributes. It intentionally excludes company names, IDs, status and other identifiers from model features. The three models are evaluated on a stratified holdout split when possible. Metrics can vary by dataset version and installed library version.

**Important:** this dataset is historical and its labels are not a universal definition of startup success. The displayed score is an experimental model estimate, not a calibrated real-world probability or guarantee. Do not use it as investment advice. Real-world validation needs current, representative data and careful testing for leakage, bias and calibration.

## Live startup ecosystem links
The “Explore ecosystem” section links to Y Combinator's company directory, Product Hunt, Wellfound, Startup India and Crunchbase. These are outbound links to current sites; this demo does not scrape or claim to sync live startup records. For a true live integration, obtain official API access, comply with terms/rate limits, and show the last-updated timestamp.

## Assistant
The assistant is rule-based and local by default, so it runs without an API key. You can replace `assistant_answer()` in `app.py` with an LLM API integration later. Keep API keys in environment variables and never commit them to GitHub.

## Project structure
```text
LaunchLens_AI_Startup_Planner/
├── app.py
├── requirements.txt
├── run_windows.bat
├── run_mac_linux.sh
├── README.md
├── data/
│   └── startup_data.csv
├── static/
│   └── style.css
├── templates/
│   ├── base.html
│   ├── landing.html
│   ├── auth.html
│   ├── dashboard.html
│   ├── ideas.html
│   ├── predict.html
│   ├── assistant.html
│   └── error.html
└── instance/   # generated locally at first run
```

## Production checklist
- Set a strong random `SECRET_KEY` environment variable.
- Turn off debug mode (debug is off by default).
- Use HTTPS and a production WSGI server for deployment.
- Add CSRF protection, password reset, email verification, rate limiting and privacy/retention controls before public launch.
- Audit model calibration, fairness, leakage and data rights before presenting scores to real founders.
