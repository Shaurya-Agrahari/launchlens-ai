import os, sqlite3, pickle, secrets
from functools import wraps
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "startup_data.csv"
DB_PATH = ROOT / "instance" / "launchlens.db"
MODEL_PATH = ROOT / "instance" / "model_bundle.pkl"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key-before-deployment")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
DB_PATH.parent.mkdir(exist_ok=True)

FEATURES = ["category_code", "funding_total_usd", "relationships", "funding_rounds",
            "milestones", "age_first_funding_year", "avg_participants", "has_VC", "has_angel"]
NUMERIC = [f for f in FEATURES if f != "category_code"]
CATEGORIES = [
    {"name":"AI workflow copilot for small businesses","category":"AI / SaaS","customer":"SMBs, agencies and local service businesses","problem":"Teams lose time switching between tools and repeating admin work.","mvp":"A focused assistant for one workflow (quotes, follow-ups or reporting) with human approval.","revenue":"Monthly subscription per team","pros":["Clear recurring pain if you pick one niche","Can start with a small web MVP","Recurring revenue is possible"],"cons":["Crowded AI market","API and inference costs need monitoring","Trust and data privacy matter"],"score":86},
    {"name":"Smart inventory forecasting","category":"RetailTech / AI","customer":"Small retailers, pharmacies and D2C sellers","problem":"Stock-outs and overstock tie up money.","mvp":"CSV upload, demand dashboard and low-stock alerts before integrating POS systems.","revenue":"Monthly SaaS tier by number of products","pros":["ROI can be measured in reduced waste","Useful for repeat-purchase businesses","Start with spreadsheet integrations"],"cons":["Historical data can be messy","Forecasts need clear uncertainty ranges","Integrations can slow sales"],"score":82},
    {"name":"Personalised learning coach","category":"EdTech / AI","customer":"College students and training centres","problem":"Learners struggle to turn broad goals into consistent practice.","mvp":"Skill assessment, weekly plan, quizzes and progress analytics for one exam or skill.","revenue":"Freemium, subscription or institute licence","pros":["Easy to test with a small learner group","Progress data can improve the experience","Several potential distribution channels"],"cons":["Retention is difficult","Content quality and accuracy are critical","Many free alternatives exist"],"score":79},
    {"name":"Local business digital growth toolkit","category":"SMB SaaS","customer":"Salons, tutors, repair shops and restaurants","problem":"Small businesses lack time to manage online enquiries and customer follow-up.","mvp":"Mini landing page, enquiry inbox, review reminders and simple analytics.","revenue":"Monthly subscription plus optional setup fee","pros":["Local customer discovery is practical","Value is easy to demonstrate","Can begin with one city or niche"],"cons":["Business owners may need onboarding","Churn can be high","Local sales can take time"],"score":84},
    {"name":"Circular fashion resale assistant","category":"ClimateTech / Marketplace","customer":"Second-hand clothing sellers and budget-conscious buyers","problem":"Listing, pricing and discovering pre-owned items is time-consuming.","mvp":"Listing helper, photo checklist, suggested price range and seller profile.","revenue":"Transaction fee or seller tools subscription","pros":["Supports reuse and affordability","Can start with a narrow category","Community can drive discovery"],"cons":["Marketplace liquidity is hard","Fraud and quality disputes need handling","Logistics can reduce margins"],"score":72},
    {"name":"Accessible appointment and queue manager","category":"HealthTech / SaaS","customer":"Clinics, diagnostic centres and service providers","problem":"Manual scheduling creates missed appointments and long waits.","mvp":"Booking page, reminders, queue view and basic utilisation report.","revenue":"Monthly fee per location","pros":["Problem is concrete and observable","Pilot can be run with one provider","Time savings can be measured"],"cons":["Sensitive data requires care","Existing tools may already be used","Reliability is essential"],"score":77}
]
LINKS = [
    {"name":"Y Combinator — Startup Directory","desc":"Explore companies and their products from a startup accelerator.","url":"https://www.ycombinator.com/companies"},
    {"name":"Product Hunt","desc":"Discover newly launched products and emerging product categories.","url":"https://www.producthunt.com/"},
    {"name":"Wellfound","desc":"Browse startup companies, teams and startup job listings.","url":"https://wellfound.com/"},
    {"name":"Startup India","desc":"Indian startup ecosystem resources, schemes and programmes.","url":"https://www.startupindia.gov.in/"},
    {"name":"Crunchbase","desc":"Research company profiles, industries and funding information.","url":"https://www.crunchbase.com/"}
]

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS saved_plans(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            idea TEXT NOT NULL, category TEXT, score INTEGER,
            created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id))""")
        con.execute("""CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, question TEXT,
            answer TEXT, created_at TEXT NOT NULL)""")

def load_dataset():
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            if not df.empty:
                return df
        except Exception:
            pass
    # Small fallback dataset ensures the app still runs if the CSV is missing.
    rng = np.random.default_rng(42)
    n = 260
    df = pd.DataFrame({
        "category_code": rng.choice(["software","web","mobile","enterprise","ecommerce","biotech"], n),
        "funding_total_usd": rng.lognormal(13, 1.4, n).clip(1000, 100_000_000),
        "relationships": rng.integers(0, 20, n), "funding_rounds": rng.integers(0, 7, n),
        "milestones": rng.integers(0, 8, n), "age_first_funding_year": rng.uniform(0, 8, n),
        "avg_participants": rng.uniform(0, 8, n), "has_VC": rng.integers(0, 2, n),
        "has_angel": rng.integers(0, 2, n)
    })
    logit = -1.0 + .000000025*df.funding_total_usd + .08*df.relationships + .25*df.milestones + .4*df.has_VC
    df["labels"] = (rng.random(n) < 1/(1+np.exp(-logit))).astype(int)
    return df

def train_models():
    df = load_dataset()
    target = "labels" if "labels" in df.columns else ("is_top500" if "is_top500" in df.columns else None)
    usable = [f for f in FEATURES if f in df.columns]
    if target is None or len(usable) < 4:
        return {"available": False, "reason": "Dataset is missing a usable binary target or enough model features."}
    data = df[usable + [target]].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])
    # The supplied dataset's labels field is a historical binary outcome proxy.
    # Keep only binary values and avoid using outcome/status identifiers as features.
    data = data[data[target].isin([0, 1])]
    if len(data) < 30 or data[target].nunique() < 2:
        return {"available": False, "reason": "Not enough labelled rows to train a binary model."}
    X = data[usable].copy()
    y = data[target].astype(int)
    categorical = [f for f in usable if X[f].dtype == "object" or f == "category_code"]
    numeric = [f for f in usable if f not in categorical]
    for c in categorical:
        X[c] = X[c].fillna("unknown").astype(str)
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ], remainder="drop")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=180, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    }
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42)
    results, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        try:
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            prob = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe, "predict_proba") else pred
            try: auc = roc_auc_score(yte, prob)
            except Exception: auc = None
            results.append({"name":name, "accuracy":round(accuracy_score(yte,pred),3),
                            "precision":round(precision_score(yte,pred,zero_division=0),3),
                            "recall":round(recall_score(yte,pred,zero_division=0),3),
                            "f1":round(f1_score(yte,pred,zero_division=0),3),
                            "auc":round(float(auc),3) if auc is not None else None})
            fitted[name] = pipe
        except Exception as e:
            continue
    if not fitted:
        return {"available":False,"reason":"The models could not be trained on this dataset."}
    best = max(results, key=lambda r: r["f1"])
    bundle = {"models":fitted, "results":results, "best_name":best["name"], "features":usable,
              "target":target, "rows":int(len(data)), "positive_rate":round(float(y.mean()),3)}
    try:
        with open(MODEL_PATH,"wb") as f: pickle.dump(bundle,f)
    except Exception: pass
    return {"available":True, **{k:v for k,v in bundle.items() if k!="models"}}

def get_bundle():
    try:
        with open(MODEL_PATH,"rb") as f: return pickle.load(f)
    except Exception:
        train_models()
        try:
            with open(MODEL_PATH,"rb") as f: return pickle.load(f)
        except Exception: return None

def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access your dashboard.", "info")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapped

def safe_num(value, default=0):
    try: return float(value)
    except (TypeError, ValueError): return default

def estimate_score(form):
    """Model-derived probability when available; otherwise transparent heuristic estimate."""
    bundle = get_bundle()
    if bundle and bundle.get("models"):
        features = bundle["features"]
        row = {}
        for f in features:
            if f == "category_code": row[f] = form.get("category_code", "software")
            else: row[f] = safe_num(form.get(f), 0)
        frame = pd.DataFrame([row], columns=features)
        model = bundle["models"][bundle["best_name"]]
        try:
            prob = float(model.predict_proba(frame)[0][1])
            return int(round(100*prob)), "Trained ML model", bundle
        except Exception:
            pass
    # Fallback is a planning score, not a statistically calibrated probability.
    score = 35 + min(18, safe_num(form.get("relationships"))*1.2) + min(15, safe_num(form.get("milestones"))*2)
    score += min(15, safe_num(form.get("funding_total_usd"))/2_000_000)
    score += min(8, safe_num(form.get("funding_rounds"))*1.3)
    score += 5 if form.get("has_VC") == "1" else 0
    score += 3 if form.get("has_angel") == "1" else 0
    return int(max(5,min(95,round(score)))), "Rule-based planning estimate (ML model unavailable)", bundle

def make_recommendations(form, score):
    recs = []
    funding = safe_num(form.get("funding_total_usd"))
    relationships = safe_num(form.get("relationships"))
    milestones = safe_num(form.get("milestones"))
    rounds = safe_num(form.get("funding_rounds"))
    if funding < 100000: recs.append(("Validate before spending","Interview 15–20 target customers and test a landing page before committing significant capital.","High"))
    if relationships < 5: recs.append(("Build founder relationships","Schedule conversations with potential customers, mentors and domain experts each week.","High"))
    if milestones < 2: recs.append(("Define measurable milestones","Set 30/60/90-day targets for interviews, MVP usage, retention and paid pilots.","High"))
    if rounds < 2: recs.append(("Plan funding around evidence","Estimate 12 months of runway and choose bootstrapping, grants or fundraising based on traction.","Medium"))
    if form.get("has_VC") != "1" and form.get("has_angel") != "1": recs.append(("Map funding options","Compare bootstrapping, incubators, government schemes and angel funding; do not raise money before you understand the need.","Medium"))
    if score < 50: recs.append(("Run a focused experiment","Choose one customer segment and test the riskiest assumption in a two-week experiment.","High"))
    recs.append(("Create a simple KPI dashboard","Track weekly active users, activation, retention, conversion, gross margin and cash runway.","Medium"))
    return recs[:6]

def assistant_answer(q):
    s = (q or "").lower()
    if any(w in s for w in ["idea","what should","suggest","business idea"]):
        return "Start with a specific customer problem, not just a technology. Current example directions: AI workflow tools for one niche, inventory forecasting for small retailers, local-business digital tools, or personalised learning. Interview 10–20 potential users before building."
    if any(w in s for w in ["fund","invest","money","funding","vc","angel"]):
        return "Funding paths include bootstrapping, customer revenue, incubators, grants, angel investors and venture capital. Prepare a use-of-funds plan, runway estimate, traction evidence and a clear explanation of the market. Funding is not a substitute for customer validation."
    if any(w in s for w in ["market","competitor","competition","customer","validate"]):
        return "Define one ideal customer profile, list 5–10 alternatives competitors use today, conduct 15–20 problem interviews, and test a landing page or clickable prototype. Ask about recent behaviour and spending rather than only asking whether people like the idea."
    if any(w in s for w in ["pitch","pitch deck","investor"]):
        return "A concise pitch deck usually covers: problem, target customer, solution/demo, market, business model, traction, competition, go-to-market, team, financial assumptions and funding ask. Keep every number traceable to a source or clearly label it as an estimate."
    if any(w in s for w in ["mvp","product","build","technology"]):
        return "Build the smallest version that tests the riskiest assumption. Start with one user segment and one core workflow, add analytics and privacy basics, then run a small pilot. Avoid adding features until users repeatedly use the core feature."
    if any(w in s for w in ["success score","probability","score","prediction"]):
        return "The score is a model-based estimate from historical dataset patterns, not a guarantee of future success. It may reflect dataset bias and may not generalise to your market, country or sector. Use it alongside customer interviews, unit economics, founder execution and current market research."
    if any(w in s for w in ["legal","company","register","compliance","privacy"]):
        return "Choose a business structure after considering ownership, liability, taxes, fundraising plans and compliance. Requirements vary by jurisdiction; check official government guidance or consult a qualified professional before making legal or tax decisions."
    if any(w in s for w in ["marketing","customer acquisition","growth","sales"]):
        return "Pick one channel where your target customers already spend time. Test a small campaign or direct outreach, measure qualified leads and conversion, and calculate acquisition cost. Keep the message specific to the problem you solve."
    return "I can help with idea validation, MVP planning, customer research, competitor mapping, business models, marketing, funding, pitch decks and interpreting the success score. Tell me your startup idea, target customer and current stage for more specific guidance."

@app.route("/")
def home():
    return render_template("landing.html", ideas=CATEGORIES[:3])

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name","").strip()
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        if not name or not email or len(password) < 8:
            flash("Enter your name, a valid email and a password of at least 8 characters.", "error")
        else:
            try:
                with db() as con:
                    con.execute("INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)",
                                (name,email,generate_password_hash(password),datetime.utcnow().isoformat()))
                flash("Account created. You can now log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That email is already registered. Please log in.", "error")
    return render_template("auth.html", mode="register")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        with db() as con: user = con.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
        if user and check_password_hash(user["password_hash"],password):
            session.clear(); session["user_id"] = user["id"]; session["name"] = user["name"]
            return redirect(url_for("dashboard"))
        flash("Email or password is incorrect.", "error")
    return render_template("auth.html", mode="login")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/dashboard")
@login_required
def dashboard():
    df = load_dataset()
    counts = {}
    if "category_code" in df.columns:
        counts = df["category_code"].fillna("unknown").astype(str).value_counts().head(7).to_dict()
    bundle = get_bundle()
    metrics = bundle.get("results",[]) if bundle else []
    with db() as con:
        saved = con.execute("SELECT * FROM saved_plans WHERE user_id=? ORDER BY id DESC LIMIT 5",(session["user_id"],)).fetchall()
    return render_template("dashboard.html", ideas=CATEGORIES, links=LINKS, category_counts=counts,
                           model_results=metrics, model_bundle=bundle, saved=saved)

@app.route("/ideas")
@login_required
def ideas():
    return render_template("ideas.html", ideas=CATEGORIES, links=LINKS)

@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    result = None
    form = {"category_code":"software","funding_total_usd":"25000","relationships":"4","funding_rounds":"1",
            "milestones":"1","age_first_funding_year":"1","avg_participants":"1.5","has_VC":"0","has_angel":"0"}
    if request.method == "POST":
        form.update({k:request.form.get(k,form.get(k,"")) for k in FEATURES})
        score, method, bundle = estimate_score(form)
        recs = make_recommendations(form, score)
        result = {"score":score,"method":method,"recommendations":recs,
                  "level":"Promising signals" if score >= 70 else ("Needs validation" if score >= 50 else "High uncertainty"),
                  "note":"This is an experimental decision-support estimate, not a calibrated real-world probability or guarantee."}
        with db() as con:
            con.execute("INSERT INTO saved_plans(user_id,idea,category,score,created_at) VALUES(?,?,?,?,?)",
                        (session["user_id"],request.form.get("idea_name","My startup"),form.get("category_code"),score,datetime.utcnow().isoformat()))
    return render_template("predict.html", form=form, result=result)

@app.route("/assistant", methods=["GET","POST"])
@login_required
def assistant():
    answer = None
    question = ""
    if request.method == "POST":
        question = request.form.get("question","").strip()
        if question:
            answer = assistant_answer(question)
            with db() as con:
                con.execute("INSERT INTO feedback(user_id,question,answer,created_at) VALUES(?,?,?,?)",
                            (session["user_id"],question,answer,datetime.utcnow().isoformat()))
    return render_template("assistant.html", question=question, answer=answer)

@app.route("/api/assistant", methods=["POST"])
@login_required
def assistant_api():
    data = request.get_json(silent=True) or {}
    q = str(data.get("question",""))[:1500]
    return jsonify({"answer":assistant_answer(q)})

@app.route("/health")
def health():
    return jsonify({"status":"ok","app":"LaunchLens AI","timestamp":datetime.utcnow().isoformat()})

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="We couldn't find that page."), 404

init_db()
# Train once at startup; failure is handled gracefully so the dashboard still opens.
try: train_models()
except Exception as e: print("Model training skipped:", e)

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG","0") == "1", host="127.0.0.1", port=int(os.environ.get("PORT",5000)))
