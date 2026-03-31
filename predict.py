import joblib
import numpy as np
import pandas as pd
import sys

# ─── Load Model & Files Safely ─────────────────────────────
try:
    model        = joblib.load('best_model.pkl')
    encoders     = joblib.load('label_encoders.pkl')
    target_enc   = joblib.load('target_encoder.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
except Exception as e:
    print("❌ Error loading model files.")
    print("Make sure you ran train_model.py first.")
    print(f"Details: {e}")
    sys.exit()


print("=" * 50)
print("  Placement Prediction System")
print("=" * 50)
print("Enter student details below:\n")


# ─── Input Function ───────────────────────────────────────
def get_input(prompt, choices=None, dtype=float, min_val=None, max_val=None):
    while True:
        val = input(prompt).strip()

        # Choice validation
        if choices:
            if val not in choices:
                print(f"  Please enter one of: {choices}")
                continue

        # Type conversion
        try:
            converted = dtype(val)
        except:
            print("  Invalid input, try again.")
            continue

        # Range validation
        if min_val is not None and converted < min_val:
            print(f"  Value must be at least {min_val}.")
            continue

        if max_val is not None and converted > max_val:
            print(f"  Value must be at most {max_val}.")
            continue

        return converted


# ─── User Inputs ──────────────────────────────────────────
age         = get_input("Age (18–24): ", dtype=int, min_val=18, max_val=24)
gender      = get_input("Gender (Male/Female): ", choices=['Male','Female'], dtype=str)
degree      = get_input("Degree (B.Tech/BCA/MCA/B.Sc): ", choices=['B.Tech','BCA','MCA','B.Sc'], dtype=str)
branch      = get_input("Branch (CSE/ECE/ME/Civil/IT): ", choices=['CSE','ECE','ME','Civil','IT'], dtype=str)
cgpa        = get_input("CGPA (4.5–9.8): ", dtype=float, min_val=4.5, max_val=9.8)
internships = get_input("Internships (0–3): ", dtype=int, min_val=0, max_val=3)
projects    = get_input("Projects (1–6): ", dtype=int, min_val=1, max_val=6)
coding      = get_input("Coding Skills (1–10): ", dtype=int, min_val=1, max_val=10)
comm        = get_input("Communication Skills (1–10): ", dtype=int, min_val=1, max_val=10)
aptitude    = get_input("Aptitude Test Score (35–100): ", dtype=int, min_val=35, max_val=100)
soft        = get_input("Soft Skills Rating (1–10): ", dtype=int, min_val=1, max_val=10)
certs       = get_input("Certifications (0–3): ", dtype=int, min_val=0, max_val=3)
backlogs    = get_input("Backlogs (0–3): ", dtype=int, min_val=0, max_val=3)


# ─── Early Rule-Based Check ───────────────────────────────
print()

if comm < 5:
    print("=" * 50)
    print("  Result     : ❌ NOT PLACED")
    print("  Reason     : Communication Skills below 5")
    print("  Confidence : 100.0%")
    print("=" * 50)
    sys.exit()

if backlogs >= 2:
    print("=" * 50)
    print("  Result     : ❌ NOT PLACED")
    print("  Reason     : 2 or more backlogs")
    print("  Confidence : 100.0%")
    print("=" * 50)
    sys.exit()


# ─── Encode Categorical Data ──────────────────────────────
try:
    gender_enc = encoders['Gender'].transform([gender])[0]
    degree_enc = encoders['Degree'].transform([degree])[0]
    branch_enc = encoders['Branch'].transform([branch])[0]
except Exception as e:
    print(f"\n⚠️ Encoding error: {e}")
    print("Using fallback encoding (0).")
    gender_enc, degree_enc, branch_enc = 0, 0, 0


# ─── Create Input Row ─────────────────────────────────────
row = pd.DataFrame([[
    age, gender_enc, degree_enc, branch_enc, cgpa,
    internships, projects, coding, comm, aptitude,
    soft, certs, backlogs
]], columns=feature_cols)

# Ensure correct column order
row = row[feature_cols]


# ─── Prediction ───────────────────────────────────────────
pred = model.predict(row)[0]

# Safe probability handling
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(row)[0]
    conf  = max(proba) * 100
    placed_pct     = proba[1] * 100
    not_placed_pct = proba[0] * 100
else:
    proba = [0.5, 0.5]
    conf  = 50.0
    placed_pct = 50.0
    not_placed_pct = 50.0

label = target_enc.inverse_transform([pred])[0]


# ─── Output Result ────────────────────────────────────────
print("=" * 50)
print(f"  Result     : {'✅ PLACED' if label == 'Placed' else '❌ NOT PLACED'}")
print(f"  Confidence : {conf:.1f}%")
print(f"  Placed     : {placed_pct:.1f}%  |  Not Placed: {not_placed_pct:.1f}%")
print("=" * 50)


# ─── Suggestions System ───────────────────────────────────
tips = []

if cgpa < 6.5:
    tips.append("📚 Improve your CGPA — aim for 7.0+")
if coding < 6:
    tips.append("💻 Practice coding (DSA, LeetCode)")
if comm < 7:
    tips.append("🗣 Improve communication (mock interviews)")
if internships == 0:
    tips.append("🏢 Do at least one internship")
if certs == 0:
    tips.append("📜 Earn certifications (Coursera/NPTEL)")
if projects < 2:
    tips.append("🛠 Build 2–3 strong projects")

if tips and label != 'Placed':
    print("\n  Suggestions to improve:")
    for t in tips:
        print(f"   {t}")
    print()
