import joblib
import numpy as np
import pandas as pd

# Load saved model and encoders
model        = joblib.load('best_model.pkl')
encoders     = joblib.load('label_encoders.pkl')
target_enc   = joblib.load('target_encoder.pkl')
feature_cols = joblib.load('feature_cols.pkl')


print("=" * 50)
print("  Placement Prediction System")
print("=" * 50)
print("Enter student details below:\n")

def get_input(prompt, choices=None, dtype=float, min_val=None, max_val=None):
    while True:
        val = input(prompt).strip()
        if choices:
            if val not in choices:
                print(f"  Please enter one of: {choices}")
                continue
        try:
            converted = dtype(val)
        except:
            print("  Invalid input, try again.")
            continue
        if min_val is not None and converted < min_val:
            print(f"  Value must be at least {min_val}.")
            continue
        if max_val is not None and converted > max_val:
            print(f"  Value must be at most {max_val}.")
            continue
        return converted

age         = get_input("Age (e.g. 21): ",                    dtype=int,   min_val=18,  max_val=24)
gender      = get_input("Gender (Male/Female): ",             choices=['Male','Female'],  dtype=str)
degree      = get_input("Degree (B.Tech/BCA/MCA/B.Sc): ",    choices=['B.Tech','BCA','MCA','B.Sc'], dtype=str)
branch      = get_input("Branch (CSE/ECE/ME/Civil/IT): ",     choices=['CSE','ECE','ME','Civil','IT'], dtype=str)
cgpa        = get_input("CGPA (4.5-9.8): ",                   dtype=float, min_val=4.5, max_val=9.8)
internships = get_input("Internships (0-3): ",                dtype=int,   min_val=0,   max_val=3)
projects    = get_input("Projects (1-6): ",                   dtype=int,   min_val=1,   max_val=6)
coding      = get_input("Coding Skills (1-10): ",             dtype=int,   min_val=1,   max_val=10)
comm        = get_input("Communication Skills (1-10): ",      dtype=int,   min_val=1,   max_val=10)
aptitude    = get_input("Aptitude Test Score (35-100): ",     dtype=int,   min_val=35,  max_val=100)
soft        = get_input("Soft Skills Rating (1-10): ",        dtype=int,   min_val=1,   max_val=10)
certs       = get_input("Certifications (0-3): ",             dtype=int,   min_val=0,   max_val=3)
backlogs    = get_input("Backlogs (0-3): ",                   dtype=int,   min_val=0,   max_val=3)

# ── Early eligibility check (rules found in data) ────────────────────────────

print()
if comm < 5:
    print("=" * 50)
    print("  Result     : ❌ NOT PLACED")
    print("  Reason     : Communication Skills below 5")
    print("  Confidence : 100.0%")
    print("=" * 50)
    exit()

if backlogs >= 2:
    print("=" * 50)
    print("  Result     : ❌ NOT PLACED")
    print("  Reason     : 2 or more backlogs")
    print("  Confidence : 100.0%")
    print("=" * 50)
    exit()

# ── Encode categorical features ───────────────────────────────────────────────

try:
    gender_enc = encoders['Gender'].transform([gender])[0]
    degree_enc = encoders['Degree'].transform([degree])[0]
    branch_enc = encoders['Branch'].transform([branch])[0]
except Exception as e:
    print(f"\n⚠️  Encoding error: {e}")
    print("   Using fallback encoding (0). Predictions may be less accurate.")
    gender_enc, degree_enc, branch_enc = 0, 0, 0

# ── Build feature row and predict ─────────────────────────────────────────────

row = pd.DataFrame(
    [[age, gender_enc, degree_enc, branch_enc, cgpa,
      internships, projects, coding, comm, aptitude,
      soft, certs, backlogs]],
    columns=feature_cols
)

pred  = model.predict(row)[0]
proba = model.predict_proba(row)[0]
label = target_enc.inverse_transform([pred])[0]
conf  = max(proba) * 100

placed_pct     = proba[1] * 100
not_placed_pct = proba[0] * 100

# ── Display result ─────────────────────────────────────────────────────────────

print("=" * 50)
print(f"  Result     : {'✅ PLACED' if label == 'Placed' else '❌ NOT PLACED'}")
print(f"  Confidence : {conf:.1f}%")
print(f"  Placed     : {placed_pct:.1f}%  |  Not Placed: {not_placed_pct:.1f}%")
print("=" * 50)

# ── Friendly tips based on weak areas ─────────────────────────────────────────

tips = []
if cgpa < 6.5:
    tips.append("📚 Improve your CGPA — aim for at least 7.0")
if coding < 6:
    tips.append("💻 Strengthen coding skills — practice DSA regularly")
if comm < 7:
    tips.append("🗣  Work on communication skills — join clubs or mock GDs")
if internships == 0:
    tips.append("🏢 Try to complete at least one internship")
if certs == 0:
    tips.append("📜 Earn certifications (Coursera, NPTEL, etc.)")
if projects < 2:
    tips.append("🛠  Build more projects — at least 2-3 for your resume")

if tips and label != 'Placed':
    print("\n  Suggestions to improve your chances:")
    for tip in tips:
        print(f"    {tip}")
    print()
