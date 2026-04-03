import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="wide"
)



# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e6f0; }

.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a0a2e 50%, #0a1628 100%);
    border: 1px solid rgba(138,92,246,0.2);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(138,92,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0; line-height: 1.15;
}
.hero-sub { font-size: 1rem; color: #94a3b8; font-weight: 300; margin: 0; }
.hero-badges { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
.badge { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; color: #cbd5e1; font-weight: 500; }
.badge.purple { border-color: rgba(138,92,246,0.4); color: #a78bfa; background: rgba(138,92,246,0.08); }
.badge.blue   { border-color: rgba(56,189,248,0.4);  color: #38bdf8; background: rgba(56,189,248,0.08); }
.badge.green  { border-color: rgba(52,211,153,0.4);  color: #34d399; background: rgba(52,211,153,0.08); }

.section-card { background: #111118; border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; }
.section-label { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #6366f1; margin-bottom: 18px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #f1f0f8; margin-bottom: 20px; }

.result-placed     { background: linear-gradient(135deg, #052e16, #0a3d20); border: 1px solid #16a34a; border-radius: 16px; padding: 28px 32px; text-align: center; }
.result-not-placed { background: linear-gradient(135deg, #1c0707, #2d0f0f); border: 1px solid #dc2626; border-radius: 16px; padding: 28px 32px; text-align: center; }
.result-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; }
.result-sub   { font-size: 0.9rem; color: #94a3b8; margin: 0; }

.metric-tile  { background: #15151f; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 18px 20px; text-align: center; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.75rem; color: #64748b; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.06em; }

.tip-item { background: #13131e; border-left: 3px solid #6366f1; border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 10px; font-size: 0.88rem; color: #cbd5e1; }

.stSlider > div > div > div > div { background: #6366f1 !important; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 14px 0 !important; letter-spacing: 0.04em !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load('best_model.pkl')
    encoders     = joblib.load('label_encoders.pkl')
    target_enc   = joblib.load('target_encoder.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, encoders, target_enc, feature_cols

model, encoders, target_enc, feature_cols = load_artifacts()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🎓 Placement Predictor</p>
  <p class="hero-sub">Enter your academic profile and get an instant placement prediction powered by Machine Learning.</p>
  <div class="hero-badges">
    <span class="badge purple">Decision Tree Model</span>
    <span class="badge blue">50,000 Student Records</span>
    <span class="badge green">100% Accuracy on Test Set</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Step 01</p><p class="section-title">Academic & Personal Details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age         = st.slider("🎂 Age",                   18, 24, 21)
    gender      = st.selectbox("👤 Gender",             ["Male", "Female"])
    degree      = st.selectbox("🎓 Degree",             ["B.Tech", "BCA", "MCA", "B.Sc"])
    branch      = st.selectbox("🏫 Branch",             ["CSE", "ECE", "ME", "Civil", "IT"])
    cgpa        = st.slider("📊 CGPA",                  4.5, 9.8, 7.0, step=0.1, format="%.1f")
with col2:
    internships = st.slider("🏢 Internships",           0, 3, 0)
    projects    = st.slider("🛠 Projects",              1, 6, 2)
    coding      = st.slider("💻 Coding Skills",         1, 10, 5)
    comm        = st.slider("🗣 Communication Skills",  1, 10, 5)
    aptitude    = st.slider("🧠 Aptitude Test Score",   35, 100, 60)
with col3:
    soft        = st.slider("🌟 Soft Skills Rating",    1, 10, 5)
    certs       = st.slider("📜 Certifications",        0, 3, 1)
    backlogs    = st.slider("⚠️ Backlogs",              0, 3, 0)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
predict_btn = st.button("🔍  Predict My Placement", use_container_width=True, type="primary")

if predict_btn:

    # Early eligibility rules
    if comm < 5:
        st.markdown("""
        <div class="result-not-placed">
          <p class="result-title" style="color:#f87171;">❌ NOT PLACED</p>
          <p class="result-sub">Communication Skills must be at least 5 to be eligible for placement.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="tip-item">🗣 Focus on communication — join public speaking clubs, attend mock GDs and interviews regularly.</div>', unsafe_allow_html=True)
        st.stop()

    if backlogs >= 2:
        st.markdown("""
        <div class="result-not-placed">
          <p class="result-title" style="color:#f87171;">❌ NOT PLACED</p>
          <p class="result-sub">Students with 2 or more backlogs are not eligible for placement.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="tip-item">📚 Clear your backlogs first — most companies enforce a strict 0–1 backlog policy.</div>', unsafe_allow_html=True)
        st.stop()

    # Encode categoricals
    try:
        gender_enc = encoders['Gender'].transform([gender])[0]
        degree_enc = encoders['Degree'].transform([degree])[0]
        branch_enc = encoders['Branch'].transform([branch])[0]
    except Exception as e:
        st.warning(f"Encoding issue: {e}")
        gender_enc, degree_enc, branch_enc = 0, 0, 0

    row = pd.DataFrame(
        [[age, gender_enc, degree_enc, branch_enc, cgpa,
          internships, projects, coding, comm, aptitude,
          soft, certs, backlogs]],
        columns=feature_cols
    )

    pred           = model.predict(row)[0]
    proba          = model.predict_proba(row)[0]
    label          = target_enc.inverse_transform([pred])[0]
    placed_pct     = round(proba[1] * 100, 1)
    not_placed_pct = round(proba[0] * 100, 1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Result card
    if label == "Placed":
        st.markdown(f"""
        <div class="result-placed">
          <p class="result-title" style="color:#4ade80;">✅ PLACED!</p>
          <p class="result-sub">Your profile meets placement criteria. Confidence: <strong style="color:#4ade80">{placed_pct}%</strong></p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-not-placed">
          <p class="result-title" style="color:#f87171;">❌ NOT PLACED</p>
          <p class="result-sub">Your profile needs improvement. Confidence: <strong style="color:#f87171">{not_placed_pct}%</strong></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric tiles
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#4ade80">{placed_pct}%</p><p class="metric-label">Placed chance</p></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#f87171">{not_placed_pct}%</p><p class="metric-label">Not placed chance</p></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#a78bfa">{cgpa}</p><p class="metric-label">Your CGPA</p></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#38bdf8">{projects}</p><p class="metric-label">Projects done</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(int(placed_pct), text=f"Placement probability: {placed_pct}%")
    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<p class="section-label">Feature Importance (Random Forest)</p>', unsafe_allow_html=True)
        features    = ['Projects','CGPA','Certifications','Coding Skills','Aptitude Score','Communication','Internships','Backlogs']
        importances = [0.314, 0.178, 0.156, 0.104, 0.093, 0.075, 0.064, 0.009]
        colors      = ['#f87171' if f == 'Backlogs' else '#6366f1' for f in features]
        fig1, ax1   = plt.subplots(figsize=(5, 3.8))
        fig1.patch.set_facecolor('#111118')
        ax1.set_facecolor('#111118')
        bars = ax1.barh(features, importances, color=colors, height=0.6, edgecolor='none')
        ax1.set_xlabel('Importance', color='#64748b', fontsize=8)
        ax1.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.set_xlim(0, 0.38)
        for bar, val in zip(bars, importances):
            ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', color='#94a3b8', fontsize=7.5)
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    with chart_col2:
        st.markdown('<p class="section-label">Your Placement Probability</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
        fig2.patch.set_facecolor('#111118')
        ax2.set_facecolor('#111118')
        wedges, texts, autotexts = ax2.pie(
            [placed_pct, not_placed_pct],
            explode=(0.04, 0),
            colors=['#4ade80', '#f87171'],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': '#111118', 'linewidth': 2},
            textprops={'color': '#e2e8f0', 'fontsize': 10}
        )
        for at in autotexts:
            at.set_fontsize(10); at.set_color('#0a0a0f'); at.set_fontweight('bold')
        ax2.legend(
            handles=[mpatches.Patch(color='#4ade80', label=f'Placed ({placed_pct}%)'),
                     mpatches.Patch(color='#f87171', label=f'Not Placed ({not_placed_pct}%)')],
            loc='lower center', framealpha=0, labelcolor='#94a3b8', fontsize=8
        )
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Tips
    tips = []
    if cgpa < 6.5:       tips.append("📚 Improve your CGPA — aim for at least 7.0")
    if coding < 6:       tips.append("💻 Strengthen coding skills — practice DSA on LeetCode / HackerRank")
    if comm < 7:         tips.append("🗣 Work on communication — attend mock GDs and interviews")
    if internships == 0: tips.append("🏢 Complete at least one internship before placement season")
    if certs == 0:       tips.append("📜 Earn certifications on Coursera, NPTEL, or Google")
    if projects < 2:     tips.append("🛠 Build 2–3 strong projects and host them on GitHub")

    if tips:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">💡 Suggestions to Improve Your Chances</p>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.78rem;padding:16px 0;">
  ML Mini Project &nbsp;·&nbsp; Placement Prediction System &nbsp;·&nbsp; Built with Streamlit & scikit-learn
</div>
""", unsafe_allow_html=True)    
