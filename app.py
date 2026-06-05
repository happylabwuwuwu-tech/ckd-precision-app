import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="CKD Precision", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 1.5rem !important; }
    .stButton > button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

PKL_PATH = os.path.join(os.path.dirname(__file__), 'best_model_pipeline_thr040.pkl')

if not os.path.exists(PKL_PATH):
    st.error(f"找不到模型檔案：{PKL_PATH}")
    st.stop()

pipeline = joblib.load(PKL_PATH)
features = list(pipeline.feature_names_in_)

# ── Header ──
st.markdown("<h2 style='font-size:22px; font-weight:bold; margin-bottom:2px;'>CKD Precision — AI Risk Assessment</h2>", unsafe_allow_html=True)
st.caption("預測整個追蹤期間 eGFR 下降超過 7.5% 的風險（Threshold = 0.40）")
st.markdown("<hr style='margin:10px 0 18px'>", unsafe_allow_html=True)

# ── SEC-01: Patient Demographics ──
st.markdown("#### Patient Demographics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    patient_id = st.text_input("Patient ID", value="NEW-PATIENT")
with c2:
    crage = st.number_input("Age", value=60, step=1)
with c3:
    male = 1 if st.radio("Gender", ["Male", "Female"], horizontal=True) == "Male" else 0
with c4:
    baseline_egfr = st.number_input("eGFR Baseline (mL/min)", value=60.0, format="%.1f")

c5, c6, c7, c8 = st.columns(4)
with c5:
    nHBA1C = st.number_input("HbA1c (%)", value=7.0, format="%.1f")
with c6:
    nGLU = st.number_input("Glucose", value=100.0, format="%.1f")
with c7:
    nUPCR = st.number_input("UPCR", value=0.5, format="%.2f")
with c8:
    cciscore = st.number_input("CCI Score", value=2, step=1)

st.markdown("<hr style='margin:12px 0'>", unsafe_allow_html=True)

# ── SEC-02: Lab Results ──
st.markdown("#### Lab Results")
lab_cols = st.columns(5)
lab_fields = [
    ("nALT", "ALT"), ("nAST", "AST"), ("nBUN", "BUN"),
    ("nCHOL", "Cholesterol"), ("nTG", "Triglyceride"),
    ("nHDLC", "HDL-C"), ("nLDLC", "LDL-C"),
    ("nGA", "Glycated Albumin"), ("nUACR", "UACR"),
]
lab_values = {}
for i, (key, label) in enumerate(lab_fields):
    with lab_cols[i % 5]:
        lab_values[key] = st.number_input(label, value=20.0, format="%.1f", key=key)

st.markdown("<hr style='margin:12px 0'>", unsafe_allow_html=True)

# ── SEC-03: Comorbidities ──
st.markdown("#### Comorbidities")
comorbidity_fields = [
    ("eDM", "Diabetes (DM)"), ("ehtn", "Hypertension"),
    ("echf", "Heart Failure"), ("ecva", "Stroke (CVA)"),
    ("eami", "AMI"), ("ecancer", "Cancer"), ("gerd", "GERD"),
]
comorbidity_values = {}
com_cols = st.columns(7)
for i, (key, label) in enumerate(comorbidity_fields):
    with com_cols[i]:
        comorbidity_values[key] = 1 if st.checkbox(label, key=key) else 0

st.markdown("<hr style='margin:12px 0'>", unsafe_allow_html=True)

# ── SEC-04: Medications ──
st.markdown("#### Medications")
medication_fields = [
    ("insulin", "Insulin"), ("acei_arb", "ACEI/ARB"), ("statin", "Statin"),
    ("nsaid", "NSAID"), ("diuretics", "Diuretics"), ("SGLT2", "SGLT2i"),
    ("beta_blocker", "Beta Blocker"), ("ccb", "CCB"), ("glp1", "GLP-1"),
    ("Biguanid", "Biguanide"), ("sulfonyl", "Sulfonylurea"), ("Meglitin", "Meglitinide"),
    ("AGIs", "AGIs"), ("TZD", "TZD"), ("ddp4", "DPP-4i"),
    ("fibrate", "Fibrate"), ("anticoagulant", "Anticoagulant"),
    ("antacids", "Antacids"), ("antiplatelet", "Antiplatelet"),
]
medication_values = {}
med_cols = st.columns(5)
for i, (key, label) in enumerate(medication_fields):
    with med_cols[i % 5]:
        medication_values[key] = 1 if st.checkbox(label, key=key) else 0

st.markdown("<hr style='margin:12px 0'>", unsafe_allow_html=True)

# ── SEC-05: eGFR Trajectory ──
st.markdown("#### eGFR Trajectory Group")
st.caption("根據過去 12 個月 eGFR 變化趨勢選擇群組，此為最重要的預測因子。")

traj_options = {
    1: "Group 1 — 持續低 eGFR（穩定緩降）",
    2: "Group 2 — 中等 eGFR（緩慢下降）",
    3: "Group 3 — 較高 eGFR（輕度下降）",
    4: "Group 4 — 高基線、快速下降 [高風險]",
}
selected_traj = st.radio("選擇軌跡群組", options=[1, 2, 3, 4],
                          format_func=lambda x: traj_options[x], horizontal=True)
traj2 = 1 if selected_traj == 2 else 0
traj3 = 1 if selected_traj == 3 else 0
traj4 = 1 if selected_traj == 4 else 0

st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)

# ── Predict ──
btn_col, result_col = st.columns([1, 2])

with btn_col:
    predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

with result_col:
    if predict_btn:
        input_dict = {
            "crage": crage, "male": male,
            "baseline_egfr": baseline_egfr,
            "nHBA1C": nHBA1C, "nGLU": nGLU,
            "nUPCR": nUPCR, "cciscore": cciscore,
            "traj2": traj2, "traj3": traj3, "traj4": traj4,
            **lab_values, **comorbidity_values, **medication_values,
        }

        try:
            input_df = pd.DataFrame([input_dict])[features]
            pred     = pipeline.predict(input_df)[0]
            proba    = pipeline.predict_proba(input_df)[0]
            prob_decline = proba[1]

            if pred == 1:
                st.error(f"**High Risk — eGFR 下降 >7.5% 風險偏高**　（機率 {prob_decline*100:.1f}%）")
            else:
                st.success(f"**Low Risk — eGFR 下降 >7.5% 風險偏低**　（機率 {prob_decline*100:.1f}%）")

            # 機率進度條
            st.markdown("**Risk Probability**")
            st.progress(float(prob_decline))

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("No Decline", f"{proba[0]*100:.1f}%")
            with col_b:
                st.metric("Decline", f"{prob_decline*100:.1f}%")

            st.caption("Analysis based on 45 clinical parameters｜AUROC 0.7871｜Threshold = 0.40")

        except Exception as e:
            st.error(f"預測錯誤：{e}")
    else:
        st.info("點擊左側按鈕執行預測。")
