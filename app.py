import streamlit as st
import joblib
import pandas as pd
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="CKD Precision", layout="wide", page_icon="🫘")

# ─── Global CSS ─────────────────────────────────────────────────────────────
_CSS_PATH = os.path.join(os.path.dirname(__file__), "ui", "styles.css")
with open(_CSS_PATH, encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ─── Model Load ─────────────────────────────────────────────────────────────
PKL_PATH = os.path.join(os.path.dirname(__file__), 'ckd_pipeline.pkl')
THRESHOLD = 0.40

# The 100-dot grid states "about X in 100 similar patients will decline" — a
# frequency claim that only holds if the model is calibrated. AUROC measures
# ranking, not calibration, and no calibration evidence exists yet
# (model_card.md §5). Hidden until that evidence lands; flip back to True then.
SHOW_FREQUENCY_GRID = False

if not os.path.exists(PKL_PATH):
    st.error(f"找不到模型檔案：{PKL_PATH}")
    st.stop()

try:
    import sklearn, numpy
    st.sidebar.markdown(
        f"<p style='font-family:monospace;font-size:11px;color:#4A6B72;'>sklearn {sklearn.__version__}<br>numpy {numpy.__version__}</p>",
        unsafe_allow_html=True
    )
    pipeline = joblib.load(PKL_PATH)
except ModuleNotFoundError as e:
    st.error(f"缺少模組：**{e.name}** — {e}")
    st.stop()
except Exception as e:
    st.error(f"載入失敗：{type(e).__name__}: {e}")
    st.stop()

features = list(pipeline.feature_names_in_)

# ─── Session State Init ──────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "outdated" not in st.session_state:
    st.session_state.outdated = False
if "view" not in st.session_state:
    st.session_state.view = "clinical"
if "patient_id_val" not in st.session_state:
    st.session_state.patient_id_val = "NEW-PATIENT"

def mark_outdated():
    if st.session_state.result is not None:
        st.session_state.outdated = True

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ckd-header">
  <div class="ckd-header-left">
    <h1>CKD<span>.</span>Precision — <span style="font-weight:300;color:#82999E">AI Risk Assessment</span></h1>
    <p>預測追蹤期間 eGFR 下降超過 7.5% 的可能性 &nbsp;·&nbsp; Clinical decision support only</p>
  </div>
  <div class="ckd-meta">
    <div class="ckd-meta-item"><div class="val">45</div><div class="lbl">Parameters</div></div>
    <div class="ckd-meta-item"><div class="val">0.7871</div><div class="lbl">AUROC</div></div>
    <div class="ckd-meta-item"><div class="val">0.40</div><div class="lbl">Threshold</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── View toggle (only if result exists) ────────────────────────────────────
if st.session_state.result is not None:
    col_v1, col_v2, _ = st.columns([1, 1, 6])
    with col_v1:
        if st.button("Clinical View", type="secondary" if st.session_state.view == "patient" else "primary"):
            st.session_state.view = "clinical"
            st.rerun()
    with col_v2:
        if st.button("Patient View", type="secondary" if st.session_state.view == "clinical" else "primary"):
            st.session_state.view = "patient"
            st.rerun()
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PATIENT VIEW
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.view == "patient" and st.session_state.result is not None:
    r = st.session_state.result
    prob = r["prob_decline"]
    pred = r["pred"]
    risk_cls = "high" if pred == 1 else "low"
    risk_label = "較高風險" if pred == 1 else "較低風險"
    pct_int = round(prob * 100)

    st.markdown(f"""
    <div class="pv-card">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#4A6B72;margin-bottom:4px">
        腎臟功能展望 — Kidney Function Outlook
      </div>
      <div style="font-size:22px;font-weight:500;color:#F3F8F7;margin-bottom:2px">
        {r["patient_id"]}
      </div>
      <div class="pv-pct {risk_cls}">{pct_int}<span style="font-size:32px">%</span></div>
      <div style="font-size:13px;color:#82999E;margin-bottom:16px">
        模型估計：未來追蹤期間，您發生顯著腎功能下降（eGFR 下降 &gt;7.5%）的可能性
      </div>
    """, unsafe_allow_html=True)

    # 100-person frequency icon — see SHOW_FREQUENCY_GRID
    if SHOW_FREQUENCY_GRID:
        filled = pct_int
        dots_html = '<div class="freq-grid">'
        for i in range(100):
            cls = f"filled {risk_cls}" if i < filled else "empty"
            dots_html += f'<div class="freq-dot {cls}" title="{i+1}"></div>'
        dots_html += "</div>"
        st.markdown(dots_html, unsafe_allow_html=True)

        st.markdown(f"""
          <div style="font-size:12px;color:#82999E;">
            每 100 位與您臨床特徵相似的患者中，約 <strong style="color:#F3F8F7">{filled}</strong> 位在追蹤期間可能發生顯著 eGFR 下降。
          </div>
        """, unsafe_allow_html=True)

    # closes .pv-card opened above (kept outside the flag — the card must close
    # whether or not the frequency block renders)
    st.markdown("</div>", unsafe_allow_html=True)

    # Key inputs summary
    st.markdown("""
    <div class="pv-card">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#4A6B72;margin-bottom:12px">
        本次評估使用的主要數值
      </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("基礎 eGFR", f"{r['baseline_egfr']:.1f} mL/min")
    with sc2:
        st.metric("HbA1c", f"{r['nHBA1C']:.1f}%")
    with sc3:
        st.metric("UPCR", f"{r['nUPCR']:.2f}")
    with sc4:
        st.metric("CCI 分數", str(r['cciscore']))

    traj_names = {1: "穩定緩降", 2: "緩慢下降", 3: "輕度下降", 4: "快速下降"}
    st.markdown(f"""
      <div style="margin-top:12px;font-size:12px;color:#82999E;">
        eGFR 軌跡群組：<strong style="color:#F3F8F7">Group {r['traj']} — {traj_names.get(r['traj'],'')}</strong>
        &nbsp;·&nbsp; 性別：<strong style="color:#F3F8F7">{'男' if r['male']==1 else '女'}</strong>
        &nbsp;·&nbsp; 年齡：<strong style="color:#F3F8F7">{r['crage']} 歲</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Active comorbidities & meds
    active_com = r.get("active_comorbidities", [])
    active_med = r.get("active_medications", [])

    st.markdown("""
    <div class="pv-card">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#4A6B72;margin-bottom:10px">
        目前共病與用藥摘要
      </div>
    """, unsafe_allow_html=True)

    if active_com:
        st.markdown(f"<div style='font-size:12px;color:#82999E;margin-bottom:6px'>共病</div>", unsafe_allow_html=True)
        chips = "".join([f"<span style='display:inline-block;padding:3px 10px;margin:3px;border:1px solid #1D3940;border-radius:3px;font-size:12px;color:#5AD1BD;background:rgba(90,209,189,0.08)'>{c}</span>" for c in active_com])
        st.markdown(chips, unsafe_allow_html=True)

    if active_med:
        st.markdown(f"<div style='font-size:12px;color:#82999E;margin:10px 0 6px'>用藥</div>", unsafe_allow_html=True)
        chips = "".join([f"<span style='display:inline-block;padding:3px 10px;margin:3px;border:1px solid #1D3940;border-radius:3px;font-size:12px;color:#82999E'>{m}</span>" for m in active_med])
        st.markdown(chips, unsafe_allow_html=True)

    if not active_com and not active_med:
        st.markdown("<div style='font-size:12px;color:#4A6B72'>無勾選項目</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Education Block (Patient View) ──────────────────────────────────────
    def get_ckd_stage(egfr):
        if egfr >= 90:   return "G1"
        elif egfr >= 60: return "G2"
        elif egfr >= 45: return "G3a"
        elif egfr >= 30: return "G3b"
        elif egfr >= 15: return "G4"
        else:            return "G5"

    ckd_stage  = get_ckd_stage(r["baseline_egfr"])
    is_high    = r["pred"] == 1
    risk_hex   = "#F1C96A" if is_high else "#5AD1BD"

    edu_data = {
        ("G3a", False): {
            "stage_label": "G3a  ·  eGFR 45–59",
            "headline": "目前腎功能風險較低，現在是保護腎臟的最好時機",
            "body": [
                ("🧂", "飲食",     "每日鹽分 &lt; 5g；蛋白質每公斤體重約 0.8g；優先選擇魚、豆腐、蛋等優質蛋白。"),
                ("💊", "用藥",     "避免止痛藥（NSAIDs）及來路不明保健品；確認是否適合 ACEI/ARB 或 SGLT2i。"),
                ("❤️", "血壓血糖", "血壓目標 &lt; 130/80 mmHg；HbA1c 依醫師建議個別控制。"),
                ("🏃", "運動",     "每週至少 150 分鐘中等強度有氧運動（快走、騎車、游泳）。"),
                ("📅", "回診",     "每 6 個月追蹤 eGFR 與尿蛋白。"),
            ]
        },
        ("G3a", True): {
            "stage_label": "G3a  ·  eGFR 45–59",
            "headline": "模型預測腎功能可能持續下降，請盡快與醫師討論強化管理計畫",
            "body": [
                ("🧂", "飲食",       "立即執行低鹽飲食；請腎臟科營養師安排個別蛋白質與礦物質控制計畫。"),
                ("💊", "用藥",       "與醫師確認是否啟用 SGLT2i（DKD 首選）；停止使用 NSAIDs 及可能腎毒性藥物。"),
                ("❤️", "血壓血糖",   "嚴格控制血壓（&lt; 130/80）與血糖；未達標請回診調整用藥。"),
                ("👥", "多學科照護", "建議轉介腎臟科個管師，安排護理師＋藥師＋營養師聯合評估。"),
                ("📅", "回診",       "縮短追蹤至每 3 個月一次。"),
            ]
        },
        ("G3b", False): {
            "stage_label": "G3b  ·  eGFR 30–44",
            "headline": "腎功能已進入 G3b，飲食與用藥需進一步調整",
            "body": [
                ("🧂", "飲食",     "蛋白質降至 0.6–0.8 g/kg；開始限磷（避免加工食品含磷添加劑）；視血液報告決定是否限鉀。"),
                ("💊", "用藥",     "醫師可能依腎功能調整劑量；確認 SGLT2i 是否適用；避免所有腎毒性藥物。"),
                ("🔬", "定期監測", "每次回診檢查：eGFR、尿蛋白、血鉀、血磷、血色素。"),
                ("📖", "了解 KRT", "開始認識腎臟替代治療（血液透析、腹膜透析、腎移植）的基本概念。"),
                ("📅", "回診",     "每 3–6 個月回診；出現水腫、食慾不振、噁心請提早就醫。"),
            ]
        },
        ("G3b", True): {
            "stage_label": "G3b  ·  eGFR 30–44",
            "headline": "腎功能惡化風險顯著，請立即啟動強化照護並討論 KRT 準備",
            "body": [
                ("🧂", "飲食",       "立即安排腎臟科營養師：嚴格低蛋白（0.6 g/kg）、限磷、視報告限鉀。"),
                ("💊", "用藥",       "藥師介入全面用藥審查；優先確認 SGLT2i、ACEI/ARB 是否最適劑量；停止 NSAIDs。"),
                ("🩸", "貧血骨骼",   "若 Hb &lt; 10 g/dL 評估 EPO；檢查血磷與副甲狀腺素（PTH）。"),
                ("📋", "KRT 討論",   "與醫師討論血液透析、腹膜透析、腎移植三種選項，表達個人偏好。"),
                ("👥", "多學科照護", "啟動 MDT：腎臟科醫師＋護理師＋藥師＋營養師＋社工。"),
                ("📅", "回診",       "每 3 個月回診，狀況不穩定時每月追蹤。"),
            ]
        },
        ("G4", False): {
            "stage_label": "G4  ·  eGFR 15–29",
            "headline": "腎功能嚴重下降，請提早準備腎臟替代治療",
            "body": [
                ("🗣️", "確立 KRT 偏好", "與醫療團隊討論：血液透析（每週 3 次）、腹膜透析（每日在家換液）、腎移植，哪種最適合你的生活。"),
                ("🩸", "提早建立通路",   "選擇血液透析者，動靜脈廔管需 2–3 個月成熟，現在評估可避免日後緊急插管。"),
                ("🧂", "飲食全面控制",   "低蛋白、低磷、低鉀、低鈉；水分攝取依醫師指示；安排腎臟科營養師個別諮詢。"),
                ("💊", "貧血骨骼管理",   "定期監測 Hb、PTH、血磷；依需要使用 EPO、磷結合劑、活性維生素 D。"),
                ("💬", "家屬一起參與",   "邀請家人共同了解透析生活；討論預立醫療決定。"),
                ("📅", "回診",           "每 3 個月回診；出現呼吸困難、大量水腫、尿量急劇減少請立即就醫。"),
            ]
        },
        ("G4", True): {
            "stage_label": "G4  ·  eGFR 15–29",
            "headline": "⚠ 腎衰竭風險極高，請本週內安排緊急腎臟科評估",
            "body": [
                ("🚨", "立即行動",     "請本週內與腎臟科醫師預約，告知預測結果，要求加速評估 KRT 準備狀況。"),
                ("🩸", "血管通路優先", "若尚未建立動靜脈廔管，請立即安排外科評估，越早越安全。"),
                ("🗣️", "KRT 確認",    "請在下次回診前閱讀各 KRT 選項說明，並與家人討論意願。"),
                ("🧂", "飲食緊急調整", "立即停止所有高磷、高鉀食物；聯繫腎臟科營養師安排緊急飲食評估。"),
                ("💬", "心理家庭準備", "面對透析感到焦慮是正常的；可請醫師轉介社工或心理師。"),
                ("📅", "回診",         "每月追蹤；出現呼吸困難、意識混亂、無尿、嚴重嘔吐請立即急診。"),
            ]
        },
    }

    content = edu_data.get((ckd_stage, is_high))

    # Fallback: 對於未收錄的分期（G1/G2/G5）或組合，顯示通用衛教而非靜默消失
    if content is None:
        stage_range = {
            "G1": "G1  ·  eGFR ≥ 90", "G2": "G2  ·  eGFR 60–89",
            "G3a": "G3a  ·  eGFR 45–59", "G3b": "G3b  ·  eGFR 30–44",
            "G4": "G4  ·  eGFR 15–29", "G5": "G5  ·  eGFR < 15",
        }
        content = {
            "stage_label": stage_range.get(ckd_stage, ckd_stage),
            "headline": ("模型預測腎功能可能持續下降，請儘快與醫師討論後續管理計畫"
                         if is_high else
                         "目前腎功能風險較低，維持良好生活與用藥習慣並定期追蹤"),
            "body": [
                ("🩺", "與醫師討論", "本分期尚無專屬衛教清單，請攜此結果回診，由醫師依完整病史提供個別建議。"),
                ("🧂", "飲食",       "維持低鹽、均衡飲食；蛋白質攝取請依醫師或營養師指示。"),
                ("💊", "用藥",       "避免自行使用止痛藥（NSAIDs）及來路不明保健品；用藥調整請諮詢醫師。"),
                ("📅", "定期追蹤",   "依醫囑定期檢查 eGFR 與尿蛋白；出現水腫、食慾不振、尿量改變請提早就醫。"),
            ],
        }

    if content:
        st.markdown(f"""
        <div class="pv-card" style="border-color:{risk_hex}44;margin-top:4px;">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:#4A6B72;margin-bottom:4px">
            衛教建議 — Health Education
          </div>
          <div style="font-family:monospace;font-size:12px;color:{risk_hex};margin-bottom:8px">
            {content['stage_label']}
          </div>
          <div style="font-size:14px;font-weight:500;color:#F3F8F7;line-height:1.5;margin-bottom:16px;">
            {content['headline']}
          </div>
        """, unsafe_allow_html=True)

        rows_html = ""
        for icon, title, detail in content["body"]:
            rows_html += f"""
            <details style="border:1px solid #1D3940;border-radius:4px;margin-bottom:6px;background:#0B1B20;">
              <summary style="padding:10px 14px;cursor:pointer;font-size:13px;color:#F3F8F7;list-style:none;display:flex;align-items:center;gap:8px;">
                <span>{icon}</span><span>{title}</span>
              </summary>
              <div style="padding:10px 14px 12px;font-size:13px;color:#82999E;line-height:1.8;border-top:1px solid #1D3940;">
                {detail}
              </div>
            </details>"""
        st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Disclaimer ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      ⚠ 本評估結果為輔助臨床決策之參考，非診斷依據。請由醫療人員結合完整病史、理學檢查及其他檢驗結果進行綜合判斷。
      本工具不提供藥物調整建議，亦不代表任何診斷結論。
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# CLINICAL VIEW
# ════════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([3, 2], gap="medium")

with left_col:

    # ── SEC-01: Patient Demographics ───────────────────────────────────────
    st.markdown("""
    <div class="section-card">
      <div class="section-title"><div class="dot"></div>01 &nbsp; Patient Demographics</div>
      <div class="section-body">
    """, unsafe_allow_html=True)

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        patient_id = st.text_input("Patient ID", value=st.session_state.patient_id_val, key="pid_input")
        st.session_state.patient_id_val = patient_id
    with r1c2:
        crage = st.number_input("Age", value=60, step=1, min_value=18, max_value=120, on_change=mark_outdated)
    with r1c3:
        gender_val = st.radio("Gender", ["Male", "Female"], horizontal=True, on_change=mark_outdated)
        male = 1 if gender_val == "Male" else 0
    with r1c4:
        baseline_egfr = st.number_input("eGFR Baseline (mL/min)", value=60.0, format="%.1f", min_value=0.0, max_value=150.0, on_change=mark_outdated)
        if baseline_egfr < 5:
            st.markdown("<div style='color:#E8715A;font-size:11px;margin-top:2px'>⚠ eGFR 偏低，請確認</div>", unsafe_allow_html=True)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        nHBA1C = st.number_input("HbA1c (%)", value=7.0, format="%.1f", min_value=3.0, max_value=20.0, on_change=mark_outdated)
    with r2c2:
        nGLU = st.number_input("Glucose (mg/dL)", value=100.0, format="%.1f", min_value=30.0, max_value=600.0, on_change=mark_outdated)
    with r2c3:
        nUPCR = st.number_input("UPCR", value=0.5, format="%.2f", min_value=0.0, max_value=50.0, on_change=mark_outdated)
    with r2c4:
        cciscore = st.number_input("CCI Score", value=2, step=1, min_value=0, max_value=20, on_change=mark_outdated)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ── SEC-02: Lab Results ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-card">
      <div class="section-title"><div class="dot"></div>02 &nbsp; Lab Results</div>
      <div class="section-body">
    """, unsafe_allow_html=True)

    lab_fields = [
        ("nALT", "ALT (U/L)"), ("nAST", "AST (U/L)"), ("nBUN", "BUN (mg/dL)"),
        ("nCHOL", "Cholesterol"), ("nTG", "Triglyceride"),
        ("nHDLC", "HDL-C"), ("nLDLC", "LDL-C"),
        ("nGA", "Glycated Albumin"), ("nUACR", "UACR"),
    ]
    lab_values = {}
    lab_row1 = st.columns(5)
    lab_row2 = st.columns(4)
    for i, (key, label) in enumerate(lab_fields):
        row = lab_row1 if i < 5 else lab_row2
        with row[i % 5 if i < 5 else i - 5]:
            lab_values[key] = st.number_input(label, value=None, format="%.1f", key=key, min_value=0.0, placeholder="必填", on_change=mark_outdated)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ── SEC-03: Comorbidities ──────────────────────────────────────────────
    st.markdown("""
    <div class="section-card">
      <div class="section-title"><div class="dot"></div>03 &nbsp; Comorbidities</div>
      <div class="section-body">
    """, unsafe_allow_html=True)

    comorbidity_fields = [
        ("eDM", "Diabetes (DM)"), ("ehtn", "Hypertension"),
        ("echf", "Heart Failure"), ("ecva", "Stroke (CVA)"),
        ("eami", "AMI"), ("ecancer", "Cancer"), ("gerd", "GERD"),
    ]
    comorbidity_values = {}
    com_cols = st.columns(7)
    for i, (key, label) in enumerate(comorbidity_fields):
        with com_cols[i]:
            comorbidity_values[key] = 1 if st.checkbox(label, key=key, on_change=mark_outdated) else 0

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ── SEC-04: Medications ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-card">
      <div class="section-title"><div class="dot"></div>04 &nbsp; Medications</div>
      <div class="section-body">
    """, unsafe_allow_html=True)

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
            medication_values[key] = 1 if st.checkbox(label, key=key, on_change=mark_outdated) else 0

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ── SEC-05: eGFR Trajectory ────────────────────────────────────────────
    st.markdown("""
    <div class="section-card">
      <div class="section-title">
        <div class="dot"></div>05 &nbsp; eGFR Trajectory Group
        <span style="margin-left:auto;font-size:10px;color:#D99A3A;letter-spacing:0.05em">★ 最重要預測因子</span>
      </div>
      <div class="section-body">
    """, unsafe_allow_html=True)

    traj_options_display = {
        1: ("Group 1", "持續低 eGFR", "穩定緩降"),
        2: ("Group 2", "中等 eGFR", "緩慢下降"),
        3: ("Group 3", "較高 eGFR", "輕度下降"),
        4: ("Group 4", "高基線", "快速下降 ⚠"),
    }
    traj_labels = {
        1: "Group 1 — 持續低 eGFR（穩定緩降）",
        2: "Group 2 — 中等 eGFR（緩慢下降）",
        3: "Group 3 — 較高 eGFR（輕度下降）",
        4: "Group 4 — 高基線、快速下降 [高風險]",
    }
    selected_traj = st.radio(
        "選擇軌跡群組",
        options=[1, 2, 3, 4],
        format_func=lambda x: traj_labels[x],
        horizontal=False,
        on_change=mark_outdated,
    )
    traj2 = 1 if selected_traj == 2 else 0
    traj3 = 1 if selected_traj == 3 else 0
    traj4 = 1 if selected_traj == 4 else 0

    st.markdown("</div></div>", unsafe_allow_html=True)

# ─── RIGHT COLUMN: Predict + Result ─────────────────────────────────────────
with right_col:

    st.markdown("""
    <div class="section-card">
      <div class="section-title"><div class="dot"></div>Risk Assessment</div>
      <div class="section-body">
    """, unsafe_allow_html=True)

    run_btn = st.button("▶  RUN PREDICTION", use_container_width=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    if run_btn:
        missing_labs = [label for (key, label) in lab_fields if lab_values.get(key) is None]
        if missing_labs:
            st.error("請填寫以下檢驗值後再執行預測：" + "、".join(missing_labs))
        else:
            st.session_state.outdated = False
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
                proba = pipeline.predict_proba(input_df)[0]
                prob_decline = float(proba[1])
                pred = 1 if prob_decline >= THRESHOLD else 0

                active_com = [label for (key, label) in comorbidity_fields if comorbidity_values.get(key, 0) == 1]
                active_med = [label for (key, label) in medication_fields if medication_values.get(key, 0) == 1]

                st.session_state.result = {
                    "prob_decline": prob_decline,
                    "prob_no_decline": float(proba[0]),
                    "pred": pred,
                    "patient_id": patient_id,
                    "baseline_egfr": baseline_egfr,
                    "nHBA1C": nHBA1C,
                    "nUPCR": nUPCR,
                    "cciscore": cciscore,
                    "crage": crage,
                    "male": male,
                    "traj": selected_traj,
                    "active_comorbidities": active_com,
                    "active_medications": active_med,
                }
                st.rerun()
            except Exception as e:
                st.error(f"預測錯誤：{e}")

    # Result display
    if st.session_state.result is not None:
        r = st.session_state.result
        prob = r["prob_decline"]
        pred = r["pred"]
        risk_cls = "high" if pred == 1 else "low"
        risk_label = "HIGH RISK" if pred == 1 else "LOW RISK"
        risk_icon = "⬆" if pred == 1 else "⬇"
        risk_desc = "eGFR 下降 >7.5% 風險偏高" if pred == 1 else "eGFR 下降 >7.5% 風險偏低"
        pct_str = f"{prob*100:.1f}"

        if st.session_state.outdated:
            st.markdown("""
            <div class="outdated-banner">
              ⚠ 輸入已更動，請重新執行預測以更新結果
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="risk-card {risk_cls}">
          <div class="risk-badge {risk_cls}">{risk_icon} &nbsp; {risk_label}</div>
          <div class="risk-pct {risk_cls}">{pct_str}<span style="font-size:24px;font-weight:400">%</span></div>
          <div class="risk-sublabel">{risk_desc}</div>
        """, unsafe_allow_html=True)

        # Progress bar visual
        bar_color = "var(--warn)" if pred == 1 else "var(--primary)"
        bar_pct = prob * 100
        threshold_pct = THRESHOLD * 100
        st.markdown(f"""
          <div style="position:relative;margin-bottom:6px;">
            <div style="background:#1D3940;border-radius:2px;height:8px;width:100%;overflow:hidden;">
              <div style="background:{bar_color};width:{bar_pct:.1f}%;height:100%;border-radius:2px;transition:width 0.3s;"></div>
            </div>
            <div style="position:absolute;top:-3px;left:{threshold_pct:.0f}%;width:2px;height:14px;background:#82999E;border-radius:1px;" title="Threshold 0.40"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:10px;color:#4A6B72;margin-bottom:16px;">
            <span>0%</span>
            <span style="color:#82999E">▲ threshold 0.40</span>
            <span>100%</span>
          </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Metric breakdown
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("No Decline", f"{r['prob_no_decline']*100:.1f}%")
        with mc2:
            st.metric("Decline", f"{r['prob_decline']*100:.1f}%")

        st.markdown(f"""
        <div style="margin-top:12px;background:#0B1B20;border:1px solid #1D3940;border-radius:4px;padding:12px 14px;">
          <div style="font-size:10px;color:#4A6B72;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Model Info</div>
          <div style="font-size:11px;color:#82999E;line-height:1.8;">
            Patient: <span style="color:#F3F8F7;font-family:monospace">{r['patient_id']}</span><br>
            Parameters: <span style="color:#F3F8F7;font-family:monospace">45</span><br>
            AUROC: <span style="color:#F3F8F7;font-family:monospace">0.7871</span><br>
            Threshold: <span style="color:#F3F8F7;font-family:monospace">0.40</span>
          </div>
          <div style="margin-top:10px;font-size:10px;color:#4A6B72;border-top:1px solid #1D3940;padding-top:8px;">
            ⚠ Clinical decision support only — not a diagnostic tool
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#0B1B20;border:1px solid #1D3940;border-radius:6px;padding:32px 20px;text-align:center;color:#4A6B72;">
          <div style="font-size:28px;margin-bottom:8px;opacity:0.4">◎</div>
          <div style="font-size:13px;">填寫左側欄位後，執行預測</div>
        </div>
        """, unsafe_allow_html=True)
