# Model Card — CKD Precision (eGFR decline risk)

> 骨架版。標「✅ 已查證」的內容是直接從 `ckd_pipeline.pkl` 反序列化讀出、可重現；
> 標「🔲 待補」的內容**無法**從模型檔得知，必須由訓練者／原始 notebook 提供。
> **在待補欄位填上真實數字前，本工具仍應視為 demo，不得進入正式臨床流程。**

最後更新：2026-07-13

---

## 1. 模型概要

| 項目 | 內容 | 來源 |
|---|---|---|
| 名稱 | CKD Precision — AI Risk Assessment | app |
| 預測目標 | 追蹤期間 eGFR 下降 > 7.5% 的機率（二元分類） | app |
| 正類定義（label = 1） | eGFR 下降 > 7.5% | app 文案 |
| 決策閾值 | 0.40（`prob ≥ 0.40` 判為高風險） | app（由來見 §5） |
| 輸出 | `predict_proba` → P(decline) ∈ [0, 1] | ✅ 已查證 |
| 用途 | 臨床決策輔助，**非診斷依據** | app disclaimer |

## 2. 架構 ✅ 已查證

sklearn `Pipeline`，三段：

1. `SimpleImputer(strategy="median")` — 缺值以**訓練集中位數**填補
2. `StandardScaler`
3. `GradientBoostingClassifier(n_estimators=300, max_depth=5)`，`classes_ = [0, 1]`

輸入特徵數：**45**。

> ⚠ 缺值處理備註：因 imputer 為 median 策略，任何未提供的特徵會被補成訓練中位數
> （例：`nCHOL`→155、`nHBA1C`→6.54、`nUPCR`→102.4）。前端目前要求檢驗值必填，
> 即是為了避免「未填欄位被靜默補成中位數、卻以高信心百分比呈現」。若未來改為允許缺值，
> **必須在 UI 明確標示哪些欄位是模型補值**。

## 3. 輸入特徵（45）✅ 已查證

順序即 `pipeline.feature_names_in_`（前端建構 DataFrame 時依此排序）：

- 人口學：`crage`, `male`
- 共病：`echf`, `ecva`, `eDM`, `ecancer`, `eami`, `ehtn`, `gerd`
- 用藥：`insulin`, `glp1`, `Biguanid`, `sulfonyl`, `Meglitin`, `AGIs`, `TZD`, `ddp4`, `SGLT2`, `diuretics`, `ccb`, `beta_blocker`, `acei_arb`, `statin`, `fibrate`, `anticoagulant`, `nsaid`, `antacids`, `antiplatelet`
- 綜合指數：`cciscore`
- 檢驗：`nALT`, `nAST`, `nBUN`, `nCHOL`, `nGA`, `nGLU`, `nHBA1C`, `nHDLC`, `nLDLC`, `nTG`, `nUACR`, `nUPCR`
- eGFR 軌跡群組（one-hot，Group 1 為基準）：`traj2`, `traj3`, `traj4`
- 基礎腎功能：`baseline_egfr`

> 🔲 待補：每個特徵的單位、量測時點（baseline？）、以及 `traj` 群組是如何從縱向 eGFR 推導出來的。

## 4. 訓練資料 🔲 待補

以下**全部**無法從模型檔得知，需訓練者提供：

- [ ] 資料來源世代（單/多中心？院所？地區？）
- [ ] 收案期間與 inclusion / exclusion 準則
- [ ] 樣本數（train / validation / test 各多少）
- [ ] 類別不平衡比例（decline vs no-decline）
- [ ] 追蹤期間定義與 eGFR 量測頻率
- [ ] 缺值比例（各特徵）與遺漏機制假設

## 5. 效能與驗證 🔲 待補

app header 顯示 **AUROC = 0.7871**，但**來源與切分未知**：

- [ ] AUROC 0.7871 是在哪個資料集上測的？（internal CV / hold-out / 外部驗證）
- [ ] 完整指標：AUPRC、sensitivity / specificity、PPV / NPV @ threshold 0.40、confusion matrix
- [ ] **subgroup 效能**（本專案最關鍵的一項）：依 CKD 分期（尤其 G4/G5）、年齡、性別、有無 DM 分層的 AUROC / 校準——筛檢工具在最脆弱族群是否可靠，這裡要有數字
- [ ] 校準（calibration curve / Brier score）——因為 UI 直接把機率當「每 100 人中 X 人」呈現，校準不良會直接誤導病人
- [ ] 閾值 0.40 的由來（Youden index？臨床指定 sensitivity？成本考量？）

## 6. 限制與已知風險 🔲 待補（範例，需訓練者確認）

- [ ] 外推性：對訓練世代以外的族群未驗證
- [ ] `traj` 群組需縱向 eGFR 才能決定，單次就診可能無法取得
- [ ] 前端 `patient_id` 直插 HTML（注入面）、header 硬編碼 `45 / 0.7871`（換模型即失真）——工程債，另案處理

## 7. 重現性 ✅ 已查證（部分）

| 套件 | 訓練/序列化相容版本 | 備註 |
|---|---|---|
| scikit-learn | `==1.8.0` | 已於 requirements 釘死；pickle 對此版本敏感 |
| numpy | `==2.5.1` | 已釘死為驗證版本 |
| joblib | `>=1.3.0` | 載入用 |

> 🔲 待補：實際**訓練當下**的 sklearn / numpy / joblib 版本（若與上表不同，反序列化可能有風險）。

## 8. 維護

- 模型檔：`ckd_pipeline.pkl`（GradientBoosting，1.3 MB）
- 回歸測試：`tests/test_predictor.py`, `tests/test_app.py`
- 每次更換 `ckd_pipeline.pkl` 時，§1/§2/§5 的數字（含 app header 的 AUROC / 參數數）都必須同步更新。
