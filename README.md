# ❤️ Heart Disease Risk Prediction — Clinical Triage Decision-Support
AIDS Department · Problem-Based Learning (PBL) project · Individual project by **Rohit Dnyaneshwar Mahale** · PRN 2125UDSM2043 · TY AIDS-B

Predicts whether a patient has heart disease from 11 routine clinical measurements and converts the prediction into **Low / Moderate / High risk tiers**, so doctors can prioritise cardiology referrals. Includes a leakage-free scikit-learn pipeline, full evaluation, and a Streamlit demo.

## Headline results (184-patient hold-out set, never used for tuning)
| KPI | Target | Result (95 % CI) |
|---|---|---|
| Recall | ≥ 0.90 | **0.922** (0.865–0.971) |
| Precision | ≥ 0.80 | **0.847** (0.779–0.909) |
| F1 | ≥ 0.85 | **0.883** (0.833–0.924) |
| ROC-AUC | ≥ 0.90 | **0.926** (0.883–0.962) |

Final model: **Random Forest** (chosen by cross-validation only). A high-sensitivity mode (threshold 0.35) reaches recall 0.96. Risk tiers: Low 3.8 % / Moderate 27.5 % / High 85.8 % observed disease rate; the High tier holds 90.6 % of all cases.

## Project structure
```
├── Heart_Disease_Prediction_Final.ipynb   # full analysis, executed with outputs
├── heart_utils.py                         # FeatureEngineer, IQRClipper, pipeline builder (shared by notebook + app)
├── app.py                                 # Streamlit web demo
├── heart.csv                              # dataset (Kaggle: Heart Failure Prediction, fedesoriano)
├── model/heart_model.joblib               # trained pipeline + thresholds + tier boundaries
├── model/results.json                     # all metrics used in the report
├── figures/                               # every plot saved by the notebook
├── Heart_Disease_PBL_Report.docx          # written report
├── Heart_Disease_PBL_Presentation.pptx    # slides
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Heart_Disease_Prediction_Final.ipynb   # Run All (about 2–5 min)
streamlit run app.py                                    # web demo
```
Everything is seeded (`random_state=42`) and reproducible.

## What was improved over the first notebook
| Area | Before | Now |
|---|---|---|
| Problem framing | none | problem statement, stakeholders, KPIs |
| Cholesterol = 0 (172 rows) | filled with the mean (hid a signal) | → NaN, median-imputed **plus** `Chol_Missing` flag (88 % vs 48 % disease) |
| Leakage | imputation/scaling before split | everything inside a `Pipeline`, learned on train only |
| `astype(int)` bug | truncated `Oldpeak` decimals | removed |
| Outliers | ignored | IQR winsorising, tested by ablation |
| Features | one-hot only | 5 clinical engineered features + MI/ANOVA selection |
| Models | 5 on one split | 7 models, repeated 5×3 CV, GridSearch tuning, voting ensemble |
| Metrics | accuracy, F1 | recall, precision, specificity, AUC, PR-AUC, MCC, Brier, bootstrap CIs, confusion matrix, calibration, learning curve |
| Interpretation | none | permutation importance, odds ratios, error and subgroup analysis |
| Business value | none | risk tiers, cost/prevalence analysis, 7 recommendations |
| Deployment | `.pkl` only | Streamlit app with explanation |

## Rubric mapping
| Rubric criterion | Where |
|---|---|
| 1 Problem & business | Notebook §1, Report §1 |
| 2 Dataset understanding | Notebook §2 (data dictionary, quality audit), Report §2 |
| 3 Preprocessing | Notebook §3, `heart_utils.py`, Report §3 |
| 4 EDA | Notebook §4 (plots, Mann–Whitney, χ², correlations), Report §4 |
| 5 Feature engineering / selection | Notebook §5 (ablation, leakage checklist), Report §5 |
| 6 Model selection | Notebook §6, Report §6 |
| 7 Evaluation | Notebook §7, Report §7 |
| 8 Business insights | Notebook §8, Report §8, slides 13–14 |
| 9 Documentation | this README, Report, PPT, app |

## Limitations
Small pooled 1980s–90s dataset (needs local validation) · stress-test features are unavailable at a basic GP visit · only 38 women in the test set · not a medical device.

## Data
Heart Failure Prediction Dataset — fedesoriano, Kaggle (Open Database License).
