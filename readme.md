# Wild Mushroom Toxicity Classifier API

> Predicts whether a wild mushroom is edible or poisonous from observable physical traits — a safety-screening prototype for foragers, food safety labs, and field apps.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()
[![F1](https://img.shields.io/badge/F1--weighted-0.99-brightgreen)]()
[![Recall--poisonous](https://img.shields.io/badge/Recall(poisonous)-0.98-yellow)]()

---

## Business Problem

Accidental mushroom poisoning causes thousands of emergency hospitalizations annually, with fatality rates as high as 90% for certain species. Field identification guides rely on expert knowledge and are prone to human error under pressure. This model automates toxicity screening from 22 observable morphological features, aiming to flag dangerous specimens consistently.

---

## Project Structure

```
ml_Mushrooms/
├── .gitignore
├── readme.md
├── requirements.txt
└── mushrooms/
    ├── Mushrooms.ipynb            # EDA + model comparison
    ├── main.py                    # FastAPI inference service
    ├── schema.py                  # Pydantic Literal validation, 22 fields
    ├── columns.py                 # 90-column OHE schema for alignment
    ├── model_rf_Mushrooms.pkl     # deployed model (Random Forest)
    ├── encoder_Mushrooms.pkl      # fitted OneHotEncoder
    ├── datasets/
    └── Text.txt
```

---

## Demo

**POST** `http://127.0.0.1:8000/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cap-shape": "x", "cap-surface": "s", "cap-color": "n",
    "bruises": "t", "odor": "p", "gill-attachment": "f",
    "gill-spacing": "c", "gill-size": "n", "gill-color": "k",
    "stalk-shape": "e", "stalk-root": "e",
    "stalk-surface-above-ring": "s", "stalk-surface-below-ring": "s",
    "stalk-color-above-ring": "w", "stalk-color-below-ring": "w",
    "veil-type": "p", "veil-color": "w", "ring-number": "o",
    "ring-type": "p", "spore-print-color": "k",
    "population": "s", "habitat": "u"
  }'
```

**Response** (actual shape from `main.py`):
```json
{
  "poisonous": true,
  "message": "Гриб ядовитый",
  "probability_poisonous": 97.4
}
```

> `poisonous: true` = do not eat. Target encoding: `p → 1` (poisonous), `e → 0` (edible), matching `df['class'].map({'p': 1, 'e': 0})` in the notebook.

---

## Results

| Model | Test Accuracy | F1 (weighted) | Precision (poisonous) | Recall (poisonous) |
|---|---|---|---|---|
| Logistic Regression | 1.000 | 1.00 | 1.00 | 1.00 |
| Decision Tree (`max_depth=5`) | 0.996 | 1.00 | 0.99 | 1.00 |
| **Random Forest (`max_depth=5`)** ✅ deployed | 0.991 | 0.99 | 1.00 | **0.98** |

*(1,625-row test set: 843 edible / 782 poisonous)*

**Deployed model is Random Forest, not the top scorer.** Logistic Regression scored a perfect 1.00 across every metric on this split; Random Forest — the model actually saved and served via `main.py` — has **precision 1.00 / recall 0.98** on the poisonous class, meaning **~2% of poisonous test samples were misclassified as edible (false negatives)**. For a food-safety use case that's the metric that matters most, so it's called out explicitly rather than folded into an aggregate "99% accuracy" headline.

---

## Dataset

- **Source:** UCI Mushroom Dataset (Kaggle mirror)
- **Size:** 8,124 records
- **Features:** 22 categorical morphological features → 90 binary columns after One-Hot Encoding
- **Class balance:** 51.8% edible / 48.2% poisonous — near-balanced

---

## Approach

1. **EDA** — value counts and distribution plots for `class`, `cap-shape`, `cap-color`; confirmed fully categorical feature set
2. **Missing values** — `stalk-root` uses literal `'?'` for unknowns → replaced with `NaN`, then imputed with the **training-set mode**, applied identically to `x_test`
3. **Split before encoding** — `train_test_split` run on raw categorical `x`/`y` first, encoding fitted only on `x_train` afterward — avoids leaking test-set category frequencies into the encoder
4. **Feature engineering** — `OneHotEncoder` fitted on `x_train`, saved as `encoder_Mushrooms.pkl`; full 90-column schema saved separately in `columns.py` for inference alignment
5. **Model training** — Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=5`), Random Forest (`n_estimators=100`, `max_depth=5`), all `random_state=42`
6. **Evaluation** — Accuracy, weighted Precision/Recall/F1, full `classification_report`, with explicit attention to Recall on the poisonous class
7. **Deployment** — FastAPI endpoint with full Pydantic `Literal` validation on all 22 fields; `'?'` imputation and OHE alignment reproduced at inference time

---

## Key Challenges & Solutions

**`stalk-root='?'` is a string, not a missing-value marker**
`pd.get_dummies`/`OneHotEncoder` would otherwise create a `stalk-root_?` category with no equivalent in a clean training run → replaced `'?'` with `NaN`, imputed with the **train-set mode only** (not test-set, not full-dataset) → same imputation logic reproduced in `main.py` at inference time.

**Split-before-encode to avoid leakage**
Encoding categorical features before splitting risks the encoder/scaler seeing test-set category frequencies → `train_test_split` performed on raw `x`/`y` first, `OneHotEncoder` fit only on `x_train` → confirmed no leakage.

**Recall on the poisonous class is the metric that matters, and it isn't perfect**
Aggregate accuracy (99%) can mask a costly error type in a safety context → checked per-class recall specifically → the deployed Random Forest has 0.98 recall on poisonous, i.e. a small but non-zero false-negative rate, documented here instead of rounded away.

---

## Tech Stack

| Category   | Tools                                      |
|------------|----------------------------------------------|
| Language   | Python 3.11                                |
| ML         | scikit-learn, joblib                       |
| Data       | pandas, NumPy                              |
| Viz        | Matplotlib, Seaborn                        |
| API        | FastAPI, Uvicorn, Pydantic                 |
| Validation | Pydantic `Literal` types for all 22 fields |
| Deployment | Local / Docker-ready                       |

---

## Deployment

The trained Random Forest model is served via **FastAPI**. `main.py` validates all 22 morphological inputs via Pydantic `Literal` constraints, applies the same `'?'` → mode imputation used during training, reconstructs the full 90-column OHE feature vector via the fitted `encoder_Mushrooms.pkl`, and returns a binary toxicity prediction with a probability score.

```
POST /predict
```

**To run locally:**
```bash
python main.py
# API at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/mushroom-toxicity-api
cd mushroom-toxicity-api
pip install -r requirements.txt
```

```bash
jupyter notebook mushrooms/Mushrooms.ipynb
```

```bash
python mushrooms/main.py
```

---

## Next Steps

- [ ] Investigate the 0.98 recall gap on the poisonous class (which specific specimens are misclassified — check confusion matrix by species/odor)
- [ ] Consider deploying Logistic Regression instead, given its perfect recall on this split, or ensemble both
- [ ] Add a configurable decision threshold to trade precision for recall on the poisonous class (safety-critical use case favors erring toward "poisonous")

---

[//]: # (## Author)

[//]: # ()
[//]: # (**[Your Name]** — [LinkedIn]&#40;https://linkedin.com&#41; | [GitHub]&#40;https://github.com&#41; | [Kaggle]&#40;https://kaggle.com&#41;)