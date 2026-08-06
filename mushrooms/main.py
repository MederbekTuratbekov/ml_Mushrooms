# mushrooms/main.py

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
import pandas as pd
import joblib
import uvicorn

from schema import MushroomsSchema

BASE_DIR = Path(__file__).parent


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model   = joblib.load(BASE_DIR / "model_rf_Mushrooms.pkl")
    app.state.encoder = joblib.load(BASE_DIR / "encoder_Mushrooms.pkl")
    yield


app = FastAPI(title="Mushroom Classifier", lifespan=lifespan)


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: MushroomsSchema):
    row = pd.DataFrame([data.model_dump(by_alias=True)])

    # то же самое заполнение, что было в обучении (fillna модой из train)
    if "stalk-root" in row.columns:
        row["stalk-root"] = row["stalk-root"].replace("?", "b")  # мода из train

    encoded = app.state.encoder.transform(row)  # тот же OneHotEncoder, что при обучении

    prediction = int(app.state.model.predict(encoded)[0])
    proba      = float(app.state.model.predict_proba(encoded)[0][1])

    return {
        "poisonous":  bool(prediction),  # 1 = poisonous, как в df['class'].map({'p':1,'e':0})
        "message":    "Гриб ядовитый" if prediction == 1 else "Гриб съедобный",
        "probability_poisonous": round(proba * 100, 2),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)