from schema import MushroomsSchema
from columns import all_columns
from fastapi import FastAPI
from pathlib import Path
import pandas as pd
import joblib
import uvicorn


BASE_DIR = Path(__file__).parent


model = joblib.load(BASE_DIR / 'model_rf_Mushrooms.pkl')
scaler = joblib.load(BASE_DIR / 'scaler_Mushrooms.pkl')

app = FastAPI()


@app.post('/predict')
async def predict(data: MushroomsSchema):
    data_dict = data.model_dump(by_alias=True)
    input_data = pd.DataFrame([data_dict])

    if 'stalk-root' in input_data.columns:
        input_data['stalk-root'] = input_data['stalk-root'].replace('?', 'b')

    encoded_df = pd.DataFrame(0, index=[0], columns=all_columns)

    for col in input_data.columns:
        val = input_data.loc[0, col]
        dummy_col = f"{col}_{val}"
        if dummy_col in encoded_df.columns:
            encoded_df.loc[0, dummy_col] = 1

    input_scaled = scaler.transform(encoded_df.to_numpy())

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    return {
        "poisonous": bool(prediction),
        "probability": float(proba)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)