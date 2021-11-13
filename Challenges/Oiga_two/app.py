from typing import Optional

from fastapi import FastAPI

from pydantic import BaseModel

import joblib

import pandas as pd

from tensorflow import keras

app = FastAPI()


class Match(BaseModel):
    round_: int
    localTeam: str
    visitorTeam: str
    division: int    

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/pred/")
def pred_match(match: Match):
    enc = joblib.load("onehotencoder.joblib")
    
    data = pd.DataFrame.from_dict({"localTeam": [match.localTeam], "visitorTeam": [ match.visitorTeam], "round": [match.round_], "division":match.division})
    
    X = pd.DataFrame(enc.transform(X_train.select_dtypes("object")).toarray(), columns= enc.get_feature_names())
    
    X = pd.concat([X.set_index(data.index), data.select_dtypes(int)], axis=1)
    
    scaler = joblib.load("scaler.joblib") 
    
    X_train_tr = scaler.transform(X)
        
    model = keras.models.load_model("best_model.h5")
    
    result = model.predict(X_train_tr)
    
    return {"winner": result}
#     return {"round": match.round_, "localTeam": match.localTeam, "visitorTeam": match.visitorTeam, "division": match.division}