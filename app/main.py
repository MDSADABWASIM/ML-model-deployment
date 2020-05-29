import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import pickle


app = FastAPI()


#Intialize pickled files
model = pickle.load(open('app/model/model.pickel','rb'))

class Data(BaseModel):
    daily_time_on_site: float 
    age: int 
    area_income: float
    daily_internet_usage: float 
    male: int

@app.post('/predict')
def predict(data: Data):
    try:
        data_dict = data.dict()
        to_predict = list(data_dict.values())
        to_predict= np.array(to_predict)
        #Reshaping it to provide in 1 row and unknown number of columns
        prediction = model.predict(to_predict.reshape(1,-1))
        return {"prediction": int(prediction[0])}
    except:
        log.error('something went wrong')
        return {"prediction":"error"}
