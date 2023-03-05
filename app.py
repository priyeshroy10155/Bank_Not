# 1. Library import
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

# 2 creat the app object.
app = FastAPI()
pickle_in = open('class.pkl','rb')
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000 For runing (http://127.0.0.1:8000/docs)
@app.post('/predict')
def predict_banknote(data:BankNote):
    data=data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    print('Hello')
    prediction=classifier.predict([[variance, skewness, curtosis, entropy]])
    if(prediction[0]>0.5):
        prediction='Fake note'
    else:
        prediction='It is BankNote'
    return{'prediction': prediction}

if __name__ =='__main__':
    uvicorn.run(app, host='127.0.0.1', port= 8000)

# for runing ## uvicorn app:app --reload  


