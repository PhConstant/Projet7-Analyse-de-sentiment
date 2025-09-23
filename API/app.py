# API/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from Source.preprocess_data import preprocess_text_simple
from nltk.tokenize import TweetTokenizer, WordPunctTokenizer, RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer, LancasterStemmer
import logging

app = FastAPI(title="Air Paradis Sentiment API")

class TweetIn(BaseModel):
    text: str

class PredOut(BaseModel):
    positive_proba: float
    positive: bool


class FeedbackIn(BaseModel):
    text: str
    prediction: bool
    user_feedback: str   # "Oui" ou "Non"
    right_answer: bool

# Charger le modèle "Production" depuis le MLflow Model Registry (ou chemin local)
client = MlflowClient(tracking_uri="http://localhost:8080")
registered_model_name = client.get_registered_model('log_regression_model_opt').name
MODEL_URI = f"models:/{registered_model_name}@champion"
model = mlflow.sklearn.load_model(MODEL_URI)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_api")

@app.post("/predict", response_model=PredOut)
def predict(inp: TweetIn):
    # Prétraitement du texte d'entrée
    print("Text input raw :")
    print(inp.text)
    text = preprocess_text_simple(
        text=inp.text, 
        tokenizer=TweetTokenizer().tokenize,
        stem_lem_func=WordNetLemmatizer().lemmatize
        )
    
    print("Texte prétraité :", text)



    proba = float(model.predict_proba([text])[:, 1])  # adapter selon wrapper
    print(f"¨Probablilité calculées {round(proba,5)}")

    return PredOut(positive_proba=proba, positive=proba>=0.5)


@app.post("/feedback")
def feedback(data: FeedbackIn):
    # Logger, sauvegarder en base ou fichier
    logger.info(f"[FEEDBACK] Texte='{data.text}' | Pred={data.prediction} | "
                f"User={data.user_feedback} | Right={data.right_answer}")

    print(f"[FEEDBACK] Texte='{data.text}' | Pred={data.prediction} | User={data.user_feedback}")
    return {"status": "success", "message": "Feedback reçu"}