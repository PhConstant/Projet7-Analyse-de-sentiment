# API/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
from Source.preprocess_data import preprocess_data_embedding
import tensorflow as tf
import joblib
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

MODEL_PATH = "exp_models/Best_BiLSTM_model"
model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load("exp_models/bilstm_tokenizer.pkl")



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_api")

@app.post("/predict", response_model=PredOut)
def predict(inp: TweetIn):
    # Prétraitement du texte d'entrée
    print("Text input raw :")
    print(inp.text)
    text_df = pd.Series(data=[inp.text], name='text')
    print(text_df)
    text_clean = preprocess_data_embedding(X_raw=text_df, 
                                                        stem_lem_func=None,
                                                        tokenizer=tokenizer, 
                                                        stop_words=None, 
                                                        min_count=1, # mincount = 1 car on est sur le jeu de validation
                                                        max_len = 50, 
                                                        num_words=30000)
    # text = preprocess_text_simple(
    #     text=inp.text, 
    #     tokenizer=TweetTokenizer().tokenize,
    #     stem_lem_func=WordNetLemmatizer().lemmatize
    #     )
    
    print("Texte prétraité :", text_clean)


    proba = float(model.predict(text_clean)[0][0])  
    print(f"¨Probablilité calculées {round(proba,5)}")

    return PredOut(positive_proba=proba, positive=proba>=0.5)


@app.post("/feedback")
def feedback(data: FeedbackIn):
    # Logger, sauvegarder en base ou fichier
    logger.info(f"[FEEDBACK] Texte='{data.text}' | Pred={data.prediction} | "
                f"User={data.user_feedback} | Right={data.right_answer}")

    print(f"[FEEDBACK] Texte='{data.text}' | Pred={data.prediction} | User={data.user_feedback}")
    return {"status": "success", "message": "Feedback reçu"}