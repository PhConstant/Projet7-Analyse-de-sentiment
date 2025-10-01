# API/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
from Source.preprocess_data import preprocess_data_embedding
from transformers import AutoTokenizer
import tensorflow as tf
import joblib
import logging
import os

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
MODEL_PATH = os.path.join("exp_models","Best_BERT_model2") 
print(f"Loading model from {MODEL_PATH}")
TOKENIZER_PATH = os.path.join("exp_models","Best_BERT_tokenizer")
print(f"Loading tokenizer from {TOKENIZER_PATH}")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_api")

@app.post("/predict", response_model=PredOut)
def predict(inp: TweetIn):
    # Prétraitement du texte d'entrée
    print("Text input raw :")
    print(inp.text)

    encoded_input = tokenizer(
        inp.text,
        padding=True,
        truncation=True,
        return_tensors='tf'
    )

    output = infer(**{
    "input_ids": encoded_input["input_ids"],
    "attention_mask": encoded_input["attention_mask"],
    "token_type_ids": encoded_input.get("token_type_ids", tf.zeros_like(encoded_input["input_ids"]))
    })

    proba = output['output_0'].numpy()[0][0]
    print(f"Probabilité calculée : {round(proba, 5)}")

    return PredOut(positive_proba=proba, positive=proba>=0.5)


@app.post("/feedback")
def feedback(data: FeedbackIn):
    # Logger, sauvegarder en base ou fichier
    logger.info(f"[FEEDBACK] Texte='{data.text}' | Pred={data.prediction} | "
                f"User={data.user_feedback} | Right={data.right_answer}")

    print(f"[FEEDBACK] Texte='{data.text}' | Pred={data.prediction} | User={data.user_feedback}")
    return {"status": "success", "message": "Feedback reçu"}