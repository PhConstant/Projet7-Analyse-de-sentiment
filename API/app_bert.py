# API/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
from Source.preprocess_data import preprocess_data_embedding
from transformers import AutoTokenizer
import tensorflow as tf
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


# Chargement du tokenizer HF
TOKENIZER_PATH = os.path.join("exp_models","final_tokenizer")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print(f"Loading tokenizer from {TOKENIZER_PATH}")
MAX_LEN = 64
THRESHOLD = 0.45034 # Seuil optimal après conversion tflite float16
# Charger le modèle TFLite float16
MODEL_PATH = os.path.join("exp_models","bert_model_f16.tflite")
print(f"Loading model from {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
print("Model loaded.")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Sequence length for model: {MAX_LEN} tokens")
interpreter.resize_tensor_input(input_details[0]['index'], [1, MAX_LEN])
interpreter.resize_tensor_input(input_details[1]['index'], [1, MAX_LEN])
interpreter.allocate_tensors()
print("Interpreter allocated.")




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_api")

@app.post("/predict", response_model=PredOut)
def predict(inp: TweetIn):
    # Prétraitement du texte d'entrée
    print("Text input raw :")
    print(inp.text)

    inputs = tokenizer(inp.text, return_tensors="np", padding='max_length', max_length=MAX_LEN, truncation=True)

    interpreter.set_tensor(input_details[0]['index'], inputs['attention_mask'])
    interpreter.set_tensor(input_details[1]['index'], inputs['input_ids'])

    # Inference
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    proba = np.array(output[0])[0]
    print(f"Probabilité calculée : {round(proba, 5)}")

    return PredOut(positive_proba=proba, positive=proba>=THRESHOLD)


@app.post("/feedback")
def feedback(data: FeedbackIn):
    # Logger, sauvegarder en base ou fichier
    logger.info(f"[FEEDBACK] Texte='{data.text}' | Pred={data.prediction} | "
                f"User={data.user_feedback} | Right={data.right_answer}")

    print(f"[FEEDBACK] Texte='{data.text}' | Pred={data.prediction} | User={data.user_feedback}")
    return {"status": "success", "message": "Feedback reçu"}