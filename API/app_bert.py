# API/app.py

from fastapi import FastAPI
from pydantic import BaseModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from transformers import AutoTokenizer
import tensorflow as tf
import logging
from azure.monitor.opentelemetry import configure_azure_monitor

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin absolu du tokenizer
TOKENIZER_PATH = os.path.join(BASE_DIR, "exp_models", "final_tokenizer")
# Chargement du tokenizer HF
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print(f"Loading tokenizer from {TOKENIZER_PATH}")
MAX_LEN = 64
THRESHOLD = 0.45279  # Seuil optimal après conversion tflite float16
# Charger le modèle TFLite float16
MODEL_PATH = os.path.join(BASE_DIR, "exp_models","bert_model_f16.tflite")
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



# Configure OpenTelemetry to use Azure Monitor with the 
# APPLICATIONINSIGHTS_CONNECTION_STRING environment variable.
configure_azure_monitor(
    connection_string="InstrumentationKey=bc6d51f2-9c60-4b69-891b-f66da44843d3;IngestionEndpoint=https://francecentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://francecentral.livediagnostics.monitor.azure.com/;ApplicationId=ec4b110b-8c98-43e6-a86f-2353d580a57c",
    logger_name="sentiment_api_logger",  # Set the namespace for the logger in which you would like to collect telemetry for if you are collecting logging telemetry. This is imperative so you do not collect logging telemetry from the SDK itself.
)
logger = logging.getLogger("sentiment_api_logger")
logger.setLevel(logging.INFO)
logger.info("Application Insights logging initialized.")


@app.post("/predict", response_model=PredOut)
def predict(inp: TweetIn):
    # Prétraitement du texte d'entrée
    print("Text input raw :")
    print(inp.text)

    inputs = tokenizer(inp.text, return_tensors="np", padding='max_length', max_length=MAX_LEN, truncation=True)
    inputs = {k: v.astype(np.int32) for k, v in inputs.items()} # On cast lkes inputs en int32 pour assurer la compatibilité tflite interpreter

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
    logger.setLevel(logging.INFO)
    logger.info(f"[FEEDBACK] Texte='{data.text}' | Pred={data.prediction} | "
                f"User={data.user_feedback} | Right={data.right_answer}")
    if not data.right_answer:
        logger.setLevel(logging.WARNING)
        logger.warning(
            "Mauvaise prédiction",
            extra={
                "custom_dimensions": {
                    "text": data.text,
                    "predicted": data.prediction,
                    "user_feedback": data.user_feedback,
                }
            },
        )
    return {"status": "success", "message": "Feedback reçu"}