import unittest
from fastapi.testclient import TestClient
from API.app_bert import app
import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
import os




#==========================================================================================
# TESTS API
#==========================================================================================



class TestPredictEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_valid_text(self):
        # Exemple d'entrée
        data = {"text": "I love to travel with Air Paradis !"}
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        print("status code: OK")

        json_response = response.json()
        self.assertIn("positive_proba", json_response)
        self.assertIn("positive", json_response)

        # positive_proba doit être un float entre 0 et 1
        self.assertIsInstance(json_response["positive_proba"], float)
        self.assertGreaterEqual(json_response["positive_proba"], 0.0)
        self.assertLessEqual(json_response["positive_proba"], 1.0)

        # positive doit être un booléen
        self.assertIsInstance(json_response["positive"], bool)



class TestFeedbackEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_feedback_valid_data(self):
        # Exemple d'entrée
        data = {
            "text": "string",
            "prediction": True,
            "user_feedback": "string",
            "right_answer": True
        }
        response = self.client.post("/feedback", json=data)
        self.assertEqual(response.status_code, 200)

        json_response = response.json()
        self.assertIn("status", json_response)
        self.assertIn("message", json_response)

        # Vérifier que le message est bien celui attendu
        self.assertEqual(json_response["status"], "success")
        self.assertEqual(json_response["message"], "Feedback reçu")

#==========================================================================================
# TESTS MODELE
#==========================================================================================
class TestModelComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.__MAX_LEN__ = 64
        # Charger le tokenizer et le modèle une fois pour toutes les tests
        tokenizer_path = os.path.join("exp_models","final_tokenizer")
        model_path = os.path.join("exp_models","bert_model_f16.tflite")
        cls.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        cls.interpreter = tf.lite.Interpreter(model_path=model_path)

        cls.input_details = cls.interpreter.get_input_details()
        cls.output_details = cls.interpreter.get_output_details()
        cls.interpreter.resize_tensor_input(cls.input_details[0]['index'], [1, cls.__MAX_LEN__])
        cls.interpreter.resize_tensor_input(cls.input_details[1]['index'], [1, cls.__MAX_LEN__])
        cls.interpreter.allocate_tensors()

    def test_tokenizer_output(self):
        text = "This statement is a test."
        encoded = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.__MAX_LEN__, return_tensors='np')
        # Vérifier que les clés existent
        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        # token_type_ids peut être absent selon le modèle
        # Vérifier que shape des tenseurs est cohérente (batch size = 1)
        self.assertEqual(encoded["input_ids"].shape[0], 1)
        self.assertEqual(encoded["attention_mask"].shape[0], 1)

    def test_inference_output(self):
        text = "This statement is a test."
        encoded = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.__MAX_LEN__, return_tensors='np')
        encoded = {k: v.astype(np.int32) for k, v in encoded.items()} # On cast lkes inputs en int32 pour assurer la compatibilité tflite interpreter

        self.interpreter.set_tensor(self.input_details[0]['index'], encoded['attention_mask'])
        self.interpreter.set_tensor(self.input_details[1]['index'], encoded['input_ids'])

        # Inference
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        logits = np.array(output[0])[0]
        
        # Vérifier la forme (batch_size, 1)
        self.assertIsInstance(logits, (float, np.floating))
        self.assertGreaterEqual(logits, 0.0)
        self.assertLessEqual(logits, 1.0)


class TestModelModelSimpleCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.__MAX_LEN__ = 64
        # Charger le tokenizer et le modèle une fois pour toutes les tests
        tokenizer_path = os.path.join("exp_models","final_tokenizer")
        model_path = os.path.join("exp_models","bert_model_f16.tflite")
        cls.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        cls.interpreter = tf.lite.Interpreter(model_path=model_path)

        cls.input_details = cls.interpreter.get_input_details()
        cls.output_details = cls.interpreter.get_output_details()
        cls.interpreter.resize_tensor_input(cls.input_details[0]['index'], [1, cls.__MAX_LEN__])
        cls.interpreter.resize_tensor_input(cls.input_details[1]['index'], [1, cls.__MAX_LEN__])
        cls.interpreter.allocate_tensors()

    def predict_sentiment(self, text):
        encoded = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.__MAX_LEN__, return_tensors='np')
        encoded = {k: v.astype(np.int32) for k, v in encoded.items()} # On cast lkes inputs en int32 pour assurer la compatibilité tflite interpreter

        self.interpreter.set_tensor(self.input_details[0]['index'], encoded['attention_mask'])
        self.interpreter.set_tensor(self.input_details[1]['index'], encoded['input_ids'])

        # Inference
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        logits = np.array(output[0])[0]
        return logits >= 0.5  # booléen selon seuil 0.5

    def test_simple_sentiments(self):
        exemples = [
            ("I love this product It's fantastic !", True),
            ("The service is terrible, very disappointed.", False),
            ("I am happy with my experience.", True),
            ("This is the worst experience I've ever had.", False),
        ]

        for texte, expected in exemples:
            with self.subTest(texte=texte):
                prediction = self.predict_sentiment(texte)
                self.assertEqual(prediction, expected)
 

if __name__ == "__main__":
    unittest.main()