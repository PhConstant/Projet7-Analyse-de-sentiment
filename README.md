Air Paradis - Analyse de sentiment basée sur les tweets  
===
*Prédiction du sentiment associé à un tweet à partir de différents modèles de machine learning - Mise en place d'une démarche MLOPs*

Description
---
Les objectifs du projet sont multiples : 

1. Evaluer plusieurs algorithmes de machine learning et de deep learning afin d'évaluer l'aspect positif/négatif d'un tweet portant sur la compagnie (fictive) *Air Paradis*
2. Mettre en place un suivi et un monitoring des différents modèles créés lors de la phase de recherche et d'optimisation des hyperparamètres 
3. Sélectionner le modèle le plus adapté et le préparer pour un déploiment sur le cloud
4. Mettre en place une API permettant d'appeler diretement la modèle
5. Créer une interface utilisateur simple permettant à l'utilisateur final de saisir du texte et d'obtenir la classe prédite de sentiment (positif ou négatif)

Le jeu de données utilisé pour l'entrainement des modèles est disponible en téléchargement direct  [ici][1] ou sur [Kaggle][2]. Ici seul les colonnes `text` et `target` du dataset ont été exploitées. 

Les différents notebooks du dossier **Notebooks** présentent la démarche de recherche appliquée : 
- `Model1_LogReg.ipynb` : Analyse exploratoire et modélisation par regression logistique simple
- `Model2_WordEmbeddings.ipynb` et `Modèle3_BiLSTM.ipynb` : Modélisations par des modèles de réseaux de neurones simples (RNN, GRU, LSTM et BiLSTM)
- `Model4_BERT.ipynb` : Utilisation de modèles de types transformers (BERT, RoBERTa) préentrainés et modifiés pour le cas d'application souhaité. 
- `Model5_USE.ipynb` : Modélisation par méthodes de sentence encoding (Universal Sentence Encoder)
- `Convert_to_TFlite.ipynb` : Conversion du modèle final en vue de son déploiment sur le cloud (réduction de la taille du modèle et comparaison des performances du modèle réduit avec le modèle complet)


Liste des outils utilisés : 
---
- **Modélisation** : scikit-learn *(Régression Logistique)*, tensorflow.keras *(Réseaux de neurones)*, transformers *(BERT/USE)*
- **Tracking/Experiment** : MLFlow *(Tracking des différentes expérimentations)*, Optuna *(Optimisation des hyperparamètres)*
- **API** : FastAPI
- **Interface utilisateur** : Streamlit  

Installation : 
---
  

### Prérequis
- **Python** : v3.10.18
- **Packages indispensables pour un run API en local** : 
    - tensorflow : 2.10.1
    - transformers : 4.56.0
    - fastapi : 0.117.0
    - httpx : 0.28.1
    - pydantic : 2.11.9
    - uvicorn : 0.36.0
    - numpy : 1.26.4
    - pandas : 2.3.2
    - azure-monitor-opentelemetry : 1.8.1

### Copie du repository GitHub :
Lancer ce code depuis votre terminal :
```bash
git clone https://github.com/PhConstant/Projet7-Analyse-de-sentiment.git
cd <mon_dossier> # <- A modifier 
pip install -r requirements.txt
```

Utilisation : 
---
### Lancement en local de l'application FastAPI 
```bash
cd <mon_dossier> # A modifier (racine du projet)
uvicorn API.app_bert:app --reload --host localhost --port 8000
```
Puis dans votre navigateur visiter l'addresse : http://localhost:8000/docs


### Lien vers l'API sur le cloud Microsoft Azure

Pour un accès direct à l'application sur le cloud Microsoft Azure, visiter le lien : https://pc-p7-analyse-de-sentiment-fxecf4frfabsfxbq.francecentral-01.azurewebsites.net/docs


### Lancement de l'interface graphique en local 

```bash
cd <mon_dossier> # A modifier (racine du projet)
streamlit run "Streamlit App/Streamlit_app.py" --server.port 8050
```

Puis dans votre navigateur visiter l'addresse : http://localhost:8050


### Exemples d'appels à l'API 

#### Depuis la ligne de commande : 
L'API peut être appelée directement depuis la ligne de commande via curl : 

```bash
curl -X 'POST' \
  'https://pc-p7-analyse-de-sentiment-fxecf4frfabsfxbq.francecentral-01.azurewebsites.net/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "I love this product, it is really amazing !"}'
```
*Exemple de retour fourni en cas de succès (code 200)* :
```
{
  "positive_proba": 0.9555624127388,
  "positive": true
}
```

#### Depuis un script python via le package `requests` :
```python
import requests

API_URL = "https://pc-p7-analyse-de-sentiment-fxecf4frfabsfxbq.francecentral-01.azurewebsites.net/predict"

tweet  = "I love this product, it is really amazing !"

response = requests.post(API_URL, json={"text": tweet})
print(response.json())

```
*Exemple de retour fourni en cas de succès (code 200)* :
```python
{'positive_proba': 0.9555624127388, 'positive': True}
```

Résultats des différents modèles testés 
---
Ici on récapitule les différentes métriques enregistrées pour tous les types de modèles testés dans leur versions les plus performantes : 

|Modèle|Accuracy|F1-score -|F1-score +|Précision -|Précision +|ROC-AUC|Recall -|Recall +|
|-----:|-------:|---------:|---------:|----------:|----------:|------:|-------:|-------:|
|Régression logistique|0.78|0.77|0.78|0.78|0.77|0.85|0.77|0.79|
|RNN (Embedding custom)|0.73|0.72|0.74|0.74|0.71|0.80|0.70|0.76|
|RNN (Embedding pré-entrainé)|0.795|0.80|0.795|0.79|0.80|0.87|0.80|0.79|
|LSTM|0.79|0.80|0.79|0.78|0.81|0.88|0.82|0.77|
|BiLSTM|0.80|0.79|0.80|0.81|0.79|0.88|0.78|0.81|
|**BERT**|**0.82**|**0.82**|**0.82**|**0.825**|**0.82**|**0.90**|**0.82**|**0.83**|
|USE|0.80|0.80|0.80|0.81|0.80|0.89|0.79|0.81|

Le modèle utilisé est le modèle **BERT** le plus performant obtenu sur le dataset.


Tests unitaires 
---
Les tests unitaires sont executés pour chaque build de l'application et sont stockés dans le dossier `Tests` et peuvent être lancés via la commande : 
```bash
python -m unittest discover -s Tests -p "*.py"
```

Contributeurs
---
- **Constant Philippe** — Développeur principal  |  [PhConstant](https://github.com/PhConstant)


[1]: https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+7%C2%A0-+D%C3%A9tectez+les+Bad+Buzz+gr%C3%A2ce+au+Deep+Learning/sentiment140.zip
[2]: https://www.kaggle.com/datasets/kazanova/sentiment140