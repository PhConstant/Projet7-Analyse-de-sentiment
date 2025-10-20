

Analyse de sentiments par Deep learning - Modélisation et Déploiment de modèles - une démarche orientée MLOPs 
===

# Présentation du dataset

Les modèles ont été entrainés sur le dataset **Sentiment140** disponible sur *Kaggle* contenant **1,6 million de tweets** annotés pour l'analyse de sentiments. Chaque ligne correspond à un **tweet** et contient plusieurs informations :  


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {vertical-align: middle;}
    .dataframe tbody tr th {vertical-align: top;}
    .dataframe thead th {text-align: right;}
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th> <th>target</th> <th>ids</th> <th>date</th> <th>flag</th> <th>user</th> <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1498500</th> <td>4</td> <td>2070583051</td> <td>Sun Jun 07 17:57:16 PDT 2009</td> <td>NO_QUERY</td>   <td>bgardner</td> <td>@Corpsman_Com You wouldn't have to pay for the...</td>
    </tr>
  </tbody>
</table>
</div>

Les classes  sont **équilibrées**, garantissant un apprentissage stable lors de l'entrainemennt des modèles.

<figure style="text-align: center;">
    <img src='Livrables/target.png'>
    <figcaption style="text-align: center;"><i>Distribution du sentiment (0 : négatif, 1 : positif)</i></figcaption>
</figure>


Dans cette étude, seules deux colonnes sont utilisées : 
- `text` , qui contient le message à analyser, 
- `target`, qui sert de variable cible pour l'apprentissage supervisé. 

Ces données constituent la base d'entrainement des différents modèles - linéaires, réseaux de neurones et transformers - pour la prédiction de sentiment. 

# Modélisation

L'objectif de cette étapes est d'évaluer  plusieurs approches pour la modélisation de sentiments à partir de tweets : 
- **Regression logistique**  : modèle de référence (baseline) 
- **Réseaux de neurones récurrents (RNN, GRU, LSTM)** : prise en compte du contexte séquentiel 
- **Transformers (BERT)** : représentation contextuelle globale

--- 

## Regression logistique
<figure style="text-align: center;">
    <img src='Livrables/A-toy-logistic-regression-classifier-with-2D-Gaussian-data.png', height=300px>
    <figcaption style="text-align: center;"><i>Illustration de régression logistique</i></figcaption>
</figure>


La regression logistique sert ici de modèle de référence. Sa performance dépend fortement du **feature engineering** consistant à transformer les chaines de caractères en représentations numériques exploitables. 

Les principales étapes sont : 
- Nettoyage et normalisation du texte
- Stemming / Lemmatization
- Vectorisation (TF-IDF ou CountVectorizer)

Une fois les textes vectorisés, le modèle est entrainé sur le jeu de données équilibré. Les performances obtenues atteignent environ **78% d'accuracy** sur le jeu de validation. 

Ce modèle, léger et rapide à déployer, constitue une excellente baseline. Il ne capture cependant pas les relations entre les mots. Le texte est considéré comme un ensemble de tokens ("bag-of-words"). Les structures complexes de phrases ne sont pas correctement identifiés.

---

## Modèle de réseau de neurones (RNN, GRU, LSTM)

<figure style="text-align: center;">
    <img src='Livrables/The-indepenent-cells-of-RNN-LSTM-and-GRU.png', height=200px>
    <figcaption style="text-align: center;"><i>Architecture des différents types de cellules recursives pour les RNNs</i></figcaption>
</figure>

Le modèles de type  RNN abordent une limite majeure des approches de type "bag-of-words" : ils considèrent les textes comme des **séquences ordonnées de tokens**. Chaque unité de traitement dépend du mot précédent, ce qui permet de modéliser le contexte. 

La qualité de ces modèles dépend de plusieurs paramètres : 
- Prétraitement du texte (nettoyage, normalisation, stemming ou lemmatisation)
- Méthode d'**embedding** (Word2Vec, GloVe, FastText)
- Type de cellule (RNN Simple, GRU ou LSTM)
- Longueur des séquences (troncage et/ou bourrage) 

Les cellules **LSTM** (Long Short-Term Memory) atténuent le problème du *graddient évanescent* des RNNs classiques, en conservant la mémoire à long terme. Ces architectures offrent de meilleurs performances (**79% d'accuracy** sur le jeu de validation) mais leur entrainement est plus long et plus couteux.

---

## Modèles de transformers (BERT)

<figure style="text-align: center;">
    <img src='Livrables/BERT_Schema-1.png', height=300px>
    <figcaption style="text-align: center;"><i>Principe d'un modèle BERT pour la classification</i></figcaption>
</figure> 

Les modèles de type **Transformers**, tels que **BERT**, reposent sur les mécanismes d'**attention** qui permettent d'intégrer un contexte global à chaque token. Contrairement aux RNNs, ils ne traitent pas les séquences mot par mot mais analysent les dépendances entre tous les tokens simultanément.

Le modèle utilisé ici provient de la plateforme *Hugging Face* et est associé à son tokenizer **pré-entrainé**. Seule la **tête de classification** a été fine-tunée sur le dataset, les poids du modèle de base sont conservés pour des raisons de disponibilité de ressources.

Cette approche permet d'obtenir les meilleures performances, au prix d'un coût de calcul et de stockage beaucoup plus élevé.

---

### Bilan sur les différents modèles 

Les performances globales des meilleurs modèles pour chaque approche sont compilées dans le tableau suivant : 

|Modèle|Accuracy|F1-score -|F1-score +|Précision -|Précision +|ROC-AUC|Recall -|Recall +|
|-----:|-------:|---------:|---------:|----------:|----------:|------:|-------:|-------:|
|Régression logistique|0.78|0.77|0.78|0.78|0.77|0.85|0.77|0.79|
|LSTM|0.79|0.80|0.79|0.78|0.81|0.88|0.82|0.77|
|**BERT**|**0.82**|**0.82**|**0.82**|**0.825**|**0.82**|**0.90**|**0.82**|**0.83**|

Les résultats confirment la progression attendue : 
- Le modèle linéaire offre une baseline solide et interprétable
- Le LSTM apporte un léger gain en capturant la dimension séquentielle des textes.
- BERT obtient les meilleurs scores grâce à une compréhension contextuelle approfondie.

Ces performances ont ensuite été exploitées dans une démarche **MLOps** complète visant à suivre, optimiser et déployer les modèles en production. 

# Tracking des expériments avec MLFlow et Optuna

Dans une approche **MLOps**, le suivi et l'optimisation des expériences sont essentielles pour garantir la reproductibilité, la traçabilité et la selection du meilleur modèle. Deux outils sont utilisés : 

- **MLFlow** pour le **tracking** des entrainements et des performances, 
- **Optuna** pour l'**optimisation des hyperparamètres**.

## Suivi des performances des modèles avec MLFlow 
Lors de l'entrainement des différents types de modèles **MLFlow** permet d'enregistrer et comparer leurs performances à travers plusieurs executions (*runs*). Chaque run conserve les paramètres, métriques,artefacts et modèles entrainés permettant ainsi le suivi complet de chaque experimentation.

### Initialisation du serveur MLFlow

Tout d'abord, il est nécessaire de lancer le serveur MLFlow localement pour enregistrer les experiences : 

```bash
mlflow server --host localhost --port 8080
```

Le client MLFlow est également initialisé dans le notebook python :

```python

from mlflow import MlflowClient
import mlflow
from pathlib import Path

client = MlflowClient(tracking_uri="http://localhost:8080")
mlruns_path = Path("./mlruns").resolve() 
mlflow_uri = mlruns_path.as_uri()
mlflow.set_tracking_uri(mlflow_uri)

# Création de notre experiment MLFlow
mlflow.set_experiment("Experiment 1")
```
### Enregistrement des runs

Lors de l'entrainement d'un modèle chaque run est encapsulée dans un contexte :
```python
with mlflow.start_run(): 
        mlflow.log_input(dataset)                # <- Dataset d'entrainement 
        mlflow.log_params(params)                # <- Paramètres
        mlflow.log_metrics(output)               # <- Métriques 
        mlflow.log_artifatcts(artifact)          # <- Fichiers additionnels 
        mlflow.sklearn.log_model(model, "model") # <- Sauvegarde du modèle créé 
    
```
### Visualisation et comparaison

Le suivi des experiences s'effectue via l'interface MLFlow accessible à l'adresse de tracking :
 <figure style="text-align: center;">
    <img src='Livrables/ExperimentTracking1.png'>
    <figcaption style="text-align: center;"><i>Interface de MLFlow pour le tracking des expériences</i></figcaption>
</figure>


Les performances des modèles peuvent être comparées graphiquement pour chaque experiment :
 <figure style="text-align: center;">
    <img src='Livrables/ExperimentTracking2_optim.png'>
    <figcaption style="text-align: center;"><i>Comparaison des performances des modèles</i></figcaption>
</figure>

---

## Optimisation des hyperparamètres avec Optuna

Une fois le tracking opérationnel, la librairie **Optuna** permet d'optimiser automatiquement les hyperparamètres des modèles.

### Définition de la fonction objectif

L'optimisation est basée sur une fonction objectif à maximiser (ou minimiser):
```python
import optuna

def run_function(trial)
  param1 = trial.suggest_float("param1", min_float, max_float)      
  param2 = trial.suggest_int("param2", min_int, max_int)            
  param3 = trial.suggest_categorical("param3", ["cat1", "cat2"])    
  # Entrainement et évaluation du modèle
  ...
  return output # <- Score à optimiser
```
### Lancement de l'étude

L'étude Optuna est ensuite créée et exécutée :
```python
# Initialisation
study = optuna.create_study(direction="maximize")
# Lancement
study.optimize(run_function, n_trials=50)
```

Optuna cherche ainsi automatiquement la combinaison d'hyperparamètres maximisant (ou minimisant) la fonction objectif.

## Conclusion
Une fois les différentes expériences réalisées, le **modèle le plus performant** est sélectionné, sauvegardé et prêt à mettre en production. La prochaine étape consiste à le **déployer** et rendre accessible le service de prédiction via une **API** et une **interface utilisateur**.  

# Mise en production

Le modèle le plus performant étant sélectionné, il est mis en production pour fournir un service de prédictionpar accessible aux utilisateurs finaux.Cette étape inclut la création d'une API, d'une interface utilisateur, le déploiment sur le cloud et l'optimisation des ressources.

---

## Création d'une API FastAPI

Le modèle est exposé via une **API REST** créée avec **FastAPI**. L'API comporte deux points d'entrée : 
- `/predict` : pour la prédiction de sentiment d'un tweet
- `/feedback` : pour recevoir un retour utilisateur

 <figure style="text-align: center;">
    <img src='Livrables/API_FastAPI.png'>
    <figcaption style="text-align: center;"><i>Prise de vue de la page de doc de l'API</i></figcaption>
</figure>

L'API est d'abord testée en local avant déploiment sur le cloud.

---

## Interface utilisateur avec Streamlit

Pour faciliter l'interaction avec le modèle une interface graphique est créée avec **Streamlit**.

Les utilisateurs peuvent : 
- Saisir un tweet 
- Obtenir la prédiction ddu modèle
- Fournir un feeback sur la qualité de la prédiction

 <figure style="text-align: center;">
    <img src='Livrables/AppStreamlit.png'>
    <figcaption style="text-align: center;"><i>Interface utilisateur</i></figcaption>
</figure>

---

## Déploiment de l'API sur le cloud Microsoft Azure

L'API est ensuite déployée sur une **Web App Azure**, connectée au dépôt GitHub du projet contenant le code et le modèle.
Un **workflow GitHub Actions** automatise le déploiment en deux phases : **build** et **deploiement**. 

<figure style="text-align: center;">
    <img src='Livrables/github_azure_deployment.png'>
    <figcaption style="text-align: center;"><i>Diagramme du workflow GitHub Actions</i></figcaption>
</figure>

### Le Build

La phase de **build** construit l'application à partir : 
- Du code source et du modèle
- Des dépendances listées dans le fichier *requirements.txt*

### Tests unitaires

Les tests unitaires sont effectués durant la phse de build et permettent de tester chaque fonction du code source : 
- Type et dimensions des inputs et outputs des fonctions
- Comparaison d'une ou plusieurs réponses

```bash
python -m unittest discover -s Tests -p "*.py"
```

### Le Déploiment

Cette phase consiste à :
- Reproduire l'environnement python à partir de l'artefacts build 
- Lancer automatiquement l'application au démarrage

### Optimisation des ressources

Les resources cloud sont très limitéespar rapport aux machines de développement. Il est donc essentiels d'optimiser :
- Les packages Python : retrait des packages lourds ou inutilisés
- Les fichiers et dossiers du build : images, notebooks et runs MLFlow
- La taille du modèle (conversion Tensorflow Lite)

Une fois ces étapes réalisées l'application est prète à être utilisée par les utilisateurs finaux. La section suivante aborde **le monitoring et la gestion des alertes**, indispensable pour assurer la performance continue et la fiabilité de l'API.

---

# Monitoring de l'application

Une fois le moddèle déployé, il faut assurer sa performance continue et de détecter rapidement les dysfonctionnements. Le monitoring permet de suivre l'utilisation de l'API, de collecter des évènements importants et de déclencher des alertes. Cela est assuré via **Azure Application Insights**.

## Logging

Le logging centralisé permet de collecter les évènements et de les analyser en temps réel. Chaque retour utilisateur peut être tracé.

### Configuration Python

```python
import logging
from azure.monitor.opentelemetry import configure_azure_monitor

# Configuration du monitoring azure pour l'application
configure_azure_monitor(
    connection_string=
    """InstrumentationKey=AZURE_MONITORING_INSTRUMENTATION_KEY>; 
       IngestionEndpoint=AZURE_MONITORING_ENDPOINT_URI>""",
    logger_name="sentiment_api_logger", 
)
logger = logging.getLogger("sentiment_api_logger")
logger.setLevel(logging.INFO)
logger.info("Application Insights logging initialized.")
``` 
Le `logger` envoie automatiquement les évènements vers Application Insights, permettant de centraliser le suivi. 

## Création d'alertes 

Les alertes permettent de réagir automatiquement aux évènements critiques, comme un taux élevé de feedbacks négatifs en peu de temps.

Cela se fait en trois étapes :
- Requète sur les logs
- Création de condition d'alerte
- Action automatique : envoi d'e-mail ou SMS

<figure style="text-align: center;">
    <img src='Livrables/AlertRuleCond.png'>
    <figcaption style="text-align: center;"><i>Génération d'alerte - Condition</i></figcaption>
</figure>

<figure style="text-align: center;">
    <img src='Livrables/AlertRuleAction.png'>
    <figcaption style="text-align: center;"><i>Génération d'alerte - Action</i></figcaption>
</figure>


# Conclusion

## Bilan

Le projet a permis de construire un pipeline complet : préparation de données, comparaisons d'approches (regression logistique, LSTL, BERT), tracking (MLFlow), optimisation (Optuna), déploiement (FastAPI, Streamlit, Azure Web App) et monitoring (Application Insights). BERT a fourni les meilleures performances, tandis que la baseline reste pertinente pour un déploiement léger.

## Perspectives

Parmi les évolutions utiles : automatiser le pipeline d'entrainement (AirFlow / Azure ML Pipeline), intégrer le feedback utilisateur pour un apprentissage continu, surveiller la dérive et déclencher des réentrainements automatiques, et renforcer les tests automatisés
