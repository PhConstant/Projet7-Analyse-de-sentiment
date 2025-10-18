

Analyse de sentiments par Deep learning - Modélisation et Déploiment de modèles - une démarche orientée MLOPs 
===

# Présentation du dataset

Les modèles ont été entrainés sur le dataset ` Sentiment140 dataset with 1.6 million tweets` disponible en open source sur *Kaggle*. 

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

Ce dataset contient des informations relatives à des tweets postés entre 2009 et 2020. A chaque tweet est attribué un sentiment (positif ou négatif) défini dans la colonne `target`. Les deux classes  sont équilibrées.

<figure style="text-align: center;">
    <img src='Livrables/target.png'>
    <figcaption style="text-align: center;"><i>Distribution du sentiment (0 : négatif, 1 : positif)</i></figcaption>
</figure>


Ici, seules deux colonnes sont utilisées : la `target` et le `text`. 

# Modélisation

Plusieurs types de modélisations sont utilisées pour prédire la target à partir du text : 
- Modèle Simple linéaire : **Regression Logistique**
- Modèle de réseau de neurones :  **LSTM** 
- Modèles à base de transformers :  **BERT**

### Regression Logistique
<figure style="text-align: center;">
    <img src='Livrables/A-toy-logistic-regression-classifier-with-2D-Gaussian-data.png'>
    <figcaption style="text-align: center;"><i>Illustration de régression logistique</i></figcaption>
</figure>


Pour cette approche la partie la plus importante réside dans le feature engineering qui est appliqué pour convertir les chaines de caractères en variables interprétables par un modèle numérique. Celà passe par plusieurs étapes : 
- Nettoyage et normalisation des `text`
- Stemming/Lemmatization
- Vectorisation

Puis unbe regression logistique est entrainée sur le dataset. Les résultats obtenus avec ce modèle  permettent d'atteindre un niveau d'accuracy relativement élevée :  **78%** sur le jeu de validation. 

Ce modèle a l'avantage d'être très léger et de permettre un déploiment rapide et efficace, cependant il n'est pas capable de comprendre un cas complexe, de détecter le sarcasme ou de comprendre la double négation. Ceci est du au principe du bag-of-words qui ne prend pas le texte comme une séquence ordonnée de tokens, mais comme un ensemble de mots sans relations entre eux. 

### Modèle de réseau de neurones (RNN, GRU, LSTM)

<figure style="text-align: center;">
    <img src='Livrables/The-indepenent-cells-of-RNN-LSTM-and-GRU.png'>
    <figcaption style="text-align: center;"><i>Architecture des différents types de cellules recursives pour les RNNs</i></figcaption>
</figure>

Le modèles de réseaux de neurones de types récursifs (RNN) permettent de compenser le principal défaut des méthodes de bagging. Il considère les textes comme une séquence ordonnée de jetons. 

Pour ces types de modèles, la qualité des prédictions va principalement dépendre de plusieurs étapes : 
- Nettoyage et normalisation des chaines de caractères
- Stemming/Lemmatization
- Embedding
- Type de cellule RNN : RNN Simple, GRU, LSTM
- Longueur des séquences (troncage et/ou bourrage) des séquences 

Ces modèles ont l'avantage de traiter les différents tokens comme une séquence ordonnée (le traitement du token $x_n$, prend en compte l'output $h_{n-1}$ issue du traitement du token $x_{n-1}$). 

Ces modèles peuvent cependant être sujets au phénomène de gradient évanescent et ne sont donc pas pertinents pour de très longues séquences même si les cellules **LSTM** ont été crées afin de remédier à ce problème.

### Modèle de transformers (BERT)

<figure style="text-align: center;">
    <img src='Livrables/BERT_Schema-1.png'>
    <figcaption style="text-align: center;"><i>Principe d'un modèle BERT pour la classification</i></figcaption>
</figure>

Le principal avantage des modèles de transformers, basés sur des mécanismes d'attention, sur les modèles à base de LSTM est de fournir pour chaque token un contexte global contrairement aux modèles RNN classiques qui ne prennent en compte dans l'évaluation des tokens que les tokens ayant déjà été traités.  

Les modèles de transformers sont téléchargés directement depuis Huggingface et sont associés à un tokenizer pré-entrainé. Ils sont cependant trop gros pour être réentrainés ou fine-tunés localement.  Seule la tête de classification ajoutéespour l'application et le dataset à notre disposition est entrainée. 

### Bilan sur les différents modèles 

Les résultats des différents modèles sont compilés dans le tableau suivant : 

|Modèle|Accuracy|F1-score -|F1-score +|Précision -|Précision +|ROC-AUC|Recall -|Recall +|
|-----:|-------:|---------:|---------:|----------:|----------:|------:|-------:|-------:|
|Régression logistique|0.78|0.77|0.78|0.78|0.77|0.85|0.77|0.79|
|LSTM|0.79|0.80|0.79|0.78|0.81|0.88|0.82|0.77|
|**BERT**|**0.82**|**0.82**|**0.82**|**0.825**|**0.82**|**0.90**|**0.82**|**0.83**|

# Tracking des expériments avec MLFlow et Optuna

## Tracking des performances des modèles 
Lors de l'entrainement des différents types de modèles MLFlow permet de suivre les performances des modèles lors de l'entrainement pour sélectionner le meilleur modèle.

Tout d'abord il est nécessaire de lancer MLFlow dans l'invite de commande pour lancer le serveur qui servira à enregistrer notre experiment : 
```bash
mlflow server --host localhost --port 8080
```
Au début du notebook le client MLFlow est initialisé :
```python
from mlflow import MlflowClient
client = MlflowClient(tracking_uri="http://localhost:8080")
mlruns_path = Path("./mlruns").resolve() 
mlflow_uri = mlruns_path.as_uri()
mlflow.set_tracking_uri(mlflow_uri)
# Création de notre experiment MLFlow
mlflow.set_experiment("Experiment 1") # <-  Nouvelle étude avec MLFlow
```
Puis lors de l'entrainement du modèle choisi:
```python
with mlflow.start_run(): # <-  Nouvelle run mlflow a
        mlflow.log_input(dataset) # <- Logging du dataset utilisé pour l'entrainement 
        mlflow.log_params(params) # <- Logging des paramètres
        ... # <- La suite du code
        mlflow.log_metrics(output) # <- Logging des outputs 
        mlflow.log_artifatcts(artifact) # <- Logging d'autre fichiers : images, textes.. 
        mlflow.sklearn.log_model(model, "model") # <- Logging du modèle créé en précisant le framework (sklearn, tensorflow, pytorch ...) 
```
Le suivi de l'avancement des experiment et des performances des modèles se fait en se connectant à l'adresse de tracking dans un navigateur internet :
 <figure style="text-align: center;">
    <img src='Livrables/ExperimentTracking1.png'>
    <figcaption style="text-align: center;"><i>Interface de MLFlow pour le tracking des experiments</i></figcaption>
</figure>


Les performances des modèles peuvent être comparées graphiquement pour chaque experiment :
 <figure style="text-align: center;">
    <img src='Livrables/ExperimentTracking2_optim.png'>
    <figcaption style="text-align: center;"><i>Comparaison des performances des modèles en fonction des hyperparamètres choisis</i></figcaption>
</figure>

## Optimisation des hyperparamètres avec Optuna

La librairie Optuna permet d'optimiser les hyperparamètres des modèles. Les paramètres sont exprimés sous forme de plage de variations (pour les nombres entiers ou flottants) ou sous forme de liste (pour les variables categorielles ou les chaines de caractères)

La paramétrisation de l'optimisation est faite de manière suivante :
```python
import optuna

# Optuna optimise l'output d'une fonction (run_function)
def run_function(trial)
  param1 = trial.suggest_float("param1", min_float, max_float) # <- Paramètre de type float
  param2 = trial.suggest_int("param2", min_int, max_int)  # <- Paramètre de type int
  param3 = trial.suggest_categorical("param3", ["cat1", "cat2"])  # <- Paramètre catégoriel sous forme de liste

  # Code de la fonction
  ...
  #
  return output # <- Valeur numérique à optimiser
```
Puis l'étude d'optimisation est crée et lancée :
```python
# Initialisation
study = optuna.create_study(direction="maximize")
# Lancement
study.optimize(run_function, n_trials=50)
```

## Conclusion
Une fois les différentes expériments réalisés, le modèle le plus performant est sélectionné et sauvegardé. Il est maintenant possible de le mettre en production. 

# Mise en production

Le modèle sélectionné est mis en production par déploiment d'une API sur le cloud de MS Azure. Une interface graphique est mise à disposition des utilisateurs finaux afin d'interroger le modèle et d'obtenir les résultats pour chaque tweet proposé. 

## Création d'une API 

Une API est créée via la librairie FastAPI et testée en local. L'API créée a deux points d'entrée : 
- Pour la prédiction du modèle (`/predict`)
- Pour le feedback utilisateur (`/feedback`)

 <figure style="text-align: center;">
    <img src='Livrables/API_FastAPI.png'>
    <figcaption style="text-align: center;"><i>Prise de vue de la page de doc de l'API</i></figcaption>
</figure>

## Création d'une interface utilisateur avec Streamlit

L'interface utilisateur permet de faire appel au modèle et de fournir un feedback sur la prediction fournie : 

 <figure style="text-align: center;">
    <img src='Livrables/AppStreamlit.png'>
    <figcaption style="text-align: center;"><i>Prise de vue de l'interface graphique</i></figcaption>
</figure>

## Déploiment de l'API sur le cloud Microsoft Azure

Une fois les l'API et l'interface graphique crée, il est nécessaire de mettre à disposition des utilisateurs cette API pour les utilisateurs finaux. Pour ce faire l'API doit être déployée sur le cloud (ici MS Azure). 

Une Web App azure est crée sur le portail Azure et liée au repo GitHub de l'application. Afin de lancer le déploiment un workflow est créé en deux phases :  build et déploiment.

 <figure style="text-align: center;">
    <img src='Livrables/github_azure_deployment.png'>
    <figcaption style="text-align: center;"><i>Diagramme du workflow GitHub Actions</i></figcaption>
</figure>

### Le Build

La phase de **build** permet de construire notre application à partir : 
- Du code source et des modèles présents sur le repo GitHub
- De la liste des bibliothèques tierces présentes dans l'environnement python (*requirements.txt*)

Les **tests unitaires** ayant pour objectif de tester indépendamment chaque fonction de l'application sont exécutés durant dette phase. 

### Tests unitaires

Les tests unitaires permettent de tester chaque fonction du code source : 
- Type et dimensions des inputs et outputs de chaque fonction
- Comparaison d'une ou plusieurs réponses
```bash
python -m unittest discover -s Tests -p "*.py"
```

### Le Déploiment

Cette phase consiste à reproduire l'application qui a été build, à partir de l'artefact produit lors de la phase de build : 
- Recréation de l'environnement python 
- Lancement de l'application au démarrage

### Resources locales et sur le Cloud

Les resources des machines virtuelles fournies par MS Azure sont très différentes de celles disponibles sur une machine de développement : 
- Système d'exploitation
- Mémoire 
- Stockage

Il faut donc, optimiser les resources utilisées avant le déploiment : 
- Minimisation des packages python à installer 
  - Packages graphiques inutilisés
  - Packages lourds inutilisés
- Minimisation des fichiers/dossiers lors du build : 
  - Images uploadées sur le repo
  - Notebooks et scripts d'analyse et modélisation
  - Runs MLFlow
- Réduction de la taille du modèle (conversion Tensorflow Lite)


Une fois ces opérations réalisées il est possible d'utiliser l'application créée, mais il est maintenant nécessaire de le maintenir et d'en évaluer les performances, c'est à cet effet que les outils de monitoring proposés par Azure entrent en jeu. 

# Monitoring de l'application

Le monitoring de l'API est réalié via Azure Application Insights, il permet de déclencher des alertes lors de l'utilisation de l'API. Un exemple d'alerte est créé lorsque trois feedbacks négatifs sont renvoyés par l'utilisateur en moins de 5 minutes. 

Lors de la création du monitoring sur Azure, une URI pour le monitoring est crée ainsi qu'une clef d'instrumentation. Ces éléments permettent de lier directement l'API au services Application Insight.

## Logging

Tous les évènements à monitorer sont tracés par l'intermédiaire d'un objet `logger` qui communique directement avec le service Application Insight. 

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

## Création d'alertes 

Les éléments loggés peuvent être consultés sur le portail Azure permettent de définir des règles d'alertes. Cela se fait en trois étapes : 

- Requète sur les logs
- Création de condition d'alerte basée sur les résultats de requètes
- Automatisation d'action au déclenchement de l'alerte (envoi e-mail, SMS etc...)


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

Un modèle d'analyse de sentiment basé sur des tweets a été créé en comparant plusieurs architectures de modèles : 
- Une baseline par regression logistique simple
- Un modèle de réseau de neurone récursif basé sur les cellules LSTM
- Un modèle moderne de transformers basé sur BERT

Une fois le modèle le plus performant  déployé en mettant en place les principes MLOps :

- Analyse de données
- Modélisation (Scikit-learn, TensorFlow) 
- Experimentation et tracking des modèles (MLFlow Tracking)
- Optimisation des hyperparamètres (Optuna) 
- Controle du code source (Git/GitHub)
- Serving API (FastAPI)
- Service de prédiction (Streamlit)
- Déploiment cloud (Azure Web App)
- Monitoring et déclenchement d'alertes (Azure Application Insights)

## Pour aller plus loin

Certains principes du MLOps, comme l'automatisation du pipeline d'entrainement, n'ont pas été implémentés dans  ce projet, pour des raisons de ressources disponibles . Certaines pratiques peuvent cependant être mises en place dans le cadre de la maintenance du modèle.  

- Logging des inputs utilisateurs afin d'étendre la base de données d'entrainement
- Monitoring du taux de mauvaises prédictions : 
  -  Alerte  
  -  Réentrainement du modèle 


