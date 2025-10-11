

# Analyse de sentiments par Deep learning - Modélisation et Déploiment de modèles - une démarche orientée MLOPs 

## Présentation du dataset

Les modèles ont été entrainés sur le dataset ` Sentiment140 dataset with 1.6 million tweets` disponibel en open source sur *Kaggle*. 

Ce dataset est composé de plusieurs information sur des tweets postés entre 2009 et 2020. Ces derniers sont déjà labellisés avec un sentiment défini dans la colonne `target` est les deux classes (sentiment positif ou négatif) sont parfaitement équilibrées.

Plusieurs autres variables sont répertoriées : 

- ids: idezntifiant unique du tweet 
- date: la date de post du tweet 
- flag: Un flag représentant le type de requète
- user: le nom d'utilisateur 
- text: le contenu du tweet 

Ici, seules deux colonnes sont utilisées : la `target` et le `text`. 

## Modélisation

Plusieurs types de modélisations seront utilisés pour prédire la target à partir du text : 
- Modèle Simple linéaire : **Regression Logistique** simple
- Modèle de réseau de neurones :  **LSTM** 
- Modèles à base de transformers :  **BERT**

### Regression Logistique
    
Pour cette première approche la partie la plus importante réside dans le feature engineering qui est appliqué. Pour convertir les chaines de caractères en variables interprétables par un modèle numérique. Celà passe par plusieurs étapes : 
- Nettoyage et normalisation des chaînes de caractères
- Stemming/Lemmatization
- Vectorisation

Une fois ces étapes réalisées il est possible d'entrainer une regression logistique simple sur le dataset. 

Les résultats obtenus avec une telle méthodologie permettent d'atteindre un niveau d'accuracy relativement élevée :  **78%** sur le jeu de validation. 

Ce modèle a l'avantage d'être très léger et de permettre un déploiment rapide et efficace, cependant il n'est pas capable de comprendre un cas complexe, de détecter le sarcasme ou de comprendre la double négation. Ceci est du en grande partie au principe du bag-of-words qui ne prend pas le taxte comme une séquence ordonnée de tokens, mais comme un ensemble de mots sans relations entre eux. 

## Modèle de réseau de neurones

Le modèles de réseaux de neurones de types récursifs (RNN) permettent de compenser le principal défaut des méthodes de bagging. Il considère les textes comme une séquence ordonnée de jetons. 

Pour ces types de modèles, la qualité des prédictions va principalement dépendre de plusieurs étapes : 
- Nettoyage et normalisation des chaines de caractères
- Stemming/Lemmatization
- Embedding
- Type de cellule RNN : RNN Simple, GRU, LSTM
- Longueur des séquences (troncage et/ou bourrage) des séquences 

