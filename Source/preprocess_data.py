## preprocess_data.py
# Source/preprocess_data.py
# Fonctions de pré-traitement des données textuelles
# Auteur : Philippe CONSTANT
import re
import html
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, GRU, LSTM, Bidirectional, GlobalMaxPooling1D, Dropout, Concatenate,GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub



def normalize_text(doc: str) -> str:
    """
    Normalisation avant tokenisation :
      - lowercasing
      - masquage URLs, @user, #hashtag
      - réduction des répétitions de lettres
    """
    # 1. Lowercasing
    text = doc.lower()
    # 2. Décodage entités HTML$
    old = None
    while old != text:
        old = text
        text = html.unescape(text)
    # 3. URLs
    text = re.sub(r"http\S+|www\S+", " website ", text)
    # 4. Mentions
    text = re.sub(r"@\w+", " user ", text)
    # 5. Hashtags
    text = re.sub(r"#\w+", " hashtag ", text)
    # 6. Réduction des répétitions (>=3 mêmes lettres)
    # a) voyelles répétées >=2 → 1
    text = re.sub(r'([aiuy])\1{1,}', r'\1\1', text, flags=re.IGNORECASE)
    # b) consonnes répétées >=3 → 1
    text = re.sub(r'([^aiuy\.])\1{2,}', r'\1', text, flags=re.IGNORECASE)
    # 7. Nettoyage espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def remove_stopwords(text, stop_words):
    out = [w for w in text if w not in stop_words]
    return out

def tokenize_texts(text, tokenizer_func, join_final=False):
    """
    Tokenisation des textes avec différentes méthodes.
    """
    tokens = tokenizer_func(text)
    if join_final:
        tokens = [" ".join(token_list) for token_list in tokens]
    return tokens

def stem_or_lem_tokens(tokens, Stem_lem_func):
    """
    Stemming ou lemmatisation des tokens.
    """
    processed_tokens = [Stem_lem_func(w) for w in tokens]
    return processed_tokens


def preprocess_text_simple(
        text:str,  
        tokenizer, 
        stem_lem_func,
        stop_words = None
        )-> str:
    """
    Prétraitement complet du texte :
      - Normalisation des textes
      - Tokenisation
      - (Stemming ou lemmatisation)
    """
    # 1. Normalisation
    out = normalize_text(text)
    # 2. Tokenisation
    out = tokenize_texts(out, tokenizer, join_final=False)
    # 3. retrait des stopwords
    if stop_words is not None:
        out = remove_stopwords(out, stop_words) 
    # 4. Stemming ou lemmatisation (optionnel)
    out = stem_or_lem_tokens(out, stem_lem_func)

    out = " ".join(out)

    return out

def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+", " ", text)     # enlever les URLs
        text = re.sub(r"@\w+", " ", text)        # enlever les mentions
        text = re.sub(r"#\w+", " ", text)        # enlever les hashtags
        text = re.sub(r"[^a-zA-Z\s]", " ", text) # garder que lettres
        text = re.sub(r"\s+", " ", text).strip() # espaces multiples
        return text


def process_tokens(text, stem_lem_func, stop_words):
        tokens = text.split()
        if stop_words:
            tokens = [w for w in tokens if w not in stop_words]
        
        if stem_lem_func:
            tokens = [stem_lem_func(w) for w in tokens]
        return " ".join(tokens)


def remove_rare_words(tokens, word_counts, min_count):
    return [w for w in tokens if word_counts[w] >= min_count]


def preprocess_data_embedding(X_raw,
                             stem_lem_func,
                              tokenizer = None, # Si il n'y a pas de tokenizer en entrée alors on l'entraine directement dans le prétraitement
                              stop_words = None, 
                              min_count = 2,
                              max_len = 10, 
                              num_words = 10000, # Utile si tokenizer est None
                              return_sentences = False
                                  ): 
    # Prétraitement du texte simple
    X_clean = X_raw.apply(clean_text)

    X_processed = X_clean.apply(lambda x:process_tokens(x,stem_lem_func=stem_lem_func, stop_words=stop_words))
    # retrait des tokens rares

    all_tokens = [token for tweet in X_processed for token in tweet.split()]
    word_counts = Counter(all_tokens)

    X_filtered = X_processed.apply(lambda x: " ".join(remove_rare_words(x.split(), word_counts, min_count)))

    
    if return_sentences:
        sentences = [tweet.split() for tweet in X_filtered]
    # Tokenizer Keras
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_filtered)
    X_sequences = tokenizer.texts_to_sequences(X_filtered)
    X_sentences = pad_sequences(X_sequences, maxlen=max_len, padding="post", truncating="post")

    if return_sentences:
        return X_sentences, tokenizer, sentences
    else:
        return X_sentences




def build_embedding_matrix(tokenizer, embedding_model, latent_dim=100):
    """
    Crée une matrice d'embedding à partir d'un tokenizer et d'un modèle pré-entraîné.

    Args:
        tokenizer: objet Keras Tokenizer déjà fit sur le corpus.
        embedding_model: modèle d'embedding (Word2Vec, GloVe, FastText chargé avec gensim).
                         -> il doit être indexable comme embedding_model[word]
        latent_dim: dimension de l'espace latent (int).
    
    Returns:
        embedding_matrix (np.array), vocab_size (int)
    """
    
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1  # +1 car index 0 est réservé (padding)

    embedding_matrix = np.zeros((vocab_size, latent_dim))

    nb_found = 0
    for word, idx in word_index.items():
        if word in embedding_model:  # si le mot est connu du modèle
            vector = embedding_model[word] 
            if vector is not None and len(vector) == latent_dim:
                embedding_matrix[idx] = vector
                nb_found += 1

    coverage = round(nb_found / vocab_size, 4)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Words found in pretrained embeddings: {nb_found}/{vocab_size} ({coverage*100:.2f}%)")

    return embedding_matrix, vocab_size


def build_base_RNN(vocab_size, latent_dim,input_length, embedding_matrix,rnn_size = 64):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=latent_dim,
            weights=[embedding_matrix],
            input_length=input_length,
            mask_zero=True,
            trainable=False  # False = geler les embeddings

        ),
        SimpleRNN(rnn_size, return_sequences=True, activation="tanh"),
        GlobalMaxPooling1D(),                         
        Dense(1, activation="sigmoid")
    ])
    return model

def build_gru_RNN(vocab_size, latent_dim,input_length, embedding_matrix,rnn_size = 64):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=latent_dim,
            weights=[embedding_matrix],
            input_length=input_length,
            mask_zero=True,
            trainable=False  # False = geler les embeddings

        ),
        GRU(rnn_size, return_sequences=True, activation="tanh"),   
        GlobalMaxPooling1D(),                       
        Dense(1, activation="sigmoid")
    ])
    return model

def build_lstm_RNN(vocab_size, latent_dim,input_length, embedding_matrix,rnn_size = 64):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=latent_dim,
            weights=[embedding_matrix],
            input_length=input_length,
            mask_zero=True,
            trainable=False  # False = geler les embeddings

        ),
        LSTM(rnn_size, return_sequences=True, activation="tanh"),  
        GlobalMaxPooling1D(),                       
        Dense(1, activation="sigmoid")
    ])
    return model


def build_bilstm_RNN(vocab_size, latent_dim,input_length, embedding_matrix,rnn_size = 64):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=latent_dim,
            weights=[embedding_matrix],
            input_length=input_length,
            mask_zero=True,
            trainable=False
        ),
        Bidirectional(LSTM(
            rnn_size,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.3
        )),        
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(rnn_size//2, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
        ])
    
    model.summary()
    return model

def build_use_model(dense_size):
    use_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[],
    dtype=tf.string,
    trainable=False,  
    name="USE_embedding")

    model = Sequential([
        use_layer, 
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  
        
    ])
    model.summary()
    return model