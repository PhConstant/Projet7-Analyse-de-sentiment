import streamlit as st
import requests

# -----------------------------
# Config API
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"
FEEDBACK_URL = "http://127.0.0.1:8000/feedback"

st.set_page_config(page_title="Air Paradis - Sentiment Analysis", layout="centered")

# -----------------------------
# Session state
# -----------------------------
if "tweet" not in st.session_state:
    st.session_state["tweet"] = ""
if "result" not in st.session_state:
    st.session_state["result"] = None
if "feedback_sent" not in st.session_state:
    st.session_state["feedback_sent"] = False

# -----------------------------
# Titre
# -----------------------------
st.title("Air Paradis - Sentiment Analysis")

# -----------------------------
# Zone de texte
# -----------------------------
tweet_input = st.text_area(
    "Entrez votre tweet ici:",
    value=st.session_state["tweet"],
    key="tweet_input"
)

# -----------------------------
# Bouton Analyser
# -----------------------------
if st.button("Analyser"):
    tweet = st.session_state["tweet_input"].strip()
    if tweet == "":
        st.warning("Veuillez entrer un texte.")
    else:
        try:
            response = requests.post(API_URL, json={"text": tweet})
            response.raise_for_status()
            st.session_state["result"] = response.json()
            st.session_state["tweet"] = tweet
            st.session_state["feedback_sent"] = False
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")

# -----------------------------
# Affichage du résultat
# -----------------------------
if st.session_state["result"]:
    result = st.session_state["result"]
    positive = result["positive"]
    proba = result["positive_proba"]

    st.write(f"Probabilité positive : {proba:.2f}")
    st.markdown(f"# **Sentiment :** {'😄 Positif' if positive else '😞 Négatif'}")

    # -----------------------------
    # Feedback
    # -----------------------------
    if not st.session_state["feedback_sent"]:
        feedback_choice = st.radio(
            "Est-ce la bonne prédiction ?",
            ("Oui", "Non"),
            index=0,
            key="feedback_choice"
        )

        if st.button("Envoyer mon feedback"):
            right_answer = positive if feedback_choice == "Oui" else not positive
            fb_payload = {
                "text": st.session_state["tweet"],
                "prediction": positive,
                "user_feedback": feedback_choice,
                "right_answer": right_answer
            }

            try:
                fb_response = requests.post(FEEDBACK_URL, json=fb_payload)
                fb_response.raise_for_status()

                # Affichage du message de succès
                st.success("Feedback envoyé à l'API", icon="✅")
                st.session_state["feedback_sent"] = True

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de l'envoi du feedback : {e}")

# -----------------------------
# Bouton Tweet suivant 
# -----------------------------
if st.session_state.get("feedback_sent", False):
    if st.button("Tweet suivant"):
        for key in ["tweet", "result", "feedback_sent", "feedback_choice"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()
