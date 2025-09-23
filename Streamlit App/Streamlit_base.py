import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"
FEEDBACK_URL = "http://127.0.0.1:8000/feedback"

st.title("Air Paradis - Sentiment Analysis")

# Entrée utilisateur
tweet_input = st.text_area("Entrez votre tweet ici:")

# Bouton Analyser
if st.button("Analyser"):
    if tweet_input.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        try:
            response = requests.post(API_URL, json={"text": tweet_input})
            response.raise_for_status()
            result = response.json()

            # Stocker les résultats dans session_state
            st.session_state["tweet"] = tweet_input
            st.session_state["positive"] = result["positive"]
            st.session_state["proba"] = result["positive_proba"]

            # Afficher le résultat
            st.write(f"Probabilité positive : {result['positive_proba']:.2f}")
            if result["positive"]:
                st.markdown(f"# **Sentiment :** 😄 Positif")
            else:
                st.markdown(f"# **Sentiment :** 😞 Négatif")

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")

# Afficher feedback seulement si une prédiction a été faite
if "positive" in st.session_state:
    feedback = st.radio(
        "Est-ce la bonne prédiction ?",
        ("Oui", "Non"),
        index=0,
        key="feedback_choice"
    )

    if st.button("Envoyer mon feedback"):
        feedback_value = st.session_state.feedback_choice
        if feedback_value:
            right_answer = st.session_state.positive if feedback_value == "Oui" else not st.session_state.positive

            fb_payload = {
                "text": st.session_state.tweet,
                "prediction": st.session_state.positive,
                "user_feedback": feedback_value,
                "right_answer": right_answer
            }

            try:
                fb_response = requests.post(FEEDBACK_URL, json=fb_payload)
                fb_response.raise_for_status()
                
                # Afficher le message immédiatement après envoi
                st.success("✅ Feedback envoyé à l'API !", icon="✅")

                # Optionnel : réinitialiser le formulaire
                del st.session_state["tweet"]
                del st.session_state["positive"]
                del st.session_state["proba"]
                del st.session_state["feedback_choice"]

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de l'envoi du feedback : {e}")
        else:
            st.warning("❗ Merci de sélectionner Oui ou Non avant d’envoyer.")
