from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

DOSSIER = Path(__file__).parent

NOMS_FR = {
    "sadness": "Tristesse",
    "joy": "Joie",
    "love": "Amour",
    "anger": "Colère",
    "fear": "Peur",
    "surprise": "Surprise",
}

EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

EXEMPLES = [
    "i feel so happy today",
    "i am really scared of tomorrow",
    "i can't believe they did this to me",
    "i was shocked when i saw the news",
]


@st.cache_resource
def charger_modele():
    modele = joblib.load(DOSSIER / "svm_proba.joblib")
    tfidf = joblib.load(DOSSIER / "tfidf.joblib")
    label_names = joblib.load(DOSSIER / "label_names.joblib")
    return modele, tfidf, label_names


def predire(texte, modele, tfidf, label_names):
    vecteur = tfidf.transform([texte])
    probas = modele.predict_proba(vecteur)[0]
    resultats = pd.DataFrame({"emotion": label_names, "probabilite": probas})
    return resultats.sort_values("probabilite", ascending=False).reset_index(drop=True)


st.set_page_config(page_title="Détecteur d'émotions", page_icon="🎭", layout="centered")

st.title("🎭 Détecteur d'émotions")
st.write(
    "Écris une phrase **en anglais** ou dépose un fichier texte, et le modèle prédit l'émotion "
    "parmi six : tristesse, joie, amour, colère, peur et surprise."
)
st.caption(
    "Modèle : SVM linéaire calibré sur des vecteurs TF-IDF, entraîné sur le dataset Emotion "
    "(dair-ai). F1 macro de 0,957 sur le jeu de test."
)

modele, tfidf, label_names = charger_modele()

onglet_texte, onglet_fichier = st.tabs(["✍️ Écrire un texte", "📄 Déposer un fichier .txt"])

with onglet_texte:
    exemple = st.selectbox("Ou choisis un exemple :", [""] + EXEMPLES)
    texte_saisi = st.text_area(
        "Ton texte (en anglais) :",
        value=exemple,
        height=120,
        placeholder="ex : i feel so happy today",
    )

with onglet_fichier:
    fichier = st.file_uploader("Fichier texte (.txt)", type=["txt"])
    texte_fichier = ""
    if fichier is not None:
        texte_fichier = fichier.read().decode("utf-8", errors="ignore")
        st.text_area("Contenu du fichier :", value=texte_fichier, height=120, disabled=True)

texte = texte_fichier.strip() if texte_fichier.strip() else texte_saisi.strip()

if st.button("Analyser l'émotion", type="primary", use_container_width=True):
    if not texte:
        st.warning("Écris un texte ou dépose un fichier avant de lancer l'analyse.")
    else:
        resultats = predire(texte, modele, tfidf, label_names)
        meilleure = resultats.iloc[0]
        emotion = meilleure["emotion"]
        proba = meilleure["probabilite"]

        st.subheader("Résultat")
        col1, col2 = st.columns(2)
        col1.metric("Émotion prédite", f"{EMOJIS[emotion]} {NOMS_FR[emotion]}")
        col2.metric("Probabilité", f"{proba:.0%}")

        if proba < 0.5:
            deuxieme = resultats.iloc[1]
            st.info(
                f"Le modèle hésite : il donne aussi {deuxieme['probabilite']:.0%} à "
                f"{NOMS_FR[deuxieme['emotion']].lower()}. Il se base surtout sur les mots "
                "d'émotion présents dans la phrase."
            )

        st.write("Probabilité de chaque émotion :")
        tableau = resultats.copy()
        tableau["emotion"] = tableau["emotion"].map(lambda e: f"{EMOJIS[e]} {NOMS_FR[e]}")
        st.bar_chart(tableau.set_index("emotion")["probabilite"], horizontal=True)

st.divider()
st.caption(
    "Projet final – Mastère Dev, Data & IA. Le modèle a été entraîné sur des tweets courts en "
    "anglais : il marche mieux sur des phrases courtes qui contiennent un mot d'émotion."
)
