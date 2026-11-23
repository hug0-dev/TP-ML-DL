# Détecteur d'émotions

Projet final du Mastère Dev, Data & IA : classification de textes courts en anglais parmi 6 émotions (sadness, joy, love, anger, fear, surprise).

**Webapp en ligne : [à compléter]**

## Contenu du dépôt

- `projet_emotion_big.ipynb` : le notebook avec l'analyse du dataset, la comparaison des modèles (Naive Bayes, SVM, KNN, RNN) et le test final
- `projet_emotion_demarche.docx` : le document qui explique la démarche
- `app.py` : la webapp Streamlit
- `svm_proba.joblib`, `tfidf.joblib`, `label_names.joblib` : le modèle retenu (SVM calibré), le TF-IDF et les noms des émotions, sauvegardés depuis le notebook
- `requirements.txt` : les bibliothèques à installer

## Résultat

Le modèle retenu est un SVM linéaire calibré sur des vecteurs TF-IDF. Sur le jeu de test, il obtient 0,965 d'accuracy et 0,957 de F1 macro.

## Lancer la webapp en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
