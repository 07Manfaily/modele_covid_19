import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('nettoyé.csv')  # Remplacez par le chemin de votre fichier
    return df

df = load_data()
# st.write("Colonnes disponibles dans le dataset :", df.columns)

# Prétraitement des données
comorbidities = ['DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'CARDIOVASCULAR',
                 'RENAL_CHRONIC', 'OBESITY', 'TOBACCO']
df['total_comorbidities'] = df[comorbidities].apply(lambda row: (row == 1).sum(), axis=1)

# Sélectionner les features et la target
features = ['AGE', 'total_comorbidities'] + comorbidities
X = df[features]
y = df['CLASIFFICATION_FINAL']  # Classe finale (positif, négatif, non concluant)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle (RandomForest pour cet exemple)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prédictions et évaluation du modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Affichage du rapport de classification
report = classification_report(y_test, y_pred)

# Interface Streamlit
st.title("Détection de cas COVID-19")

# Affichage des résultats
st.write(f"**Précision du modèle :** {accuracy:.2f}")
st.write("**Rapport de classification :**")
st.text(report)

# Formulaire d'entrée pour prédiction
st.sidebar.header("Entrez les données d'un patient")

# Définir les entrées utilisateur dans le formulaire
age = st.sidebar.slider("Âge", 0, 100, 30)
diabetes = st.sidebar.selectbox("Diabète", ["Non", "Oui"])
copd = st.sidebar.selectbox("COPD", ["Non", "Oui"])
asthma = st.sidebar.selectbox("Asthme", ["Non", "Oui"])
inmsupr = st.sidebar.selectbox("Immunosupprimé", ["Non", "Oui"])
hypertension = st.sidebar.selectbox("Hypertension", ["Non", "Oui"])
cardio = st.sidebar.selectbox("Cardiovasculaire", ["Non", "Oui"])
renal_chronic = st.sidebar.selectbox("Maladie rénale chronique", ["Non", "Oui"])
obesity = st.sidebar.selectbox("Obésité", ["Non", "Oui"])
tobacco = st.sidebar.selectbox("Tabagisme", ["Non", "Oui"])

# Convertir les réponses en format numérique
input_data = {
    "AGE": age,
    "DIABETES": 1 if diabetes == "Oui" else 0,
    "COPD": 1 if copd == "Oui" else 0,
    "ASTHMA": 1 if asthma == "Oui" else 0,
    "INMSUPR": 1 if inmsupr == "Oui" else 0,
    "HIPERTENSION": 1 if hypertension == "Oui" else 0,
    "CARDIOVASCULAR": 1 if cardio == "Oui" else 0,
    "RENAL_CHRONIC": 1 if renal_chronic == "Oui" else 0,
    "OBESITY": 1 if obesity == "Oui" else 0,
    "TOBACCO": 1 if tobacco == "Oui" else 0,
}

# Créer un dataframe avec les données de l'utilisateur
input_df = pd.DataFrame([input_data])

# Assurez-vous que l'input_df a bien les mêmes colonnes que le modèle attend
input_df['total_comorbidities'] = input_df[comorbidities].sum(axis=1)  # Calculer le nombre de comorbidités
input_df = input_df[features]  # Réordonner pour correspondre aux features

# Prédiction avec les données de l'utilisateur
prediction = model.predict(input_df)

# Affichage de la prédiction
st.write(f"**Prédiction pour ce patient :** {prediction[0]}")
