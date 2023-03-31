import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xplotter.insights import *
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import os

############################
# Configuration de la page #
############################
st.set_page_config(
        page_title='Prédiction de la Gravité des Accidents de la Route',
        page_icon = "💻",
        layout="wide" )

# Définition de quelques styles css
st.markdown(""" 
            <style>
            body {font-family:'Roboto Condensed';}
            h1 {font-family:'Roboto Condensed';
                color:#603b1b;
                font-size:2.3em;
                font-style:italic;
                font-weight:700;
                margin:0px;}
            h2 {font-family:'Roboto Condensed';
                color:#8dc5bd;
                font-size:2em;
                font-style:italic;
                font-weight:700;
                margin:0px;}
            p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            </style> """, 
            unsafe_allow_html=True)

st.markdown("""
            <h1>
            6. Prédiction de la Gravité des Accidents de la Route
            </h1>
            """, 
            unsafe_allow_html=True)

st.markdown("""
            Notre étude a pour objectif de modéliser la gravité des accidents de la route en France 
            en se concentrant sur quatre classes d'usagers : indemnes, blessés légers, hospitalisés et tués.
            
            Pour ce faire, nous avons décomposé notre jeu de données en 2 sous-ensembles : 70% pour entraîner 
            notre modèle de prédiction et 30% pour le tester ; en respectant la répartition des 4 classes 
            par "stratification" qui divise le jeu de données de manière à maintenir la même proportion 
            de chaque classe dans les ensembles d'apprentissage et de test. 
            Cette approche est particulièrement utile lorsque certaines classes sont moins représentées que d'autres, 
            car elle garantit que les sous-ensembles contiennent suffisamment d'exemples de chaque classe 
            pour une évaluation précise du modèle.
            """)

###################################
# Lecture et préparation des data #
###################################
@st.cache_data #mise en cache de la fonction pour exécution unique
def load_data():
    data_path = os.path.join('data', 'df_accidents.csv')
    df = pd.read_csv(data_path)
    return df

df_accidents = load_data()

target = df_accidents['gravite_accident']
data = df_accidents.drop(labels = ['gravite_accident', 'heure', 'latitude', 'longitude', 
                         'departement', 'num_commune', 'nom_dep', 'date'], axis=1)
data['annee'] = data['annee'].astype('object')

X_train, X_test, y_train, y_test = train_test_split(data, target, stratify=target, test_size=0.3, random_state=2023)
y_train = pd.DataFrame(y_train, columns=['gravite_accident'])
y_test = pd.DataFrame(y_test, columns=['gravite_accident'])

# Répartition de la variable cible dans les 2 jeux de données
fig = plt.figure(figsize=(20, 6))
ax1 = fig.add_subplot(1,2,1)
plot_countplot(df=y_train,
               col='gravite_accident',
               order=True,
               palette=['#d4b3ac'],
               ax=ax1, orient='v',
               size_labels=12)
plt.grid(visible=False)
plt.title("Répartition des usagers selon la gravité de leur accident\n dans l'ensemble d'entraînement",
          loc="center", fontsize=16, fontstyle='italic', fontweight='bold', color="#5e5c5e")
ax2 = fig.add_subplot(1,2,2)
plot_countplot(df=y_test,
               col='gravite_accident',
               order=True,
               palette=['#d4b3ac'],
               ax=ax2, orient='v',
               size_labels=12)
plt.grid(visible=False)
plt.title("Répartition des usagers selon la gravité de leur accident\n dans l'ensemble de test",
          loc="center", fontsize=16, fontstyle='italic', fontweight='bold', color="#5e5c5e")
plt.grid(False)
fig.tight_layout()
st.pyplot(fig)

st.write("")
st.markdown("""
            **Après voir testé plusieurs modèles de prédiction pour notre classification déséquilibrée, 
            nous avons constater que le modèle qui obtient les performances les plus élevées est le modèle CatBoost.**
            """)
st.markdown("""
            **CatBoost** est un algorithme de boosting d'arbres de décision qui se distingue par son traitement 
            efficace des variables catégorielles, sa régularisation pour éviter le surapprentissage et 
            ses optimisations pour une haute performance. 
            Il est utilisé pour résoudre des problèmes de classification et de régression avec des données mixtes 
            (numériques et catégorielles). 
            
            **CatBoost est particulièrement avantageux lorsqu'il s'agit de traiter des ensembles de données 
            avec un grand nombre de variables catégorielles.**
            
            Et c'est effectivement le cas pour notre problématique puisque **nous ne disposons que de variables 
            catégorielles pour prédire la gravité des accidents de la route en France.**
            
            *Pour plus de précisions sur la méthodologie de sélection du "meilleur" modèle et le modèle CatBoost, 
            vous pouvez lire le [rapport d'étude](https://github.com/DataScientest-Studio/Jan23_BDS_Accidents)*
            """)
image_path = os.path.join('assets', 'choix modèle multiclasses avec catboost.png')
choix_model = Image.open(image_path)
st.image(choix_model, use_column_width="always")
st.markdown("""
            Les résultats obtenus avec le modèle CatBoost optimisé avec Optuna s'interprètent ainsi :
            
            **1. Balanced Accuracy (58%)** : La Balanced Accuracy est une mesure de performance qui prend en compte 
            le déséquilibre des classes. Elle est calculée comme la moyenne des taux de rappel pour chaque classe. 
            Une Balanced Accuracy de 58% indique que le modèle est capable de classer correctement les observations 
            pour chaque classe environ 58% du temps, ce qui est supérieur à une classification aléatoire.
            
            **2. Weighted Precision (67%)** : La précision pondérée est la moyenne des précisions pour chaque classe, 
            pondérée par le nombre d'observations de chaque classe. Une Weighted Precision de 67% signifie que, 
            lorsque le modèle prédit une classe, il est correct environ 67% du temps.
            
            **3. Weighted Recall (63%)** : Le rappel pondéré est la moyenne des rappels pour chaque classe, 
            pondérée par le nombre d'observations de chaque classe. Un Weighted Recall de 63% signifie que le modèle 
            identifie correctement environ 63% des observations de chaque classe.
            
            **4. Weighted F1-score (64%)** : Le F1-score pondéré est la moyenne harmonique des précisions et rappels 
            pondérés pour chaque classe. Un F1-score équilibré de 64% indique que le modèle a un bon équilibre 
            entre la précision et le rappel pour chaque classe, compte tenu du déséquilibre des classes.
            
            **5. Average AUC (85%)** : L'aire sous la courbe ROC (AUC) est une mesure de performance qui évalue 
            la capacité d'un modèle à discriminer entre les classes positives et négatives. 
            Un Average AUC de 85% indique que le modèle a une bonne capacité à distinguer entre les différentes classes.
            
            **6. Score Géo Global (72%)**  :  Le Score Géo Global est la moyenne géométrique (ou G-Mean) des rappels 
            pour chaque classe. Cette mesure de performance tient compte à la fois de la capacité du modèle à classifier 
            correctement les différentes classes et de l'équilibre entre les classes. Un G-Mean élevé indique que 
            le modèle est performant pour toutes les classes, y compris celles qui sont moins représentées. Un G-Mean 
            de 72% montre que le modèle est capable de classer correctement les observations pour chaque classe, 
            y compris les classes minoritaires, environ 72% du temps. Cela indique que le modèle a une performance 
            globale relativement élevée en tenant compte à la fois de la capacité à classifier les différentes classes 
            et de l'équilibre entre les classes.
            """)

########################
# Chargement du modèle #
########################
model_path = os.path.join('models', 'model_Catboost.pkl')
model = pickle.load(open(model_path, "rb"))

st.markdown("""
            <h1>
            7. Prédisez la Gravité de l'Accident de la Route
            </h1>
            """, 
            unsafe_allow_html=True)
st.markdown("""
            Décrivez l'accident de la route en sélectionnant à l'aide des filtres ci-desous ses caractéristiques.<br>
            Puis cliquez sur le bouton "Prédiction" pour en prédire la gravité.
            """,
            unsafe_allow_html=True)

########################################################################
# Ce code utilise st.session_state pour gérer l'état de session.       #
# Les prédictions seront réinitialisées chaque fois que l'utilisateur  #
# modifie la sélection de modalités.                                   #
########################################################################

# Initialisez l'état de session si nécessaire
if "prediction_text" not in st.session_state:
    st.session_state["prediction_text"] = None

if "prob_df" not in st.session_state:
    st.session_state["prob_df"] = None

if "reset_clicked" not in st.session_state:
    st.session_state["reset_clicked"] = False


# Récupérez et triez les variables qualitatives par ordre alphabétique
qualitative_vars = sorted(X_train.columns.tolist())

# Créez un dictionnaire pour stocker les valeurs sélectionnées
selected_values = {}

# Définissez les largeurs des colonnes principales et des colonnes vides
main_col_width = 6
empty_col_width = 0.5

# Divisez l'espace en 5 colonnes avec des largeurs personnalisées
cols = st.columns([main_col_width, empty_col_width, main_col_width, empty_col_width, main_col_width])

# Attribuez les colonnes principales aux indices 0, 2 et 4
col1, col2, col3 = cols[0], cols[2], cols[4]

# Utilisez une boucle pour créer les selectbox pour chaque variable qualitative
for idx, var in enumerate(qualitative_vars):
    # Récupérez les modalités uniques pour la variable en cours
    unique_values = X_train[var].unique().tolist()

    # Choisissez la colonne en fonction de l'index de la variable
    if idx % 3 == 0:
        current_col = col1
    elif idx % 3 == 1:
        current_col = col2
    else:
        current_col = col3

    # Créez une selectbox pour la variable en cours dans la colonne correspondante
    selected_value = current_col.selectbox(f"Sélectionnez une modalité pour **{var}** :", unique_values)

    # Stockez la valeur sélectionnée dans le dictionnaire
    selected_values[var] = selected_value

# Affichez le dictionnaire avec les valeurs sélectionnées
st.write("Valeurs sélectionnées pour chaque variable :", selected_values)

# Ajoutez un style CSS personnalisé pour personnaliser le bouton
button_css = """
<style>
.custom-button {
    background-color: #135e58;
    color: white;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
.custom-button:hover {
    background-color: #0f4a40;
}
</style>
"""
# Créez un bouton personnalisé en utilisant st.markdown()
st.markdown(button_css, unsafe_allow_html=True)
clicked = st.markdown(
    '<button class="custom-button" onclick="document.querySelector(\'form\').submit()">Cliquez pour connaître la prédiction de la gravité de l\'accident</button>',
    unsafe_allow_html=True,
)

# Vérifiez si le bouton est cliqué
if clicked:
    st.session_state.reset_clicked = False

    # Transformez les sélections en un DataFrame
    input_data = pd.DataFrame([selected_values])

    # Effectuez la prédiction avec le modèle
    prediction = model.predict(input_data)

    # Récupérez la prédiction sous forme de texte sans crochets
    st.session_state.prediction_text = prediction.item()

    # Obtenez les probabilités pour chaque classe
    probabilities = model.predict_proba(input_data)

    # Créez un DataFrame avec les probabilités et les noms de classe
    st.session_state.prob_df = pd.DataFrame(probabilities, columns=model.classes_)

    # Trier les colonnes dans l'ordre souhaité
    st.session_state.prob_df = st.session_state.prob_df[['indemne', 'blessé_léger', 'blessé_hospitalisé', 'tué']]

    # Convertissez les probabilités en pourcentage et arrondissez à 1 chiffre décimal
    st.session_state.prob_df = (st.session_state.prob_df * 100).round(1)

# Ajoutez un bouton pour effacer la prédiction et réinitialiser l'application
reset_button = st.button("Je souhaite refaire une prédiction et changer les caractéristiques de l'accident")

if reset_button:
    st.session_state.prediction_text = None
    st.session_state.prob_df = None
    st.session_state.reset_clicked = True

# Affichez le résultat de la prédiction et les probabilités si disponibles et si le bouton de réinitialisation n'a pas été cliqué
if not st.session_state.reset_clicked and st.session_state.prediction_text and st.session_state.prob_df is not None:
    st.write(f"La prédiction de la gravité de l'accident est : {st.session_state.prediction_text}")
    st.write("Les probabilités pour chaque classe sont (en %) :")
    st.write(st.session_state.prob_df)

#Stockez la prédiction et les données d'entrée dans st.session_state
st.session_state.prediction = prediction
st.session_state.probabilities = probabilities
st.session_state.input_data = input_data
st.markdown("""
            ***Pour comprendre cette prédiction, rendez-vous à la page suivante en cliquant sur
            'Explication de la prédiction' dans la barre latérale gauche.***
            """)


# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image_path = os.path.join('assets', 'logo-datascientest.png')
    logo = Image.open(image_path)
    st.sidebar.image(logo, use_column_width="always")
with col3:
    st.sidebar.write("")