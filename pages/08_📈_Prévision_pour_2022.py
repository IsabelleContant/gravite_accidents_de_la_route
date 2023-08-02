import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image
import plotly.express as px
import plotly.graph_objs as go
import os
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.serialize import model_from_json
import plotly.offline as py
import json

############################
# Configuration de la page #
############################
st.set_page_config(
        page_title="Prévision 2022 de nombre de victimes d'accidents de la route avec Prophet",
        page_icon = "📈",
        layout="wide" )

# Définition de quelques styles css
st.markdown(""" 
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed&display=swap');
            body {font-family:'Roboto Condensed';}
            h1 {font-family:'Roboto Condensed';
                color:#603b1b;
                font-size:2.3em;
                font-style:italic;
                font-weight:700;
                margin:0px;}
            h2 {font-family:'Roboto Condensed';
                color:#9EBEB8;
                font-size:2em;
                font-style:italic;
                font-weight:700;
                margin:0px;}
            p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .css-16idsys p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem; margin: 0px;}
            .css-5rimss li {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .css-184tjsw p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
    word-break: break-word;
    /* font-size: 14px; */
}
            </style> """, 
            unsafe_allow_html=True)

########################################
# Lecture des fichiers et des modèles  #
########################################

# Chargement des données réelles
@st.cache_data
def load_data():
    data_path = os.path.join('data', 'nbr_acc_jour.csv')
    df = pd.read_csv(data_path)
    df.rename(columns={'date': 'ds'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds']) # Convertir la colonne 'ds' en datetime
    return df

data = load_data()

# Fonction pour charger les modèles à partir des fichiers JSON
@st.cache_data
def load_model(file_name):
    file_path = os.path.join('models', file_name)
    with open(file_path, 'r') as f:
        model = model_from_json(json.load(f))
    return model

# Chargement des modèles
models = {
    'total_accidents': load_model('Prophet_model_tot_acc.json'),
    'gravite_accident_tué': load_model('Prophet_model_acc_tués.json'),
    'gravite_accident_blessé_léger': load_model('Prophet_model_acc_legers.json'),
    'gravite_accident_blessé_hospitalisé': load_model('Prophet_model_acc_hosp.json'),
    'gravite_accident_indemne': load_model('Prophet_model_acc_indemnes.json')
}

# Fonction pour Charger les régresseurs utilisés dans chaque modèle
@st.cache_data
def load_regressors(model_name):
    file_path = os.path.join('data', f"{model_name}_regressors.csv")
    regressors = pd.read_csv(file_path)
    regressors['ds'] = pd.to_datetime(regressors['ds']) # Convertir la colonne 'ds' en datetime
    return regressors

####################################################
# Titre et méthodologie de la modélisation Prophet #
####################################################
st.markdown("""
            <h1>
            13. Prévision 2022 du nombre de victimes d'accidents de la route en France
            </h1>
            """, 
            unsafe_allow_html=True)

st.markdown("""
Les chiffres définitifs de la sécurité routière pour 2022 ne seront disponibles qu'en Décembre 2023.

C'est pourquoi, pour répondre au **projet fil rouge de la formation MLOps (Machine Learning Engineer)**, nous avons décidé de continuer 
le projet fil rouge réalisé dans le cadre de la formation Datascientist et de <span style="color:#9EBEB8; font-style:italic; font-weight:700;">
tenter de prédire le nombre de victimes d'accidents de la route pour l'ensemble des 365 jours de l'année 2022 et ce 
pour chacune des 4 types de gravité : indemnes, blessés légers, blessés hospitalisés et tués. </span>

Pour ce faire, nous avons utilisé la bibliothèque open-source Prophet développée par Facebook et publiée en 2017 (*Cliquez dans l'encadré 
ci-dessous pour en savoir plus sur Prophet*).
""", unsafe_allow_html=True)

# Ajout d'un texte dans un encadré sur lequel on peut cliquer
expander = st.expander("***Qu'est-ce que Prophet ?***")
expander.markdown("""
Prophet est une bibliothèque open-source développée par Facebook et publiée en 2017 qui est conçue pour la prévision de séries temporelles à l'échelle. 
Elle est capable de gérer les tendances, la saisonnalité et les jours fériés, et offre une grande flexibilité pour le réglage des modèles.

Plus en détail, sur sa page Web de projets open source, Facebook déclare que : *"Prophet est une procédure de prévision de données de séries 
chronologiques basée sur un modèle additif où les tendances non linéaires sont adaptées à la saisonnalité annuelle, hebdomadaire et quotidienne, 
ainsi qu'aux effets des vacances . Cela fonctionne mieux avec des séries chronologiques qui ont de forts effets saisonniers et plusieurs saisons 
de données historiques. Prophet est robuste aux données manquantes et aux changements de tendance, et gère généralement bien les valeurs aberrantes."*

Prophet implémente un modèle additif général où chaque série chronologique y(t) est modélisée comme la combinaison linéaire d'une tendance g(t),
d'une composante saisonnière s(t), d'effets de vacances h(t) et d'un terme d'erreur ϵt , qui est normalement distribué.
$$
y(t)=g(t)+s(t)+h(t)+\epsilon t
$$

La composante de tendance modélise les changements à long terme non périodiques dans la série chronologique. 
La composante saisonnière modélise le changement périodique, qu'il soit annuel, mensuel, hebdomadaire ou quotidien. 
L'effet vacances se produit de manière irrégulière et potentiellement sur plus d'un jour. 
Enfin, le terme d'erreur représente tout changement de valeur qui ne peut être expliqué par les trois composantes précédentes.

Ce modèle ne prend pas en compte la dépendance temporelle des données, contrairement au modèle ARIMA(p, d, q), 
où les valeurs futures dépendent des valeurs passées. Ainsi, ce processus est plus proche de l'ajustement d'une courbe aux données, 
plutôt que de la recherche du processus sous-jacent. 
Bien qu'il y ait une certaine perte d'informations prédictives à l'aide de cette méthode, elle présente l'avantage d'être très flexible, 
car elle peut s'adapter à plusieurs périodes saisonnières et à des tendances changeantes. 
De plus, elle est robuste aux valeurs aberrantes et aux données manquantes, ce qui est un avantage évident.

L'inclusion de plusieurs périodes saisonnières a été motivée par l'observation que le comportement humain produit des séries chronologiques 
saisonnières sur plusieurs périodes. 
Par exemple, la semaine de travail de cinq jours peut produire un modèle qui se répète chaque semaine, 
tandis que les vacances scolaires peuvent produire un modèle qui se répète chaque année. 
Ainsi, pour prendre en compte plusieurs périodes saisonnières, Prophet utilise la série de Fourier pour modéliser plusieurs effets périodiques. 
Plus précisément, la composante saisonnière s(t) est exprimée par l'équation ci-dessous, où P est la durée de la période saisonnière en jours 
et N est le nombre de termes de la série de Fourier.
""", unsafe_allow_html=True)
expander.latex(r'''s(t)=\sum_{n=1}^N\left(a_n \cos \left(\frac{2 \pi n t}{P}\right)+b_n \sin \left(\frac{2 \pi n t}{P}\right)\right)''')
expander.markdown("""
Dans l'équation de Fourier, si nous avons une saisonnalité annuelle, P = 365,25 (car il y a 365,25 jours dans une année), pour une saisonnalité 
hebdomadaire P = 7, N est simplement le nombre de paramètres que nous souhaitons utiliser pour estimer la composante saisonnière. 
Cela présente l'avantage supplémentaire que la sensibilité de la composante saisonnière peut être ajustée en fonction du nombre de paramètres N 
estimés pour modéliser la saisonnalité. Par défaut, Prophet utilise 10 termes pour modéliser la saisonnalité annuelle et 3 termes pour modéliser 
la saisonnalité hebdomadaire.

Enfin, ce modèle permet de considérer l'effet des vacances. Les jours fériés sont des événements irréguliers qui peuvent avoir un impact clair 
sur une série temporelle. Par exemple, des événements comme le Black Friday aux États-Unis peuvent augmenter considérablement la fréquentation 
en magasin ou les ventes sur un site de commerce électronique. De même, la Saint-Valentin est probablement un indicateur fort d'une augmentation 
des ventes de chocolats et de fleurs. Par conséquent, pour modéliser l'impact des jours fériés dans une série chronologique, Prophet nous permet 
de définir une liste de jours fériés pour un pays spécifique. Les effets des vacances sont ensuite intégrés dans le modèle, en supposant qu'ils sont
tous indépendants. Si un point de données tombe sur une date de vacances, un paramètre Ki est calculé pour représenter le changement 
dans la série chronologique à ce moment précis. Plus le changement est important, plus l'effet vacances est important.

La flexibilité de Prophet peut en faire un choix attrayant pour des prévisions rapides et précises.
Cependant, il ne doit pas être considéré comme une solution unique. La documentation elle-même précise que Prophet fonctionne mieux avec 
des séries chronologiques qui ont un fort effet saisonnier avec plusieurs saisons de données historiques. Par conséquent, il peut y avoir 
des situations où Prophet n'est pas le choix idéal.
""", unsafe_allow_html=True)
st.write("")
st.markdown("""
<p style="color: #603b1b; font-weight:700; font-style:italic; margin: 0px;">
L'objectif principal du projet fil rouge MLOps n'est pas de construire des modèles de prévision : il s'agit de mettre en place une chaîne de traitement
complète de Machine Learning, de la récupération des données à la mise en production du modèle, c'est-à-dire d'implémenter tout le cycle de vie 
du modèle en production.</p>""", unsafe_allow_html=True)
st.write("")

##################################################################################
# Sélection de la Variable à prédire et de la période de temps par l'utilisateur #
##################################################################################
st.write("")
st.markdown("""
            <h2>
            Prévision 2022 du nombre de victimes d'accidents de la route en France
            </h2>
            """, 
            unsafe_allow_html=True)

st.markdown("""
Rentrons dans le vif du sujet !

Pour obtenir des prévisions quotidienne pour l'année 2022, vous devez d'abord sélectionner ce que vous souhaitez prédire parmi :
- le nombre total d'accidentés de la route
- le nombre de victimes indemnes
- le nombre de victimes légèrement blessées
- le nombre de victimes blessées hospitalisées
- et le nombre de victimes tuées

Vous devez aussi choisir la période de temps, de 1 à 365 jours, dont vous voulez obtenir la prédiction.
""", unsafe_allow_html=True)

# Définissez les largeurs des colonnes principales et des colonnes vides
main_col_width = 6
empty_col_width = 0.5
# Divisez l'espace en 3 colonnes avec des largeurs personnalisées
cols = st.columns([main_col_width, empty_col_width, main_col_width])
# Attribuez les colonnes principales aux indices 0 et 3
col1, col2, col3 = cols[0], cols[1], cols[2]
with col1:
    variable = st.selectbox('***Choisissez la variable dont vous souhaitez connaître les prédictions***', list(models.keys()))
with col3:
    days = st.slider('***Choisissez le nombre de jours que vous souhaitez prédire***', 1, 365)

############################
# Création des prédictions #
############################

# Chargement des régresseurs
regressors = load_regressors(variable).drop(columns=['Unnamed: 0'], axis=1)

# Création du DataFrame future
future = models[variable].make_future_dataframe(periods=days, freq='D', include_history=True)

# Fusion de future avec les régresseurs sur la colonne 'ds'
future = pd.merge(future, regressors, on='ds', how='left')

# Faire la prédiction
forecast = models[variable].predict(future)

#################
# Graphique n°1 #
#################
start_date = datetime(2022, 1, 1) # Début de la période de prédiction
end_date = start_date + timedelta(days=(days-1)) # Fin de la période de prédiction

fig = plot_plotly(models[variable], 
                  forecast, 
                  trend=True, 
                  changepoints=True)

fig.update_layout(
    title="Prévisions de la variable '{}' du 01 Jan 2022 au {}".format(variable, end_date.strftime('%d %b %Y')),
    title_font=dict(size=24, color="#ad7d67"),
    title_xanchor='center',
    title_x=0.5,
    width=1050, height=700, 
    font=dict(size=12),
    xaxis=dict(
        title="Date",
        type='date',
        range=['2019-01-01', '2022-12-31'],
        tickformat="%d-%m-%Y", 
        tickangle=45, 
        dtick="M1", 
        tickfont=dict(size=10), 
        ticklabelmode="period",
        ticklabelposition="outside top"),
    yaxis_title=variable,
    colorway=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        title=None,
        font=dict(size=16, color="#5e5c5e")
    ))
fig.update_xaxes(showgrid=True)
st.plotly_chart(fig)

# Ajout d'un texte dans un encadré sur lequel on peut cliquer
expander = st.expander("Plus d'informations sur ce graphique")
expander.write("""
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
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