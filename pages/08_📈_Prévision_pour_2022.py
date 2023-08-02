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

########################################
# Lecture des fichiers et des modèles  #
########################################

# Chargement des données réelles
@st.cache_data
def load_data():
    data_path = os.path.join('data', 'nbr_acc_jour.csv')
    df = pd.read_csv(data_path)
    df.rename(columns={'date': 'ds'}, inplace=True)
    return df

data = load_data()

# Fonction pour charger les modèles à partir des fichiers JSON
@st.cache_resource
def load_model(file_name):
    models_dir = "models"
    file_path = os.path.join(models_dir, file_name)
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
    regressors_dir = "data"
    file_name = f"{model_name}_regressors.csv"
    file_path = os.path.join(regressors_dir, file_name)
    regressors = pd.read_csv(file_path)
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




##################################################################################
# Sélection de la Variable à pérdire et de la période de temps par l'utilisateur #
##################################################################################

variable = st.selectbox('Choisissez la variable à prédire', list(models.keys()))
days = st.slider('Choisissez le nombre de jours à prédire', 1, 365)

# Chargement des régresseurs
regressors = load_regressors(variable)

future = models[variable].make_future_dataframe(periods=days, freq='D', include_history=True)
forecast = models[variable].predict(future)

start_date = datetime(2022, 1, 1)
end_date = start_date + timedelta(days=days)

fig = plot_plotly(models[variable], forecast, trend=True, changepoints=True)

fig.update_layout(
    title="Prévisions de la variable '{}' du 01 Jan 2022 au {}".format(variable, end_date.strftime('%d %b %Y')),
    xaxis_title="Date",
    yaxis_title=variable,
    colorway=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
)

st.plotly_chart(fig)

# Ajout d'un texte dans un encadré sur lequel on peut cliquer
expander = st.beta_expander("Plus d'informations sur ce graphique")
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