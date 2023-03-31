import streamlit as st
import pandas as pd
import streamlit_pandas as sp

############################
# Configuration de la page #
############################
st.set_page_config(
        page_title='Parcourez les données',
        page_icon = "🗃️",
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

########################
# Lecture des fichiers #
########################
@st.cache_data #mise en cache de la fonction pour exécution unique
def load_data():
    df = pd.read_csv("data\df_accidents.csv")
    return df

st.markdown("""
            <h1>
            3. Parcourez les données
            </h1>
            """, 
            unsafe_allow_html=True)
st.write("")
st.markdown("""
            Parcourez la base de données à l'aide des filtres situés dans la barre latérale à gauche.<br>
            Et téléchargez le résultat en fichier csv en cliquant sur le bouton "Download data as CSV".
            """,
            unsafe_allow_html=True)

create_data = {"gravite_accident": "multiselect",
               "catégorie_usager": "multiselect",
               "sexe": "multiselect",
               "age": "multiselect",
               "nom_dep": "multiselect",
               "region":"multiselect",
               "localisation":"multiselect",
               "tranche_heure":"multiselect",
               "weekday":"multiselect",
               "mois":"multiselect",
               "annee":"multiselect",
               "catégorie_véhicule":"multiselect",
               "obstacle_mobile_heurté":"multiselect",
               "luminosite": "multiselect",
               "condition_atmospheriques": "multiselect",
               "place_occupée":"multiselect",
               "trajet":"multiselect",
               "intersection":"multiselect",
               "type_collision":"multiselect",
               "categorie_route":"multiselect",
               "sens_circulation":"multiselect",
               "nbr_voies":"multiselect",
               "declivite_route":"multiselect",
               "trace_plan":"multiselect",
               "etat_surface":"multiselect",
               "infrastructure":"multiselect",
               "situation_accident":"multiselect",
               "vitesse_maximale_autorisee":"multiselect",
               "repère_sens_circulation":"multiselect",
               "choc_initial":"multiselect",
               "manoeuvre_principale":"multiselect"}

df_accidents = load_data()
df_accidents = df_accidents.reindex(columns=[
    'gravite_accident', 'catégorie_usager', 'sexe', 'age', 'nom_dep', 'region', 'localisation',
    'tranche_heure', 'weekday', 'mois', 'annee', 'catégorie_véhicule', 'obstacle_mobile_heurté',
    'luminosite', 'condition_atmospheriques', 'place_occupée', 'trajet', 'intersection',
    'type_collision', 'categorie_route', 'sens_circulation', 'nbr_voies', 'declivite_route',
    'trace_plan', 'etat_surface', 'infrastructure', 'situation_accident', 'vitesse_maximale_autorisee',
    'repère_sens_circulation', 'choc_initial', 'manoeuvre_principale', "secu1", "secu2", "secu3", 
    "nb_equipement_securite", "departement", "num_commune", "date", "heure", "latitude", "longitude"])

all_widgets = sp.create_widgets(df_accidents, create_data,
                                ignore_columns=["secu1", "secu2", "secu3", "nb_equipement_securite",
                                                "departement", "num_commune", "date", "heure",
                                                "latitude", "longitude"])
res = sp.filter_df(df_accidents, all_widgets)
st.markdown("""
            <h2>
            Les données originales
            </h2>
            """, 
            unsafe_allow_html=True)
st.dataframe(df_accidents)

st.markdown("""
            <h2>
            Les données filtrées
            </h2>
            """, 
            unsafe_allow_html=True)
st.dataframe(res)

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(res)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="df_accidents_filtrees.csv",
    mime="text/csv",
    key='browser-data'
)
