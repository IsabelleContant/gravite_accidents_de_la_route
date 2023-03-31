import streamlit as st
from PIL import Image

st.set_page_config(
        page_title='La gravité des accidents de la route en France',
        page_icon = "🏡",
        layout="wide"
    )
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

banniere = Image.open('assets\Bannière La gravité des accidents de la route.png')
st.image(banniere, use_column_width="always")

# Titre et sous_titre du projet
st.markdown("""
            <p style="color:Gray;text-align:center;font-size:2em;font-style:italic;font-weight:700;font-family:'Roboto Condensed';margin:0px;">
            Projet Fil Rouge - DataScientist - Promotion Janvier 2023</p>
            """, 
            unsafe_allow_html=True)
st.markdown("""
            <p style="color:Gray;text-align:center;font-size:1.5em;font-style:italic;font-family:'Roboto Condensed';margin:0px;">
            <strong>Auteurs :</strong> Anicet Arthur KOUASSI - Isabelle Contant - Omar DIANKHA - Idelphonse GBOHOUNME</p>
            """, 
            unsafe_allow_html=True)
st.write("")
st.markdown("""*Repository Github du projet : [cliquez ici](https://github.com/IsabelleContant/gravite_accidents_de_la_route)*""")
st.write("")

# Description et Objectif du projet
# Titre 1
st.write("")
st.markdown("""
            <h1>
            1. Contexte et Objectifs
            </h1>
            """, 
            unsafe_allow_html=True)
st.write("")

st.markdown("""
            L' Observatoire national interministériel de la sécurité routière met à disposition chaque année depuis 2005, 
            des [bases de données des accidents corporels de la circulation routière](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/).
            Pour chaque accident corporel (soit un accident survenu sur une voie ouverte à la
            circulation publique, impliquant au moins un véhicule et ayant fait au moins une victime
            ayant nécessité des soins), des saisies d’information décrivant l’accident sont effectuées
            par l’unité des forces de l’ordre (police, gendarmerie, etc.) qui est intervenue sur le lieu de
            l’accident.
            
            Ces saisies sont rassemblées dans une fiche intitulée bulletin d’analyse des accidents
            corporels. L’ensemble de ces fiches constitue le fichier national des accidents corporels
            de la circulation dit « Fichier BAAC » administré par l’Observatoire national interministériel
            de la sécurité routière "ONISR".
            
            ***Les bases de données, extraites du fichier BAAC, répertorient l'intégralité des accidents
            corporels de la circulation, intervenus durant une année précise en France métropolitaine,
            dans les départements d’Outre-mer (Guadeloupe, Guyane, Martinique, La Réunion et
            Mayotte depuis 2012) et dans les autres territoires d’outre-mer (Saint-Pierre-et-Miquelon,
            Saint-Barthélemy, Saint-Martin, Wallis-et-Futuna, Polynésie française et Nouvelle-
            Calédonie; disponible qu’à partir de 2019 dans l’open data) avec une description
            simplifiée.***
            
            Cela comprend des informations de localisation de l’accident, telles que renseignées ainsi
            que des informations concernant les caractéristiques de l’accident et son lieu, les
            véhicules impliqués et leurs victimes.
            """)

description_bdd = Image.open('assets\Description BDD Accidents Routiers.jpg')
st.image(description_bdd, use_column_width="always")
st.write("")

st.markdown("""
            <p style="color:#946f56;font-size:2em;font-style:italic;font-weight:700;font-family:'Roboto Condensed';margin:0px;">
            L’ objectif de ce projet est de prédire la gravité de l'ensemble des accidents routiers en France intervenus entre 2019 et 2021.</p>
            """, 
            unsafe_allow_html=True)

st.write("")
st.markdown("""
            <h1>
            2. Les Données
            </h1>
            """, 
            unsafe_allow_html=True)

st.markdown("""
            Bien que l’on puisse penser qu’il suffit d’un grand nombre de données pour avoir un algorithme performant, 
            les données dont nous disposons sont souvent non adaptées. 
            Il faut donc les comprendre et les traiter préalablement pour pouvoir ensuite les utiliser : 
            c’est l’étape d'exploration et de visualisation des données.
            
            En effet, des erreurs d’acquisition liées à des fautes humaines ou techniques peuvent corrompre 
            notre dataset et biaiser l’entraînement. 
            Parmi ces erreurs, nous pouvons citer des informations incomplètes, des valeurs manquantes ou erronées 
            ou encore des bruits parasites liés à l’acquisition de la donnée.
            
            Il est donc souvent indispensable d’établir une stratégie de pré-traitement des données à partir des données brutes 
            pour arriver à des données exploitables qui nous donneront un modèle plus performant.
            
            La particularité de notre jeu de données est que nous avons 4 fichiers par année avec les mêmes variables.
            
            Afin d'effectuer les mêmes traitements pour les 3 années, nous avons décidé d'analyser minutieusement chaque variable 
            de chaque rubrique de l'année 2021, puis d'appliquer exactement les mêmes transformations pour les rubriques 
            des années 2019 et 2020.
            
            En utilisant cette méthodologie de travail, nous nous sommes assurés d'obtenir un jeu de données par année 
            avec les mêmes variables et les mêmes transformations.
"""
)
traitements_data = Image.open('assets\Traitements des données.jpg')
st.image(traitements_data, use_column_width="always")

st.markdown("""Au total, notre base de données comprend 363 336 personnes accidentées de la route (numérotées de 0 à 363 335)
            décrites par 41 variables.
            """)

# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    logo = Image.open('assets\logo-datascientest.png')
    st.sidebar.image(logo, use_column_width="always")
with col3:
    st.sidebar.write("")

