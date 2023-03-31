import streamlit as st
from PIL import Image
import os

st.set_page_config(
        page_title='Conclusion et Perspective',
        page_icon = "🎓",
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

image_path = os.path.join('assets', 'Bannière La gravité des accidents de la route.png')
banniere = Image.open(image_path)
st.image(banniere, use_column_width="always")

st.markdown("""
            <h1>
            12. Prédire la gravité des accidents de la route ?
            </h1>
            """, 
            unsafe_allow_html=True)

st.markdown("""
            La gravité des accidents de la route en France est un sujet de préoccupation pour les autorités, 
            les citoyens et les associations de prévention routière. Les accidents de la route peuvent entraîner 
            des conséquences graves, notamment des décès, des blessures corporelles et des dégâts matériels.
            
            D'après l'[Observatoire national interministériel de la sécurité routière](https://www.onisr.securite-routiere.gouv.fr/), 
            plusieurs facteurs contribuent à la gravité des accidents de la route en France, 
            notamment **la vitesse excessive, l'alcool, la drogue, la fatigue, le non-respect des règles de circulation 
            et le manque d'entretien des véhicules**. Les comportements irresponsables des usagers de la route augmentent 
            le risque d'accidents graves.
            
            Et c'est précisément pourquoi, puisque nous ne disposons pas de ces facteurs explicatifs dans notre jeu de données, 
            que notre meilleur modèle de prédiction n'a pas, malgré tout, d'excellents résultats puisque son Accuracy *(proportion 
            de prédictions correctes parmi toutes les prédictions effectuées)* n'est que de 58% ! 
            """)
st.write("")

st.write("***Pour information : Publication le 31 janvier des chiffres quasi-définitifs du bilan de la sécurité routière en 2022.***")
st.write("[Bilan 2022 de la sécurité routière](https://www.onisr.securite-routiere.gouv.fr/etat-de-l-insecurite-routiere/bilans-annuels-de-la-securite-routiere/bilan-2022-de-la-securite-routiere)")
st.write("")
image_path = os.path.join('assets', 'Bilan 2022 securite routiere.png')
bilan_2022 = Image.open(image_path)
st.image(bilan_2022, use_column_width="always")

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