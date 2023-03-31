import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap
import pickle
from PIL import Image
import os

############################
# Configuration de la page #
############################
st.set_page_config(
        page_title='Explication de la Prédiction',
        page_icon = "🥇",
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
            .css-1offfwp li {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            </style> """, 
            unsafe_allow_html=True)

st.markdown("""
            <h1>
            8. "Pourquoi le modèle a-t-il prédit cette valeur ?"
            </h1>
            """, 
            unsafe_allow_html=True)

# Chargement du modèle
model_path = os.path.join('models', 'model_Catboost.pkl')
model = pickle.load(open(model_path, "rb"))

target_names = list(model.classes_)
# Rappel des numéros des classes et de leur libellé
# 0:"blessé_hospitalisé"
# 1:"blessé_léger"
# 2:"indemne"
# 3:"tué"

# Récupérez la prédiction et les données d'entrée à partir de st.session_state
prediction = st.session_state.prediction
probabilities = st.session_state.probabilities
input_data = st.session_state.input_data
st.session_state.prob_df = pd.DataFrame(probabilities, columns=model.classes_)
st.session_state.prob_df = st.session_state.prob_df[['indemne', 'blessé_léger', 'blessé_hospitalisé', 'tué']]

prediction_value = prediction[0]
index_prediction = target_names.index(prediction_value)
#st.write(index_prediction)

st.markdown("""
            ***Pour rappel, les modalités des caractéristiques de l'accident que vous avez sélectionnées ont comme résultats :***
            """)
st.write(f"La prédiction de la gravité de l'accident est : **{st.session_state.prediction_text}**")
st.write(f"**La probabilité que l'usager de la route appartienne à la classe {st.session_state.prediction_text} \
         est de {probabilities.max():.1%}**.")
st.write("Et, les probabilités pour chaque classe sont (en %) :")
st.write((st.session_state.prob_df * 100).round(1))

# Vérifiez si la prédiction et les données d'entrée sont disponibles
if prediction is not None and input_data is not None:
    # Créez un explainer SHAP pour le modèle CatBoost
    explainer = shap.TreeExplainer(model)

    # Calculez les valeurs SHAP pour les données d'entrée
    shap_values = explainer.shap_values(input_data)
    
else:
    st.markdown("""
                Aucune prédiction n'est disponible pour l'explication. 
                Veuillez d'abord effectuer une prédiction dans la page 04_💻_Modélisation.
                """)

st.markdown("""
            Pour comprendre et interpréter visuellement les résutats de la prédiction, il existe trois graphiques alternatifs : 
            `Force plot`, `Decision plot` et `Waterfall plot`.
            Ces trois représentations sont redondantes car elles représentent l’information de manière très similaire. 
            En même temps, certains éléments de ces graphiques sont complémentaires. 
            En mettant côte à côte les trois, on comprend d’une manière plus intuitive le résultat.
            """)
st.markdown("""
            Le `Force plot` est bon pour voir où se place le “output value” par rapport à la “base value”. 
            Nous observons également quelles variables ont un impact positif (rouge) ou négatif (bleu) sur la prédiction 
            et l’amplitude de cet impact.
            """) 

st_shap(shap.force_plot(explainer.expected_value[index_prediction], 
                        shap_values[index_prediction], 
                        input_data,
                        figsize=(20, 8),
                        ordering_keys=True,
                        text_rotation=0))

st.markdown("""
            Le `Waterfall plot` permet aussi de voir l’amplitude et la nature d’impact d’une variable avec sa quantification. 
            Il permet de voir également l’ordre d’importance des variables et les valeurs prises par chacune des variables 
            pour l’instance étudiée (l'accident que vous avez sélectionné), c'est-à-dire qu'il affiche les contributions marginales
            de chaque variable pour l'observation spécifique sélectionnée. 
            Les features qui ont une contribution positive sont affichées en rouge, tandis que les features qui ont une contribution négative
            sont affichées en bleu. Chaque barre représente la contribution marginale de chaque feature à la prédiction finale du modèle, 
            et la barre totale représente la prédiction moyenne du modèle pour l'ensemble des observations. 
            Le graphique permet de visualiser l'effet cumulatif des features sur la prédiction finale du modèle.
            """)

shap.waterfall_plot(shap.Explanation(values=shap_values[index_prediction][0], 
                                     base_values=explainer.expected_value[index_prediction], 
                                     data=input_data.iloc[0],  
                                     feature_names=input_data.columns.tolist()),
                    max_display=15,
                    show=False)
# Modifiez la taille du graphique
plt.gcf().set_size_inches(14, 10)  # Modifiez les dimensions (largeur, hauteur) comme vous le souhaitez
st.pyplot(plt.gcf())

plt.clf()  # Nettoyez la figure courante

st.markdown("""
            Le graphique ci-dessous appelé `decision_plot` est une autre manière de comprendre la prédiction.\
            Comme pour le graphique précédent, il met en évidence l’amplitude et la nature de l’impact de chaque variable \
            avec sa quantification ainsi que leur ordre d’importance. Mais surtout il permet d'observer \
            “la trajectoire” prise par la prédiction de l'usager de la route pour chacune des valeurs des variables explicatives.
            """)

shap.multioutput_decision_plot(explainer.expected_value, 
                               shap_values,
                               row_index=0,
                               feature_order='importance',
                               feature_names=input_data.columns.to_list(),
                               legend_labels=target_names,
                               feature_display_range=slice(None, -34, -1),
                               highlight=[np.argmax(probabilities[0])],
                               plot_color="BrBG_r",
                               legend_location='best',
                               show=False)
plt.title(f"Diagramme de décision\n", 
          fontsize=16, fontstyle='italic', fontweight='bold')
plt.gcf().set_size_inches(10,10)
st.pyplot(plt.gcf())

plt.clf()  # Nettoyez la figure courante


st.markdown("""
            Nous venons de comprendre et interpréter visuellement les résutats de la prédiction pour un usager de la route dont
            nous avons défini les caractéristiques de l'accident.
            
            Maintenant, nous devons répondre aux questions suivantes :
            1. Quelles sont les variables globalement les plus importantes pour comprendre la prédiction ?
            2. Quel est l'Impact de chaque caractéristique sur la prédiction de chaque classe ?
            
            *Cliquez sur "Explication globale du modèle" dans la barre latérale gauche.*
            """
            )


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