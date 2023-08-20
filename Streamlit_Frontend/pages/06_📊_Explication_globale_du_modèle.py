import streamlit as st
import pandas as pd
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
        page_title='Interprétation Globale de la prédiction',
        page_icon = "📊",
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
# Chargement du modèle
model_path = os.path.join('models', 'model_Catboost.pkl')
model = pickle.load(open(model_path, "rb"))
target_names = list(model.classes_)
# Rappel des numéros des classes et de leur libellé
# 0:"blessé_hospitalisé"
# 1:"blessé_léger"
# 2:"indemne"
# 3:"tué"

# Chargement des valeurs Shap
shap_path = os.path.join('models', 'shap_values_catboost.pkl')
shap_values_all = pickle.load(open(shap_path, "rb"))

@st.cache_data #mise en cache de la fonction pour exécution unique
def load_data():
    data_path = os.path.join('data', 'X_test_catboost.csv')
    df = pd.read_csv(data_path)
    return df

X_test = load_data()
var_cat_nominale=X_test.columns.tolist()

# Titre 1
st.markdown("""
            <h1>
            9. Quelles sont les variables globalement les plus importantes pour comprendre la prédiction ?
            </h1>
            """, 
            unsafe_allow_html=True)
st.write("")

st.write("L’importance des variables est calculée en moyennant la valeur absolue des valeurs de Shap. \
        Les caractéristiques sont classées de l'effet le plus élevé au plus faible sur la prédiction. \
        Le calcul prend en compte la valeur SHAP absolue, donc peu importe si la fonctionnalité affecte \
        la prédiction de manière positive ou négative.")

st.write("Pour résumer, les valeurs de Shapley calculent l’importance d’une variable en comparant ce qu’un modèle prédit \
        avec et sans cette variable. Cependant, étant donné que l’ordre dans lequel un modèle voit les variables peut affecter \
        ses prédictions, cela se fait dans tous les ordres possibles, afin que les fonctionnalités soient comparées équitablement. \
        Cette approche est inspirée de la théorie des jeux.")

st.write("*__Le diagramme d'importance des variables__* répertorie les variables les plus significatives par ordre décroissant.\
        Les *__variables en haut__* contribuent davantage au modèle que celles en bas et ont donc un *__pouvoir prédictif élevé__*.")
st.write("Puisque nous traitons une tâche de classification multi-classes, le tracé récapitulatif classe les variables \
         en fonction de leur contribution globale à toutes les classes et code en couleur l'ampleur de chaque classe.")

fig = plt.figure()
shap.summary_plot(
                shap_values_all,
                X_test.values,
                plot_type="bar",
                class_names=target_names,
                feature_names=var_cat_nominale,
                plot_size=(16, 12),
                max_display=33,
                show=False)
plt.title(
    "Importance des features dans la construction du modèle CatBoost multi-classes",
    fontsize=20,
    fontstyle='italic')
plt.tight_layout()
st.pyplot(fig)

st.write("***Ainsi, la catégorie du véhicule accidenté est la variable qui contribue le plus à la performance \
         du modèle CatBoost multi-classes, et surtout pour la prédiction des personnes 'indemnes'.***")

# Titre
st.markdown("""
            <h1>
            10. Quel est l'Impact de chaque caractéristique sur la prédiction de chaque classe ?
            </h1>
            """, 
            unsafe_allow_html=True)
st.write("")

st.write("Les diagrammes des valeurs SHAP ci-dessous indique également comment chaque caractéristique impacte la prédiction. \
        Les valeurs de Shap sont représentées pour chaque variable dans leur ordre d’importance. \
        Chaque point représente la valeur Shap d'une instance (un accidenté de la route).")
st.write("")

fig = plt.figure()
ax0 = fig.add_subplot(221)
shap.summary_plot(shap_values_all[3], 
                  features=X_test,
                  feature_names=var_cat_nominale,
                  class_names= target_names,
                  #cmap='PiYG_r',
                  plot_type="dot",
                  max_display=15,
                  show = False)
plt.title(f"Interprétation Globale de la classe {target_names[3]}\n", 
          fontsize=20, fontstyle='italic', fontweight='bold')

ax1 = fig.add_subplot(222)
shap.summary_plot(shap_values_all[0], 
                  features=X_test,
                  feature_names=var_cat_nominale,
                  class_names= target_names,
                  #cmap='PiYG_r',
                  plot_type="dot",
                  max_display=15,
                  show = False)
plt.title(f"Interprétation Globale de la classe {target_names[0]}\n", 
          fontsize=20, fontstyle='italic', fontweight='bold')

ax2 = fig.add_subplot(223)
shap.summary_plot(shap_values_all[1], 
                  features=X_test,
                  feature_names=var_cat_nominale,
                  class_names= target_names,
                  #cmap='PiYG_r',
                  plot_type="dot",
                  max_display=15,
                  show = False)
plt.title(f"\nInterprétation Globale de la classe {target_names[1]}\n", 
          fontsize=20, fontstyle='italic', fontweight='bold')

ax3 = fig.add_subplot(224)
shap.summary_plot(shap_values_all[2], 
                  features=X_test,
                  feature_names=var_cat_nominale,
                  class_names= target_names,
                  #cmap='PiYG_r',
                  plot_type="dot",
                  max_display=15,
                  show = False)
plt.title(f"\nInterprétation Globale de la classe {target_names[2]}\n", 
          fontsize=20, fontstyle='italic', fontweight='bold')
plt.gcf().set_size_inches(24,16)
plt.tight_layout() 
st.pyplot(plt.gcf())

plt.clf()  # Nettoyez la figure courante

st.write("Que signifie la couleur gris foncé ? \
         CatBoost gère les variables catégorielles en interne, pour cette raison les caractéristiques catégorielles \
         ne sont pas incluses dans le modèle SHAP en tant que versions codées numériques.")
st.write("Nous verrons leur importance en détail avec le tracé de dépendance des valeurs Shap dans la section suivante.")


# Titre 
st.markdown("""
            <h1>
            11. Interprétation des prédictions par classe
            </h1>
            """, 
            unsafe_allow_html=True)
st.write("Nous pouvons obtenir un aperçu plus approfondi de l'effet de chaque fonctionnalité \
          pour chaque classe sur l'ensemble de données avec un graphique de dépendance.")

################################################################################
# Création et affichage du sélecteur des variables et des graphs de dépendance #
################################################################################
var_cat_nominale = sorted(var_cat_nominale)
col1, col2, = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
with col1:
    ID_var = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                            (var_cat_nominale), index=29)
    st.write("Vous avez sélectionné la variable :", ID_var)

st.write(f"Les catégories de la variable **{ID_var}** sont sur l'axe des abscisses.")
st.write("Pour chaque classe, les valeurs de SHAP les plus élevées indiquent les catégories de la variable d'entrée \
         qui ont le plus grand impact sur les prédictions du modèle.")

fig = plt.figure()
ax0 = fig.add_subplot(222)
shap.dependence_plot(ID_var, 
                     shap_values_all[0], 
                     X_test, 
                     interaction_index=None, 
                     color='#ad7e3f',
                     x_jitter=0.5,
                     alpha=0.5,
                     ax=ax0, 
                     show = False)
plt.title(f"Graphique de dépendance\nVariable : '{ID_var}'\nClasse : '{target_names[0]}'",
          fontsize=20,
          fontstyle='italic')

ax1 = fig.add_subplot(223)
shap.dependence_plot(ID_var, 
                     shap_values_all[1], 
                     X_test, 
                     interaction_index=None, 
                     color='#cbe6e2',
                     x_jitter=0.5,
                     alpha=0.5,
                     ax=ax1, show = False)
plt.title(f"Graphique de dépendance\nVariable : '{ID_var}'\nClasse : '{target_names[1]}'",
          fontsize=20,
          fontstyle='italic')

ax2 = fig.add_subplot(224)
shap.dependence_plot(ID_var, 
                     shap_values_all[2], 
                     X_test, 
                     interaction_index=None, 
                     color='#135e58',
                     x_jitter=0.5,
                     alpha=0.5,
                     ax=ax2, show = False)
plt.title(f"Graphique de dépendance\nVariable : '{ID_var}'\nClasse : '{target_names[2]}'",
          fontsize=20,
          fontstyle='italic')

ax3 = fig.add_subplot(221)
shap.dependence_plot(ID_var, 
                     shap_values_all[3], 
                     X_test, 
                     interaction_index=None, 
                     color='#6b4516',
                     x_jitter=0.5,
                     alpha=0.5,
                     ax=ax3, show = False)
plt.title(f"Graphique de dépendance\nVariable : '{ID_var}'\nClasse : '{target_names[3]}'",
          fontsize=20,
          fontstyle='italic')

plt.gcf().set_size_inches(24,16)
plt.tight_layout()
st.pyplot(plt.gcf())

plt.clf()  # Nettoyez la figure courante


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





















