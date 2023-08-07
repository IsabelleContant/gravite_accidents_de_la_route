import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image
import plotly.graph_objs as go
import os
from prophet.plot import plot_plotly
from prophet.serialize import model_from_json
import json
import calendar

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
            h4 {font-family:'Roboto Condensed';
                color:#ad7d67;
                font-size:26px;
                text-align:center;}
            p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .css-16idsys p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem; margin: 0px;}
            .css-5rimss li {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .css-184tjsw p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .css-1offfwp li {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .st-ae {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem; background-color: #f8f9fb;}
            .st-cn {background-color: #f8f9fb; }
            .css-1wivap2 {font-family:'Roboto Condensed';}
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
<p style="color: #ad7d67; font-weight:700; font-style:italic; margin: 0px;">
Attention ! L'objectif principal du projet fil rouge MLOps n'est pas de construire des modèles de prévision : il s'agit de mettre en place une chaîne de traitement
complète de Machine Learning, de la récupération des données à la mise en production du modèle, c'est-à-dire d'implémenter tout le cycle de vie 
du modèle en production.</p>""", unsafe_allow_html=True)
st.write("")

##################################################################################
# Sélection de la Variable à prédire et de la période de temps par l'utilisateur #
##################################################################################
st.write("")

st.markdown("""
Rentrons dans le vif du sujet !

Pour obtenir des prévisions quotidiennes pour l'année 2022, vous devez d'abord sélectionner ce que vous souhaitez prédire parmi :
- le nombre total d'accidentés de la route
- le nombre de victimes indemnes
- le nombre de victimes légèrement blessées
- le nombre de victimes blessées hospitalisées
- et le nombre de victimes tuées

Vous devez aussi choisir la période de temps, de 1 à 365 jours, dont vous voulez obtenir la prédiction.
""", unsafe_allow_html=True)
st.write("")
st.write("")

# Définissez les largeurs des colonnes principales et des colonnes vides
main_col_width = 6
empty_col_width = 0.5
# Divisez l'espace en 3 colonnes avec des largeurs personnalisées
cols = st.columns([main_col_width, empty_col_width, main_col_width])
# Attribuez les colonnes principales aux indices 0 et 3
col1, col2, col3 = cols[0], cols[1], cols[2]
with col1:
    variable = st.selectbox('***Choisissez la variable dont vous souhaitez connaître les prédictions :***', list(models.keys()))
with col3:
    days = st.slider('***Choisissez le nombre de jours que vous souhaitez prédire :***', 1, 365, 181)
st.write("")
st.write("")
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

######################################################################
# les totaux  calculés sur la période sélectionnée et les évolutions #
######################################################################
# Dates pour 2022
start_date_2022  = datetime(2022, 1, 1) # Début de la période de prédiction
end_date_2022 = start_date_2022 + timedelta(days=(days)) # Fin de la période de prédiction
end_date_2022_titre = end_date_2022 - timedelta(days=1) # Fin de la période de prédiction pour le titre du graphique

# Dates pour 2021
start_date_2021 = datetime(2021, 1, 1) # Début de la période 
end_date_2021 = start_date_2021 + timedelta(days=(days)) # Fin de la période 
end_date_2021_titre = end_date_2021 - timedelta(days=1) 

# Dates pour 2020
start_date_2020 = datetime(2020, 1, 1) # Début de la période 
end_date_2020 = start_date_2020 + timedelta(days=(days)) # Fin de la période 
end_date_2020_titre = end_date_2020 - timedelta(days=1)

# Dates pour 2019
start_date_2019 = datetime(2019, 1, 1) # Début de la période 
end_date_2019 = start_date_2019 + timedelta(days=(days)) # Fin de la période 
end_date_2019_titre = end_date_2019 - timedelta(days=1)

# Filtrer les données pour chaque année
data_2021 = data[(data['ds'] >= start_date_2021) & (data['ds'] <= end_date_2021)]
data_2020 = data[(data['ds'] >= start_date_2020) & (data['ds'] <= end_date_2020)]
data_2019 = data[(data['ds'] >= start_date_2019) & (data['ds'] <= end_date_2019)]
forecast_2022 = forecast[(forecast['ds'] >= start_date_2022) & (forecast['ds'] <= end_date_2022)]

# Calculer les évolutions
evol_2020 = (data_2020[variable].sum() - data_2019[variable].sum()) / data_2019[variable].sum()
evol_2021 = (data_2021[variable].sum() - data_2020[variable].sum()) / data_2020[variable].sum()
evol_2022 = (forecast_2022['yhat'].sum() - data_2021[variable].sum()) / data_2021[variable].sum()
evol_2022_vs_2019 = (forecast_2022['yhat'].sum() - data_2019[variable].sum()) / data_2019[variable].sum()


#################
# Graphique n°1 #
#################

# Titre des métriques
st.markdown(f"#### Évolution du nombre de '{variable}' pour la totalité de la période sélectionnée de {days} jours")

left, middle, right, far_right = st.columns(4)

with left:
    st.metric(f"Du {start_date_2019.strftime('%d-%m-%Y')} au {end_date_2019_titre.strftime('%d-%m-%Y')}", 
              f"{int(data_2019[variable].sum()):,}",
              f"Non disponible")
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=data_2019, x="ds", y=variable, color="#ad7d67", ax=ax1)
    ax1.set_xlabel("Date")
    sns.despine()
    st.pyplot(fig1)
with middle:
    st.metric(f"Du {start_date_2020.strftime('%d-%m-%Y')} au {end_date_2020_titre.strftime('%d-%m-%Y')}",
              f"{int(data_2020[variable].sum()):,d}",
              f"{evol_2020:+.1%}")
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=data_2020, x="ds", y=variable, color="#ad7d67", ax=ax2)
    ax2.set_xlabel("Date")
    sns.despine()
    st.pyplot(fig2)
with right:
    st.metric(f"Du {start_date_2021.strftime('%d-%m-%Y')} au {end_date_2021_titre.strftime('%d-%m-%Y')}", 
              f"{int(data_2021[variable].sum()):,d}",
              f"{evol_2021:+.1%}")
    fig3, ax3 = plt.subplots()
    sns.lineplot(data=data_2021, x="ds", y=variable, color="#ad7d67", ax=ax3)
    ax3.set_xlabel("Date")
    sns.despine()
    st.pyplot(fig3)
with far_right:
    st.metric(f"Du {start_date_2022.strftime('%d-%m-%Y')} au {end_date_2022_titre.strftime('%d-%m-%Y')}", 
              f"{int(forecast_2022['yhat'].sum()):,d}",
              f"{evol_2022:+.1%} ({evol_2022_vs_2019:+.1%} vs. 2019)")
    fig4, ax4 = plt.subplots()
    sns.lineplot(data=forecast_2022, x="ds", y='yhat', color="#9ebeb8", ax=ax4)
    ax4.set_xlabel("Date")
    sns.despine()
    st.pyplot(fig4)

st.write("")
#################
# Graphique n°2 #
#################

fig5 = plot_plotly(models[variable], 
                  forecast, 
                  trend=True, 
                  changepoints=True)

fig5.update_layout(
    title=f"Représentation globale de la série temporelle '{variable}'<br>du 01 Jan 2019 au {end_date_2022_titre.strftime('%d %b %Y')}",
    title_font=dict(size=26, color="#ad7d67"),
    title_xanchor='center',
    title_x=0.5,
    width=1050, height=700, 
    font=dict(size=12),
    xaxis=dict(
        title="Date",
        type='date',
        range=["2019-01-01", end_date_2022.strftime('%Y-%m-%d')],
        tickformat="%d-%m-%Y", 
        tickangle=-45, 
        dtick="M1", 
        tickfont=dict(size=10), 
        ticklabelmode="period",
        ticklabelposition="outside top"),
    yaxis_title=variable,
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
fig5.update_xaxes(showgrid=True)
st.plotly_chart(fig5)

st.markdown("""
Ce graphique montre les données historiques (points noirs) et les les valeurs prédites (ligne bleue) pour la période sélectionnée.
Les valeurs prédites sont donc calculées pour l'ensemble de données complet lors du calcul prévisionnel.
La bande ombrée entourant la ligne représente un intervalle de confiance à 95%.

La ligne rouge montre la tendance, qui est la composante de tendance du modèle. La tendance est calculée à partir de la moyenne mobile sur
les changements de tendance détectés dans les données historiques. Elle nous donne une vision synthétique du signal et aide à visualiser 
les évolutions globales.
Les lignes verticales rouges sont les points de changement de tendance détectés par le modèle. Ces points de changement de tendance sont
utilisés pour calculer la tendance, mais ne sont pas utilisés pour calculer les prévisions.

**Ces prévisions semblent saisonnières**, mais il est difficile de distinguer les différentes composantes périodiques sur ce premier tracé. 
Vérifions une autre visualisation pour **comprendre comment ces modèles saisonniers affectent la sortie du modèle**:
""", unsafe_allow_html=True)
st.write("")
st.write("")

#################
# Graphique n°3 #
#################
def plot_seasonality(forecast, component, fillcolor, linecolor):
    # Créer un objet de figure Plotly
    fig6 = go.Figure()

    # Extraire le composant saisonnier
    component_df = forecast[[component]].dropna()
    values = component_df[component]

    if component == 'weekly':
        # Créer une variable 'day of week'
        component_df['day_of_week'] = forecast['ds'].dt.dayofweek
        # Grouper par 'day of week' et sommer les valeurs
        values = component_df.groupby('day_of_week')[component].mean()
        x_data = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    elif component == 'yearly':
        # Créer une variable 'day of year'
        component_df['day_of_year'] = forecast['ds'].dt.dayofyear
        # Grouper par 'day of year' et sommer les valeurs
        values = component_df.groupby('day_of_year')[component].mean()
        x_data = [i for i in range(1, 366)]  # Jours de l'année

    # Ajouter le composant saisonnier
    fig6.add_trace(go.Scatter(
        x=x_data,
        y=values,
        mode='lines',
        fill='tozeroy',  # Remplir jusqu'à l'axe des y
        fillcolor=fillcolor,  # Couleur de remplissage en hexadécimal
        line=dict(color=linecolor),  # Couleur de ligne en hexadécimal
        name=component,
    ))

    # Ajouter un titre
    fig6.update_layout(
        title=f"<b>{component} Seasonality</b>",
        title_font=dict(size=26, color="#ad7d67"),
        title_xanchor='center',
        title_x=0.5,
        yaxis=dict(
            title=variable,
            tickformat=''),
        width=900, height=350,
        font=dict(size=12),
        showlegend=False,  # Supprimer la légende
    )

    # Ajuster les labels sur l'axe des x pour le composant 'yearly'
    if component == 'yearly':
        # Premier jour de chaque mois
        tickvals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        fig6.update_xaxes(
            tickmode='array',
            tickvals=tickvals,
            tickangle=-45,
            ticktext=[calendar.month_name[i] for i in range(1, 13)]  # Noms complets des mois
        )

    # Afficher le graphique
    st.plotly_chart(fig6, theme=None)



# Centrage des graphiques dans la page
col1, col2, col3 = st.columns([0.5,8,0.5])
with col1:
    st.write("")
with col2:
    # Pour chaque composant saisonnier
    for component, fillcolor, linecolor in [('weekly', '#f779ae', '#f6408d'), ('yearly', '#abdfe1', '#84d4d5')]: 
        plot_seasonality(forecast, component, fillcolor, linecolor)
with col3:
    st.write("")

st.markdown("""
Le **graphique Weekly (Hebdomadaire)** illustre la saisonnalité hebdomadaire de la variable choisie. 
Il montre comment la variable évolue en moyenne chaque jour de la semaine, du lundi au dimanche. 
Par exemple, si nous observons un pic le vendredi, cela signifie que cette journée a tendance à avoir des valeurs plus élevées 
par rapport aux autres jours de la semaine.

Le **graphique Yearly (Annuel)** dépeint la saisonnalité annuelle. Il montre les tendances et les variations de la variable au fil des mois de l'année, 
aidant à identifier des périodes spécifiques de l'année où la variable augmente ou diminue.

<p style="color: #ad7d67; font-weight:700; font-style:italic; margin: 0px;">
Comment le modèle prend des décisions ?</p>

Nous pouvons examiner un seul composant et voir comment sa contribution aux prévisions globales évolue au fil du temps. 
Les différentes composantes qui influencent les prévisions sont la tendance, les saisonnalités et les régresseurs externes. 
Nous avons déjà observé l'impact des saisonnalités hebdomadaires et annuelles, regardons maintenant les régresseurs externes comme 
les vacances et les autres variables inclues dans le modèle.
""", unsafe_allow_html=True)


#################
# Graphique n°4 #
#################
# Fonction pour construire un graphique pour chaque régresseur
def plot_regresseur(forecast, regresseur, fillcolor, linecolor):
    # Créer un objet de figure Plotly
    fig7 = go.Figure()

    # Extraire le regresseur
    regresseur_df = forecast[[regresseur, 'ds']].dropna()
    values = regresseur_df[regresseur]

    # Ajouter le regresseur
    fig7.add_trace(go.Scatter(
        x=regresseur_df['ds'],
        y=values,
        mode='lines',
        fill='tozeroy',  # Remplir jusqu'à l'axe des y
        fillcolor=fillcolor,  # Couleur de remplissage en hexadécimal
        line=dict(color=linecolor),  # Couleur de ligne en hexadécimal
        name=regresseur,
    ))
    
    # Modifier le nom du régresseur pour le titre
    title_regresseur = regresseur.replace("_additive", "").replace("_multiplicative", "")
    
    # Ajouter un titre
    fig7.update_layout(
        title=f"<b>Impact du régresseur '{title_regresseur}' sur la prévision</b>",
        title_font=dict(size=26, color="#ad7d67"),
        title_xanchor='center',
        title_x=0.5,
        xaxis=dict(
            title="Date",
            type='date',
            range=["2019-01-01", end_date_2022.strftime('%Y-%m-%d')],
            tickformat="%d-%m-%Y", 
            tickangle=-45, 
            dtick="M1", 
            tickfont=dict(size=10), 
            ticklabelmode="period",
            ticklabelposition="outside top"),
        yaxis=dict(
            title=variable,
            tickformat=''
            ),
        width=900, height=350, 
        font=dict(size=12),
        showlegend=False,  # Supprimer la légende
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=16, color="#5e5c5e")))

    # Afficher le graphique
    st.plotly_chart(fig7, theme=None)


# Obtenez une liste de toutes les colonnes qui contiennent "extra_regressors"
regresseurs = [col for col in forecast.columns if "extra_regressors" in col and "_lower" not in col and "_upper" not in col]

# Ajoutez "holidays" à la liste des régresseurs
regresseurs.append('holidays')

# Définissez une correspondance de couleurs pour chaque régresseur
color_mapping = {
    'holidays': ('#f779ae', '#f6408d'),
    'extra_regressors_multiplicative': ('#abdfe1', '#84d4d5'),
    'extra_regressors_additive': ('#abdfe1', '#84d4d5') 
}

# Centrage des graphiques dans la page
col1, col2, col3 = st.columns([0.5,8,0.5])
with col1:
    st.write("")
with col2:
    # Pour chaque regresseur
    for regresseur in regresseurs:
        fillcolor, linecolor = color_mapping.get(regresseur, ('#f779ae', '#f6408d'))  # couleur par défaut si le régresseur n'est pas dans le mappage
        plot_regresseur(forecast, regresseur, fillcolor, linecolor)
with col3:
    st.write("")

st.markdown("""
**Le graphique Holidays (Jours fériés)** met en évidence l'impact des jours fériés sur la variable et des périodes "chocs" comme les confinements. 
Les jours fériés peuvent souvent entraîner des variations significatives, en fonction de la nature de la variable. 
Certains jours fériés, par exemple, peuvent entraîner une augmentation des accidents de la route.

**Le graphique Extra Regressors (Régresseurs supplémentaires)** présente l'influence des variables externes sur la variable principale. 
Ces régresseurs supplémentaires montrent comment ils impactent la prévision de la variable principale.

Les modèles utilisés sont **additifs**. Dans ces modèles, les effets de la saisonnalité sont simplement ajoutés à la tendance. 
Ils sont appropriés lorsque les variations saisonnières restent constantes au fil du temps et ne dépendent pas de la tendance.

<p style="color: #ad7d67; font-weight:700; font-style:italic; margin: 0px;">
Voici comment les prévisions sont calculées pour chaque variable, c'est-à-dire comment les composants sont combinés pour obtenir les prévisions finales.</p>
""", unsafe_allow_html=True)


# Sélectionnez les dates dans le DataFrame de prévisions
forecast_period = forecast[(forecast['ds'] >= start_date_2022) & 
                             (forecast['ds'] <= end_date_2022)]

# Fonction pour calculer les contributions
def calculate_contributions(forecast_period):
    # Liste des composants communs
    common_components = ['trend', 'holidays', 'weekly', 'yearly', 'extra_regressors_additive']
    
    # Calculez les contributions pour les composants communs
    contributions = {component: round(forecast_period[component].sum()) for component in common_components}
    
    # Calculez le total des prévisions
    total_forecast = round(forecast_period['yhat'].sum())
    
    # Ajoutez le total au dictionnaire des contributions
    contributions['Total prévu'] = total_forecast
    
    return contributions

# Utilisez la fonction pour calculer les contributions pour le modèle 'total_accidents'
contributions = calculate_contributions(forecast_period)

# Créer un graphique en cascade
fig8 = go.Figure(go.Waterfall(
    x=list(contributions.keys()),
    y=list(contributions.values()),
    text=[f"<b>{x:.0f}</b>" for x in contributions.values()],
    measure=["relative"] * (len(contributions) - 1) + ["total"],
    connector={"line":{"color":"rgb(255, 255, 255)"}},
    increasing={"marker":{"color":"#042244"}},
    decreasing={"marker":{"color":"#ed3c64"}},
    totals={"marker":{"color":"#66cccc"}},
))

# Mettre à jour la mise en page du graphique
fig8.update_layout(
    title=f"<b>Contribution des composants à la prévision <br> du {start_date_2022.strftime('%d %b %Y')} au {end_date_2022_titre.strftime('%d %b %Y')}</b>",
    title_font=dict(size=26, color="#ad7d67"),
    title_xanchor='center',
    title_x=0.5,
    yaxis_title="Contribution",
    xaxis_title="Composants",
    width=800,
    height=550,
)

# Centrage du graphique dans la page
col1, col2, col3 = st.columns([0.5,8,0.5])
with col1:
    st.write("")
with col2:
    # Pour chaque regresseur
    st.plotly_chart(fig8, theme=None)
with col3:
    st.write("")
# Afficher le graphique


st.write(f"Le modèle prévoit {contributions['Total prévu']:,d} accidents de la route pour la période sélectionnée.")
st.write("Il s'agit de la somme des contributions de cinq composants différents :")
st.write(f"- la tendance : + {contributions['trend']:,d},")
st.write(f"- les jours fériés : + {contributions['holidays']:,d},")
st.write(f"- la saisonnalité hebdomadaire : + {contributions['weekly']:,d},")
st.write(f"- la saisonnalité annuelle : + {contributions['yearly']:,d},")
st.write(f"- et les régresseurs supplémentaires : + {contributions['extra_regressors_additive']:,d}.")


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