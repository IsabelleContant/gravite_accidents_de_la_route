
 <h1 align = "center">Road accidents in France </h1>
 
 
## Presentation
Welcome to our GitHub repository for the project **Road Accidents in France**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).
This repository contains all the necessary code and data to reproduce our analysis and visualizations.

The project has the following file structure:

- **data**: This folder contains all the datasets used in the project.
- **notebooks**: This folder contains Jupyter notebooks with the data analysis and model development.
- **streamlit_app**: This folder contains the code for a Streamlit app to interact with the project.
- **.gitignore**: This file contains the list of files and directories that are ignored by Git.
- **Les_accidents_de_la_Route_en_France-DataScientest-Projet_Fil_Rouge.pdf**: This file contains the final report for the project.
- **README.md**: This file that you are currently reading.
- **requirements.txt**: This file contains the list of Python libraries and their versions needed to run the project.

## Context and Objective
Our study aims to ***predict the severity of road accidents in France*** by focusing on four categories of users: uninjured, slightly injured, hospitalized, and killed between 2019 and 2021. The analysis will also identify the risk factors associated with road accidents in France.
The data used in this project is from the French National Road Safety Observatory ([here](https://www.data.gouv.fr/fr/datasets/base-de-donnees-accidents-corporels-de-la-circulation/)). It includes details on the location, time, and circumstances of each accident, as well as information on the vehicles and individuals involved.

## Methodology

The methodology for this project includes:

- Exploratory data analysis to identify trends and patterns in the data.
- Feature engineering to select relevant features and transform them into a format suitable for machine learning algorithms.
- Splitting the data into training and testing sets (see the [data](https://github.com/DataScientest-Studio/Jan23_BDS_Accidents/tree/main/data) directory).
- Training and evaluating several machine learning models to predict the severity of accidents.
- Fine-tuning the selected model and assessing its performance using various metrics.

The results of the project, including model performance metrics and visualizations, will be summarized in 9 Jupyter notebooks located in the [notebooks](https://github.com/DataScientest-Studio/Jan23_BDS_Accidents/tree/main/notebooks) directory divided as follows:
- **Partie_1_EDA_accidents_de_la_route_france**:               
These notebooks contain exploratory data analysis of road accidents in France for the years 2019, 2020, and 2021. We analyze the various factors such as weather conditions, road type, and time of the day that contribute to road accidents. We also visualize the data to get a better understanding of the trends and patterns in road accidents over the years.

- **Partie_2_preprocessing**          
In this notebook, we preprocess the data to prepare it for building predictive models. We handle missing values, perform feature scaling, and encode categorical variables to prepare the data for modeling.

- **Partie_3_Classification_Binaire**                      
In this notebook, considering that the proportion of killed users is low (2.6%), we have chosen to start with a binary classification which consists in modeling the uninjured users (41.8%) against the other three classes (58.2%) in order to use binary classification models on balanced data.

- **Partie_4_Classification_Multiclass**                     
In this notebook, we build a multiclass classification model to predict the severity of road accidents. Several classification models were chosen for evaluation, including basic models (DummyClassifier) and more advanced models (LogisticRegression, RandomForestClassifier,  XGBClassifier,  LGBMClassifier and BalancedRandomForestClassifier).

- **Partie_5_Interprétabilité_des_Predictions**                        
In this notebook, we interpret the predictions of the models built in Partie_3 and Partie_4. We analyze the features that contribute the most to the predictions and visualize the results.

- **Partie_6_Modélisation_Catboost**                      
In this notebook, we use the CatBoost algorithm to build a predictive model for road accidents. CatBoost is a gradient boosting algorithm that is particularly useful for handling categorical variables. We compare the performance of CatBoost with the models built in Partie_4.

- **Partie_7_Classification_OvO.ipynb**                               
In this part, we explore the one-vs-one (OvO) classification technique, which is used for multi-class classification problems. We build an OvO model to predict the severity of road accidents, and evaluate its performance using the same metrics as in Partie 4.

# Contributors

- Anicet Arthur KOUASSI ([LinkedIn](https://www.linkedin.com/in/koffi-anicet-arthur-kouassi-b517bb1a5/?originalSubdomain=fr))
- Isabelle CONTANT ([LinkedIn](https://www.linkedin.com/in/isabellecontant/))
- Omar DIANKHA ([LinkedIn](https://www.linkedin.com/in/omar-diankha-9081161a7/))
- Idelphonse GBOHOUNME ([LinkedIn](https://www.linkedin.com/in/idelphonse-gbohounme/))

## Streamlit App
Additionally, we have also created a Streamlit application to explore the data and make predictions using the trained models. To run the Streamlit application, execute the following command:

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```


## Acknowledgments
- We deeply thank our project mentor **Manon Georget**([Manon-Datascientest](https://fr.linkedin.com/in/manon-georget-b16b111b6)), for her supporting advice, for her detailed additive proposals and for her careful reading of the whole report during it was written.

<!--The app should then be available at [localhost:8501](http://localhost:8501)./>
<!--([LinkedIn](https://www.linkedin.com))/>
