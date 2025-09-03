import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st
import streamlit_js_eval
import json
import yaml
import streamlit_authenticator as stauth
from PIL import Image
from urllib import request
from io import BytesIO
from st_clickable_images import clickable_images
import base64
import folium
from streamlit_folium import st_folium
import emailer
import plotly.graph_objects as go

st.set_page_config(page_title='CodeBlueMD', layout="wide", page_icon=':hospital:')

# =======================
# Header section
# =======================
z1, z2 = st.columns([30, 20])
with z1:
    new_title = '<p style="font-family:verdana; color:Green; font-size: 50px; float:right;">CodeBlueMD</p>'
    st.markdown(new_title, unsafe_allow_html=True)
with z2:
    logo = Image.open("logo.jpg")
    logo = logo.resize((50, 50))
    st.image(logo)

x1, x2, x3 = st.columns([3, 20, 3])
with x2:
    st.markdown('### Welcome to CodeBlueMD, The emergency care solution you can count on...')

# =======================
# ML model + dataset
# =======================
df = pd.read_csv("Dataset.csv").dropna(axis=1)
le = LabelEncoder()
le.fit_transform(df['prognosis'])
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = None
with open("rf.pkl", "rb") as rf_File:
    rf = pickle.load(rf_File)

symptoms = x.columns.values
symptom_index = {value: index for index, value in enumerate(symptoms)}

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": le.classes_
}


def predictDisease(symptoms):
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    input_data = np.array(input_data).reshape(1, -1)
    return rf.predict(input_data)[0]


# =======================
# User input for symptoms
# =======================
col1, col2, col3 = st.columns([7, 1, 3])
with col1:
    selcte_symtoms = st.multiselect('Select The Symptoms Here ', symptoms)

with col3:
    st.write('##')
    search = st.button('Search')

# =======================
# Prediction branch
# =======================
if selcte_symtoms:
    if len(selcte_symtoms):
        predicted_disease = predictDisease(selcte_symtoms)

    x1, x2, x3 = st.columns([4, 10, 4])
    with x2:
        st.markdown("###")
        st.markdown("#### Based on our analysis, we predict that you may have  " + predicted_disease)

    with open('disease_data.json', 'r') as disesase_file:
        disease_data = json.loads(disesase_file.read())
        disease_info = disease_data[predicted_disease]

        defination = disease_info.get("defination")
        Specialty = disease_info.get("Specialty")
        image_url = disease_info.get("Wiki_Img")
        othernames = disease_info.get("Other names")
        type1 = disease_info.get("Types")
        causes = disease_info.get("Causes")
        riskfactor = disease_info.get("Risk factors")
        diagnostic = disease_info.get("Diagnostic method")
        treatment = disease_info.get("Treatment")
        frequcy = disease_info.get("Frequency")
        deaths = disease_info.get("Deaths")
        wiki_name = disease_info.get("Wiki_Name")
        prevention = disease_info.get("Prevention")
        medication = disease_info.get("Medication")
        fields = disease_info.get("Fields ofemployment")
        Sys = disease_info.get("Symptoms")

        rows = []

        # Disease image fetch with fallback
        if image_url is not None:
            try:
                x1, x2, x3 = st.columns([6, 10, 5])
                with x2:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    req = request.Request(image_url, headers=headers)
                    res = request.urlopen(req).read()
                    image = Image.open(BytesIO(res))
                    image = image.resize((500, 500))
                    st.image(image)
            except Exception as e:
                st.warning(f"Could not load image: {e}")

        if defination is not None:
            rows.append(["Definition", defination])
        if Sys is not None:
            rows.append(["Symptoms", Sys])
        if riskfactor is not None:
            rows.append(["Risk Factor", riskfactor])
        if diagnostic is not None:
            rows.append(["Diagnostic Methods", diagnostic])
        if treatment is not None:
            rows.append(["Treatment", treatment])
        if type1 is not None:
            rows.append(["Type", type1])
        if causes is not None:
            rows.append(["Causes", causes])
        if prevention is not None:
            rows.append(["Prevention", prevention])
        if Specialty is not None:
            rows.append(["Specialty", Specialty])
        if frequcy is not None:
            rows.append(["Frequency", frequcy])
        if deaths is not None:
            rows.append(["Deaths", deaths])

        outputdframe = pd.DataFrame(rows, columns=["Name", "Information"])
        th_props = [
            ('font-size', '25px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#6d6d6d'),
            ('background-color', '#f7ffff')
        ]
        td_props = [
            ('font-size', '20px')
        ]
        td1_props = {
            ("display", "none")
        }
        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
            dict(selector="th:first-child", props=td1_props)
        ]
        df2 = outputdframe.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
        st.table(df2)

# =======================
# Articles branch
# =======================
else:
    with open('articles.json', 'r') as articles_file:
        articles = json.loads(articles_file.read())

        for article in articles:
            name = article["name"]
            art_img = article["image"]
            data = article['data']
            red_url = article['redirect_url']

            c1, c2 = st.columns([9, 30])
            with c1:
                try:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    req = request.Request(art_img, headers=headers)
                    img = request.urlopen(req).read()
                    art_image = Image.open(BytesIO(img))
                    image = art_image.resize((250, 200))
                    st.image(image)
                except Exception as e:
                    st.warning(f"Could not load article image: {e}")

            with c2:
                st.markdown('#####')
                st.markdown(f"#### {name}")
                st.markdown(f"{data}\n [Read More... ]({red_url})")

# =======================
# Map + Geolocation
# =======================
st.markdown("---")
loc_data = streamlit_js_eval.get_geolocation('SCR')
latitude = loc_data["coords"]["latitude"]
longitude = loc_data["coords"]["longitude"]

if latitude is not None and longitude is not None:
    m = folium.Map(location=[latitude, longitude])
    folium.Marker(location=[latitude, longitude]).add_to(m)
    st_folium(m, width=1400, height=625, zoom=16)
else:
    latitude = 19.085649
    longitude = 72.908218
    m = folium.Map(location=[latitude, longitude])
    folium.Marker(location=[latitude, longitude]).add_to(m)
    st_folium(m, width=1400, height=625, zoom=16)

# =======================
# Emergency button
# =======================
x1, x2, x3 = st.columns([6, 10, 5])
with x2:
    st.markdown("#### In case of emegrgency Please press the below button ")

images = []
for file in ['hospital.png']:
    with open(file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
        images.append(f"data:image/png;base64,{encoded}")

clicked = clickable_images(
    images,
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap", "cursor": "pointer"},
    img_style={"margin": "5px", "height": "300px", "width": "300px"},
)

GMAIL_EMAIL = st.secrets["GMAIL_EMAIL"]
GMAIL_PASSWORD = st.secrets["GMAIL_PASSWORD"]

if not clicked:
    if loc_data is not None:
        body = f"Latitude- {latitude}\nLongitude- {longitude}"
        emailer.send_email_using_gmail(
            GMAIL_EMAIL, GMAIL_PASSWORD,
            "roshan.tiwari.24.5.2001@gmail.com",
            "Emergency Help", body
        )
        x1, x2, x3 = st.columns([7, 11, 5])
        with x2:
            st.markdown("#### Help Requested Succesfully, Help Is on the way ")
