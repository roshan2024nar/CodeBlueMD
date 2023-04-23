import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st
import streamlit_js_eval
import json
import yaml
import streamlit_authenticator as stauth
import streamlit as st
from PIL import Image
from urllib import request
from io import BytesIO
from st_clickable_images import clickable_images
import base64
import folium
from streamlit_folium import st_folium , folium_static
import emailer
import plotly.graph_objects as go 


st.set_page_config(page_title = 'CodeBlueMD' , layout= "wide" ,  page_icon = ':hospital:')
z1, z2 = st.columns([30,20])
with z1:
    new_title = '<p style="font-family:verdana; color:Green; font-size: 50px; float:right;">CodeBlueMD</p>'
    st.markdown(new_title, unsafe_allow_html=True)
with z2:
    logo = Image.open("logo.jpg")
    logo = logo.resize((50, 50))
    st.image(logo)

x1 ,x2 , x3 = st.columns([3,20,3])
with x2:
    st.markdown('### Welcome to CodeBlueMD, The emergency care solution you can count on...')



df = pd.read_csv("Dataset.csv").dropna(axis = 1)
le = LabelEncoder()
le.fit_transform(df['prognosis'])
x = df.iloc[:,:-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.2, random_state = 0)
rf = None
with open("rf.pkl", "rb") as rf_File:
    rf = pickle.load(rf_File)


symptoms = x.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom_index[value] = index


data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":le.classes_
}

def predictDisease(symptoms):
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    input_data = np.array(input_data).reshape(1,-1)
    return rf.predict(input_data)[0]

col1, col2, col3 = st.columns([7,1,3])
with col1 :
    selcte_symtoms = st.multiselect('Select The Symptoms Here ' , symptoms)
    

with col3:
    st.write('##')
    search = st.button('Search')


if  selcte_symtoms:
    if(len(selcte_symtoms)):
        predicted_disease =  predictDisease(selcte_symtoms)

    x1 , x2 , x3 = st.columns([4,10,4])
    with x2:
        st.markdown("###")
        st.markdown("#### Based on our analysis, we predict that you may have  " + predicted_disease)
    with open('disease_data.json','r') as disesase_file:
            disease_data = json.loads(disesase_file.read())

            disease_info = disease_data[predicted_disease]

            defination = disease_info["defination"] if "defination" in disease_info else None
            Specialty = disease_info["Specialty"] if "Speciality" in disease_info else None
            image_url = disease_info["Wiki_Img"] if "Wiki_Img" in disease_info else None
            othernames = disease_info["Other names"] if "Other names" in disease_info else None
            type1 = disease_info["Types"] if "Types" in disease_info else None
            causes = disease_info["Causes"] if "Causes" in disease_info else None
            riskfactor = disease_info["Risk factors"] if "Risk factors" in disease_info else None
            diagnostic = disease_info["Diagnostic method"] if "Diagnostic method" in disease_info else None
            treatment = disease_info["Treatment"] if "Treatment" in disease_info else None
            frequcy = disease_info["Frequency"] if "Frequency" in disease_info else None
            deaths = disease_info["Deaths"] if "Deaths" in disease_info else None
            wiki_name = disease_info["Wiki_Name"] if "Wiki_Name" in disease_info else None
            prevention = disease_info["Prevention"] if "Prevention" in disease_info else None
            medication = disease_info["Medication"] if "Medication" in disease_info else None
            fields = disease_info["Fields ofemployment"] if "Fields ofemployment" in disease_info else None
            Sys = disease_info["Symptoms"] if "Symptoms" in disease_info else None



            rows = []
            
            if image_url is not None:
                x1,x2,x3 =   st.columns([6,10,5])
                with x2:
                    res = request.urlopen(image_url).read()
                    image = Image.open(BytesIO(res))
                    image = image.resize((500,500))

                    st.image(image)
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
                rows.append(["Prevention"  , prevention])

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

            # table
            df2=outputdframe.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
            st.table(df2)
            




else:
    with open('articles.json','r') as articles_file:
        articles = json.loads(articles_file.read())

        for article in articles:
            name = article["name"]
            art_img = article["image"]
            data = article['data']
            red_url = article['redirect_url']
            c1, c2 = st.columns([9,30])
            with c1:
                
                img = request.urlopen(art_img).read()
                art_image = Image.open(BytesIO(img))
                image = art_image.resize((250,200))
                st.image(image)

            with c2:
                st.markdown('#####')
                st.markdown(f"#### {name}")
                st.markdown(f"{data}\n [Read More... ]({red_url})")



st.markdown("---")
loc_data = streamlit_js_eval.get_geolocation('SCR')
latitude = loc_data["coords"]["latitude"]
longitude = loc_data["coords"]["longitude"]


if latitude is not None and longitude is not None:
    m = folium.Map(location=[latitude,longitude])
    folium.Marker(location= [latitude , longitude]).add_to(m)
    st_folium(m , width=1400 , height=625 , zoom = 16)

else:
    latitude = 19.085649
    longitude = 72.908218
    m = folium.Map(location=[latitude,longitude])
    folium.Marker(location= [latitude , longitude]).add_to(m)
    st_folium(m , width=1400 , height=625 , zoom = 16)



# TO GET THE LOCATION
x1,x2,x3 = st.columns([6,10,5])
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
    img_style={"margin": "5px", "height": "300px" , "width": "300px"},
)




GMAIL_EMAIL = st.secrets["GMAIL_EMAIL"]
GMAIL_PASSWORD = st.secrets["GMAIL_PASSWORD"]

if not clicked:
    if loc_data is not None:
        body = f"Latitude- {latitude}\nLongitude- {longitude}"
        emailer.send_email_using_gmail(GMAIL_EMAIL, GMAIL_PASSWORD, "roshan.tiwari.24.5.2001@gmail.com", "Emergency Help", body)

        x1,x2,x3 = st.columns([7,11,5])
        with x2:
            st.markdown("#### Help Requested Succesfully, Help Is on the way ")
        



