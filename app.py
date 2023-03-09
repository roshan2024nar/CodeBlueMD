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

st.set_page_config(page_title = 'CodeBlueMD' , layout= "wide" ,  page_icon = ':hospital:')
z1, z2 = st.columns([30,20])
with z1:
    new_title = '<p style="font-family:verdana; color:Green; font-size: 50px; float:right;">CodeBlueMD</p>'
    st.markdown(new_title, unsafe_allow_html=True)
with z2:
    logo = Image.open("hospital.jpg")
    logo = logo.resize((50, 50))
    st.image(logo)


st.markdown('### Hello and Welcome to CodeBlueMD Please select the below options')

# login = st.button('Login')
# forget_password = st.button('Forget Password')

# hashed_passwords = stauth.Hasher(['abc', 'def']).generate()

# with open('D:/Projects/ML-II/config.yaml') as file:
#     config = yaml.load(file , Loader=yaml.SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )
# name, authentication_status, username = authenticator.login('Login', 'main')
# if login:
#     if authentication_status:
#         authenticator.logout('Logout', 'main')
#         st.write(f'Welcome *{name}*')
#         st.title('Some content')
#     elif authentication_status is False:
#         st.error('Username/password is incorrect')
#     elif authentication_status is None:
#         st.warning('Please enter your username and password')

#     if st.session_state["authentication_status"]:
#         authenticator.logout('Logout', 'main')
#         st.write(f'Welcome *{st.session_state["name"]}*')
#         st.title('Some content')
#     elif st.session_state["authentication_status"] is False:
#         st.error('Username/password is incorrect')
#     elif st.session_state["authentication_status"] is None:
#         st.warning('Please enter your username and password')
#     elif authentication_status:
#         if forget_password:
#             try:
#                 if authenticator.reset_password(username, 'Reset password'):
#                     st.success('Password modified successfully')
#             except Exception as e:
#                 st.error(e)

#     try:
#         if authenticator.register_user('Register user', preauthorization=False):
#             st.success('User registered successfully')
#     except Exception as e:
#         st.error(e)

#     try:
#         username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
#         if username_forgot_pw:
#             st.success('New password sent securely')
#             # Random password to be transferred to user securely
#         else:
#             st.error('Username not found')
#     except Exception as e:
#         st.error(e)

#     try:
#         username_forgot_username, email_forgot_username = authenticator.forgot_username('Forgot username')
#         if username_forgot_username:
#             st.success('Username sent securely')
#             # Username to be transferred to user securely
#         else:
#             st.error('Email not found')
#     except Exception as e:
#         st.error(e)

#     if authentication_status:
#         try:
#             if authenticator.update_user_details(username, 'Update user details'):
#                 st.success('Entries updated successfully')
#         except Exception as e:
#             st.error(e)

#     if authentication_status:
# with open('D:/Projects/ML-II/config.yaml', 'w') as file:
#     yaml.dump(config, file, default_flow_style=False)


df = pd.read_csv("D:/Projects/ML-II/Dataset.csv").dropna(axis = 1)
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
    # symptom = " ".join([i.capitalize() for i in value.split("_")])
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

if search:
    if(len(selcte_symtoms)):
       predicted_disease =  predictDisease(selcte_symtoms)
       st.write('your dieses may simlirr' , predicted_disease)
       with open('disease_data.json','r') as disesase_file:
            disease_data = json.loads(disesase_file.read())

            disease_info = disease_data[predicted_disease]

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

            if image_url is not None:
                res = request.urlopen("https:" + image_url).read()
                image = Image.open(BytesIO(res))

                st.image(image, caption='Sunrise by the mountains')
            if Specialty is not None:
                st.write(Specialty)
            if othernames is not None:
                st.write(othernames)  
            if type1 is not None:
                st.write(type1)
            if causes is not None:
                st.write(causes)
            if riskfactor is not None:
                st.write(riskfactor)
            if diagnostic is not None:
                st.write(diagnostic)
            if treatment is not None:
                st.write(treatment)
            if frequcy is not None:
                st.write(frequcy)
            if deaths is not None:
                st.write(deaths)
            if wiki_name is not None:
                st.write(wiki_name)
            if prevention is not None:
                st.write(prevention)
            if medication is not None:
                st.write(medication)
    else:
        st.write('Please select the symptoms')






# TO GET THE LOCATION
loc = st.button("Get Location")
loc_data = streamlit_js_eval.get_geolocation('SCR')
if loc:
    if loc_data is not None:
        

        latitude = loc_data["coords"]["latitude"]
        longitude = loc_data["coords"]["longitude"]