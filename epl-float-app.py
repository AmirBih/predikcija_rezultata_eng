import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import streamlit.components.v1 as components

#ucitavanje dva csv-a za forme timova (h i a):
forma_h = pd.read_csv("forma_h.csv")
forma_a = pd.read_csv("forma_a.csv")

#ucitavanje modela
model = tf.keras.models.load_model("epl_float_model.h5")

#ucitavanje kodera
with open("onehot_encoder_domacin.pkl","rb") as file:
    onehot_encoder_domacin = pickle.load(file)

with open("onehot_encoder_gost.pkl","rb") as file:
    onehot_encoder_gost = pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)

    
#streamlit aplikacija 
st.title("Predviđanje Pobjednika Meča")
st.title("Engleska Premijer Liga")

#iskljucivanje klubova sa nedovoljnim podacima
iskljuciti = ["Sunderland","Leeds"]

def opcija(opc):
    if opc in iskljuciti:
        return f"{opc} - Nema Podataka"
    return opc

#polja za unošenje - korisnički unos
domacin = st.selectbox("Domacin", onehot_encoder_domacin.categories_[0] ,format_func= opcija)
gost = st.selectbox("Gost", onehot_encoder_gost.categories_[0] ,format_func= opcija)


#Domacin i Gost da ne budu iste ekipe:
if domacin == gost:
    st.warning("Domacin i gost moraju biti različiti.")
    st.stop()

#Pojašnjenje i onemogućavanje iskljucenih timova
if domacin in iskljuciti:
    st.warning("Odabrani Domacin Onemogućen.")
    st.stop()

if gost in iskljuciti:
    st.warning("Odabrani Gost Onemogućen.")
    st.stop()

#Domacin - prikazi formu
forma_domacin = forma_h[forma_h["Klub"] == domacin]["Rezultat"].values[0]

#Domacin - Odredi boju na osnovu vrednosti
if forma_domacin > 0.40:
    color1 = '#28B463' #Zelena '#28B463' 
elif forma_domacin > 0.19:
    color1 = "#98FB98" #Svijetlo zelena
elif forma_domacin < -0.40: 
    color1 = "#C82333"  # Crvena '#FF5733' 
elif forma_domacin < -0.19:
    color1 = "#FFB6C1" #Svijetlo crvena
else:
    color1 = '#D3D3D3'

# Prikaz vrednosti sa obojenim krugom
st.write(f"Forma {domacin}: {forma_domacin}")
components.html(f"""
<div style='display: inline-block; width: 35px; height: 35px; background-color: {color1}; border-radius: 50%;'></div>
""", height=50)


#Gost
forma_gost = forma_a[forma_a["Klub"] == gost]["Rezultat"].values[0]

#Gost - Odredi boju na osnovu vrednosti
if forma_gost > 0.40:
    color1 = "#C82333" # Crvena '#FF5733 
elif forma_gost > 0.19:
    color1 = "#FFB6C1" #Svijetlo crvena
elif forma_gost < -0.40:
    color1 = '#28B463'  #Zelena '#28B463' 
elif forma_gost < -0.19:
    color1 = "#98FB98" #Svijetlo zelena
else:
    color1 = '#D3D3D3'


# Prikaz vrednosti sa obojenim krugom
st.write(f" Forma {gost}: ({forma_gost} invertovan) ")
components.html(f"""
<div style='display: inline-block; width: 35px; height: 35px; background-color: {color1}; border-radius: 50%;'></div>
""", height=50)

#pripremanje podataka za unos
unos_data = pd.DataFrame({
    "Domacin": [domacin],
    "Gost": [gost],
    
})

#JednoVruceKodiranje Osobine H
H_kodiran = onehot_encoder_domacin.transform([[domacin]]).toarray()
H_kodiran_df = pd.DataFrame(H_kodiran, columns = onehot_encoder_domacin.get_feature_names_out(["H"]))

#JednoVruceKodiranje Osobine A
A_kodiran = onehot_encoder_gost.transform([[gost]]).toarray()
A_kodiran_df = pd.DataFrame(A_kodiran, columns = onehot_encoder_gost.get_feature_names_out(["A"]))


#SJedinjavanje Unosa za Domacina i Gosta
unos_data = pd.concat([H_kodiran_df.reset_index(drop=True),A_kodiran_df],axis=1)

#skaliranje podataka za unos
unos_data_skaliran = scaler.transform(unos_data)

#predviđanje pobjednika
predvidjanje = model.predict(unos_data_skaliran)
predvidjanje_pobjednika = predvidjanje[0][0]


st.info(f'Rezultat Predviđanja: {predvidjanje_pobjednika:.2f}')

sentimentalnost = (f"Pobjednik **{domacin}**") if predvidjanje[0][0] >0.55 else (f"Pobjednik **{gost}**") if predvidjanje[0][0] < -0.55 else "Pobjednik Neizvjestan"
st.info(sentimentalnost)


#Bila su 2 Problema: 1-Previdio sam pisanje koda i izostavio "_" posle "...categories"
                #    2-Nisam Dobro Upisao Imena Dvije Osobine iz DataSeta-, bolje receno, treniranog modela, na što je ukazivano displejom 
                #    3-Osobinu "Estimated Salary" sam ubacio medju predvidjacke... a to je osobina koja se predvidja - izbacio sam je posle.

#Ucitavanje ove .py radne knjigu u aplikaciju streamlit lokalno, u terminal=> "streamlit run B2-app.py"
