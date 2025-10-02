import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

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

#polja za unošenje - korisnički unos
domacin = st.selectbox("Domacin", onehot_encoder_domacin.categories_[0])
gost = st.selectbox("Gost", onehot_encoder_gost.categories_[0])


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

st.write(f'Rezultat Predviđanja: {predvidjanje_pobjednika:.2f}')

sentimentalnost = "Pobjednik: Domacin" if predvidjanje[0][0] >0.52 else "Pobjednik: Gost" if predvidjanje[0][0] < -0.52 else "Pobjednik Neizvjestan"
st.write(sentimentalnost)


#Bila su 2 Problema: 1-Previdio sam pisanje koda i izostavio "_" posle "...categories"
                #    2-Nisam Dobro Upisao Imena Dvije Osobine iz DataSeta-, bolje receno, treniranog modela, na što je ukazivano displejom 
                #    3-Osobinu "Estimated Salary" sam ubacio medju predvidjacke... a to je osobina koja se predvidja - izbacio sam je posle.

#Ucitavanje ove .py radne knjigu u aplikaciju streamlit lokalno, u terminal=> "streamlit run B2-app.py"
