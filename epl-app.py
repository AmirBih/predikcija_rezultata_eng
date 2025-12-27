import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # Import ispravan modul


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

st.write("Odaberi Timove:")
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
    color3 = '#28B463' #Zelena '#28B463' 
elif forma_domacin > 0.19:
    color3 = "#98FB98" #Svijetlo zelena
elif forma_domacin < -0.40: 
    color3 = "#C82333"  # Crvena '#FF5733' 
elif forma_domacin < -0.19:
    color3 = "#FFB6C1" #Svijetlo crvena
else:
    color3 = '#D3D3D3'

# Prikaz vrednosti sa obojenim krugom
st.markdown(f"""
<div style='display: flex; align-items: center;'>
    <span>Trenutno Stanje {domacin}:</span>
    <div style='display: inline-block; width: 15px; height: 15px; background-color: {color3}; border-radius: 50%; margin-left: 10px;'></div>
</div>""", unsafe_allow_html=True)

#Gost
forma_gost = forma_a[forma_a["Klub"] == gost]["Rezultat"].values[0]

#Gost - Odredi boju na osnovu vrednosti
if forma_gost > 0.40:
    color2 = "#C82333" # Crvena '#FF5733 
elif forma_gost > 0.19:
    color2 = "#FFB6C1" #Svijetlo crvena
elif forma_gost < -0.40:
    color2 = '#28B463'  #Zelena '#28B463' 
elif forma_gost < -0.19:
    color2 = "#98FB98" #Svijetlo zelena
else:
    color2 = '#D3D3D3'


# Prikaz vrednosti sa obojenim krugom
st.markdown(f"""
<div style='display: flex; align-items: center;'>
    <span>Trenutno Stanje {gost}:</span>
    <div style='display: inline-block; width: 15px; height: 15px; background-color: {color2}; border-radius: 50%; margin-left: 10px;'></div>
</div>""", unsafe_allow_html=True)

#ODNOS FORME DOMACINA I GOSTA
odnos_forme = (forma_domacin + forma_gost) / 2

st.write("")
        
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

st.write("Predikcija Modela:",round(predvidjanje[0][0],2))
#----------------------------------------
# Variables for the labels
left_label = domacin
right_label = gost
invertovana_predikcija = round(predvidjanje[0][0], 2) * -1
prediction_value = invertovana_predikcija

# Create the plot
fig, ax = plt.subplots(figsize=(6, 0.5))

# Draw the gradient line
x = np.linspace(-1, 1, 500)
y = np.zeros_like(x)
# Custom gradient colors
colors = [(0, 'black'), (0.3, 'orange'),(0.6, "orange") , (1, 'black')]
cmap = LinearSegmentedColormap.from_list('custom', colors)

# Plot the line with gradient
for i in range(len(x) - 1):
    ax.plot(x[i:i+2], y[i:i+2], color=cmap((x[i] + 1) / 2), linewidth=2)

# Plot the prediction value
ax.plot(prediction_value, 0, "bo", markersize=10)

# Add labels
ax.text(-1.05, 0, left_label, ha='right', va='center', color="blue")
ax.text(1.05, 0, right_label, ha='left', va='center', color="blue")

# Customize plot appearance
ax.set_xlim(-1.2, 1.2)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Display the plot in Streamlit
st.pyplot(fig)
#----------------------------------------
pred = round(predvidjanje[0][0],2)
#st.write("Predikcija Modela:",round(predvidjanje[0][0],2))
#-------------------------------------------------------------
if odnos_forme >= 0.35:
    if predvidjanje[0][0] < -0.35:
        #color1 = "#C82333" # Crvena '#FF5733
        st.warning("Nije Moguće Predviditi Ovaj Meč!")
        st.write("Trenutno Stanje Ekipa i Predikcija Modela Nisu Usklađeni.")
        st.stop()
if odnos_forme <= -0.2:
    if predvidjanje[0][0] >= 0.2:
        #color1 = "#C82333" # Crvena '#FF5733
        st.warning("Predikcija i trenutno stanje modela nisu usklađeni.")
if odnos_forme >= -0.25 and odnos_forme <= 0.25:
    if predvidjanje[0][0] >= -0.55 and predvidjanje[0][0] <= 0.55:
        st.success("Trenutno stanje ekipa i predikcija modela su usklađeni.")
if odnos_forme >= -0.25 and odnos_forme <= 0.25:
    if predvidjanje[0][0] <= -0.55 or predvidjanje[0][0] >= 0.55:
        st.success("Trenutno stanje ekipa i predikcija modela su usklađeni.")
elif odnos_forme >= 0:
    if predvidjanje[0][0] >=-0.20:
        #color1 = '#28B463' #Zelena '#28B463'
         st.success("Trenutno stanje ekipa i predikcija modela su usklađeni.")
elif odnos_forme <=0:
    if predvidjanje[0][0] <= 0:
        #color1 = '#28B463' #Zelena '#28B463'
         st.success("Trenutno stanje ekipa i predikcija modela su usklađeni.")

#-------------------------------------------------------------
st.write("Zaključak:")

if pred >= 0.55:
    st.write(f"Model je odredio **{domacin}** kao pobjednika.")
elif pred < 0.55 and pred >= 0.20:
    st.write(f"**{domacin}** ima prednost ali utakmica je neizvjesna.")
elif pred < 0.20 and pred > -0.20:
    st.write(f"Model nije odredio pobjednika i veoma je neizvjesno.")
elif pred < -0.20 and pred > -0.55:
    st.write(f"**{gost}** ima prednost ali utakmica je neizvjesna.")
elif pred <= -0.55:
    st.write(f" Model je odredio **{gost}** kao pobjednika.")


#----------------------------------------

#st.info(f'Rezultat Predviđanja: {predvidjanje_pobjednika:.2f}')

sentimentalnost = (f"Pobjednik **{domacin}**") if predvidjanje[0][0] >0.55 else (f"Pobjednik **{gost}**") if predvidjanje[0][0] < -0.55 else "Pobjednik Neizvjestan"
st.info(sentimentalnost)

        
st.write("")

odnos_f_p = (odnos_forme + predvidjanje[0][0]) / 2

# Inicijalizacija stanja sesije za dugme
if 'show_rows' not in st.session_state:
    st.session_state.show_rows = False

# Funkcija za promenu stanja
def toggle_rows():
    st.session_state.show_rows = not st.session_state.show_rows

# Dugme za prikaz/sakrivanje
st.button('Pojedinosti', on_click=toggle_rows)

# Prikazivanje redova informacija na osnovu stanja sesije
if st.session_state.show_rows:
    st.write("Predikcija pobjednika meča radi u opsegu od 1 do -1:")
    st.write("1. Ako je taj broj veći od 0.5, model predvidja odabranog domaćina kao pobjednika.")
    st.write("2. Što je predvidjanje bliže nuli, pobjednik meča je neizvjesniji.")
    st.write("3. Ako model predvidja blizu -1, pobjednik meča je taj odabrani gost.")
    


#Bila su 2 Problema: 1-Previdio sam pisanje koda i izostavio "_" posle "...categories"
                #    2-Nisam Dobro Upisao Imena Dvije Osobine iz DataSeta-, bolje receno, treniranog modela, na što je ukazivano displejom 
                #    3-Osobinu "Estimated Salary" sam ubacio medju predvidjacke... a to je osobina koja se predvidja - izbacio sam je posle.

#Ucitavanje ove .py radne knjigu u aplikaciju streamlit lokalno, u terminal=> "streamlit run B2-app.py"
