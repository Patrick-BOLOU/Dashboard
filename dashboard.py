import streamlit as st
import requests as req
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.write("Bienvenu dans l'application SCORING")
X_test=pd.read_csv("X_test_real_data.csv",index_col='SK_ID_CURR')
listIdclient=list(X_test.index)
idClient=st.sidebar.selectbox("Id Client : ",listIdclient)
st.write("Voici le client sélectionné : ",idClient)

st.dataframe(X_test.loc[[int(idClient)]].drop(columns=["TARGET"]))
#predictResul=req.post(url="http://127.0.0.1:8000/predict_client",json={"client_id": idClient}).json()
predictResul=req.post(url="https://api-ps-seven.herokuapp.com/predict_client",json={"client_id": idClient}).json()
#st.write("Resultat de la prédiction : ",predictResul)
probability=float(predictResul["probability"])*100 
if predictResul["prediction"]==0 :
    colordelta='green'
else :
    colordelta='red'

fig = go.Figure(go.Indicator(
    mode='gauge+number+delta',
    value=probability,
    domain={'x':[0,1], 'y':[0,1]},
    title={'text':'Score du client sélectionné','font':{'size':25}},
    delta={'reference':55,'increasing':{'color':colordelta}},
    gauge={ 'axis': {'range': [None, int(100)], 'tickwidth': 1.5, 'tickcolor': "black"},
            'bar': {'color': "#2E00FF"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [{'range': [0, 55], 'color': 'Lime'},{'range': [55, probability], 'color': 'red'}],
            'threshold': {'line': {'color': "red", 'width': 4}}
    }
))
fig.update_layout(paper_bgcolor='#BFFFFC',font={'color':'darkblue','family':'Arial'})
st.write(fig)

listVarToPlot=["AMT_CREDIT","AMT_ANNUITY","DAYS_EMPLOYED","CNT_CHILDREN"]
featureToPlot=st.selectbox('',listVarToPlot)
valClient=int(X_test.loc[[int(idClient)]][featureToPlot].values)
fig,ax=plt.subplots()
t0=X_test.loc[X_test["TARGET"] == 0]
t1=X_test.loc[X_test["TARGET"] == 1]
ax=sns.kdeplot(t0[featureToPlot],color="green",label="Client Solvable")
ax=sns.kdeplot(t1[featureToPlot],color="red",label="Client Non Solvable")

plt.axvline(valClient,color="blue")
plt.legend(fontsize=10)
st.pyplot(fig)
