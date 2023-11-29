#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np 
import pandas as pd 
import plotly.express as px
import warnings  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style
import seaborn as sns
import seaborn as sns
warnings.filterwarnings("ignore")
C_NOMINAL=185
YEAR = 2023
MES = 10


# In[31]:


df=pd.read_excel(r"C:\Users\adelacruz\OneDrive - Minera Boroo Misquichilca S.A\Escritorio\DESCARGAS_LN.xlsx")
df


# In[32]:


#filtro para las condiciones de las cargas 10 10 20 
conditions = [
    (df['Ton_Med'] >= 200),
    (df['Ton_Med'] >= 190) & (df['Ton_Med'] < 200),
    (df['Ton_Med'] >= 180) & (df['Ton_Med'] < 190),
    (df['Ton_Med'] >= 150) & (df['Ton_Med'] < 180),
    (df['Ton_Med'] < 180),
]
labels = ['MAYOR200', 'MAYOR190MENOR200', 'MAYOR180MENOR190', 'MAYOR150MENOR180','MENOR180']
df['categoria'] = np.select(conditions, labels, default='Not Specified')
df["Camion"]=[s.strip() for s in df["Camion"]]
df = df[(df['Año'] == YEAR) & (df['Mes'] ==9)|(df['Mes'] ==10)]

#filtro para las condiciones de camiones y sus balanzas
conditions = [
    (df["Ton_Med"] >= 210),
    (df["Ton_Med"] < 120),
]
labels = ["OVER_W","UNDER_W"]
df['pesaje'] = np.select(conditions, labels, default="NORMAL_W")
df["pesaje"].value_counts()
df


# In[33]:


#FILTRADO POR CANTIDAD DE TONELAJE
df1=df[df['Ton_Med']>0]
df1


# In[34]:


df1.info()


# In[35]:


df1.describe().T


# In[36]:


df1['Descarga'].unique
df1.groupby(['Descarga']).mean('Ton_Med')


# In[37]:


#FILTRADO POR ZONA DE DESCARGA
df1['Descarga']=[s.strip() for s in df1['Descarga']]
df1['Pala']=[s.strip() for s in df1['Pala']]
df1=df1[(df1['Descarga'] == 'F6-4200-001') | (df1['Descarga'] == 'F6-4220-002')]
df1


# In[38]:


df1.groupby(['Camion']).mean('Ton_Med')
df1['Camion'].value_counts()


# In[39]:


camiones = df1['Camion'].unique()
camiones_ordenados = sorted(camiones)  
num_camiones = len(camiones_ordenados)

num_filas = (num_camiones // 4) + 1
num_columnas = min(num_camiones, 4)

plt.figure(figsize=(20, 5 * num_filas))
plt.suptitle("Distribución de Ton_Med por Camión", fontsize=30,color="green")

for i, camion in enumerate(camiones_ordenados, 1):
    plt.subplot(num_filas, num_columnas, i)
    df_f = df1[df1['Camion'] == camion]
    sns.distplot(df_f['Ton_Med'], color="blue", kde=True, bins=80, label=camion)
    plt.axvline(180,label='180TON', c='red', linestyle='-', linewidth=1)
    plt.axvline(200,label='200TON', c='red', linestyle='-', linewidth=1)
    plt.xlabel("Ton_Med")
    plt.legend()

plt.show()


# In[40]:


plt.figure(figsize=(14,3))
sns.violinplot(x=df1["Ton_Med"], width=1);
plt.axvline(np.percentile(df1["Ton_Med"],10), label='10%', c='orange', linestyle=':', linewidth=1)
plt.axvline(np.percentile(df1["Ton_Med"],25), label='25%', c='darkblue', linestyle=':', linewidth=1)
plt.axvline(np.percentile(df1["Ton_Med"],50), label='50%', c='green', linestyle=':', linewidth=1)
plt.axvline(np.percentile(df1["Ton_Med"],75), label='75%', c='gold', linestyle=':', linewidth=1)
plt.axvline(np.percentile(df1["Ton_Med"],90), label='90%', c='red', linestyle=':', linewidth=1)
plt.axvline(180,label='180TON', c='red', linestyle='-', linewidth=2)
plt.axvline(200,label='200TON', c='red', linestyle='-', linewidth=2)
plt.legend()
plt.title('weight control')


# In[44]:


a2=df1["categoria"].value_counts().sort_values(ascending=False)
df1["categoria"].hist(figsize=(8,8))
a2


# In[42]:


resumen = df1.groupby(['Camion', 'pesaje'])['pesaje'].count().unstack(fill_value=1)
resumen['TOTAL'] = resumen.sum(axis=1)
resumen['PROB_NORMAL_W'] = resumen['NORMAL_W']*100 / resumen['TOTAL']
resumen = resumen.sort_values(by='PROB_NORMAL_W', ascending=False)
resumen['PROM_W']= df1.groupby('Camion')["Ton_Med"].mean()
resumen


# In[ ]:




