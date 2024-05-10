# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:13:17 2024

@author: victo
"""

#########LIBRERIAS A UTILIZAR
import  numpy as np
import  pandas as pd
import  math      #para redondear
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix


#########IMPORTANDO LA DATA ###################3
#
# 1.- importar datos desde web
#url_test = 'https://www.kaggle.com/competitions/titanic/data?select=test.csv'
url_test  = 'https://drive.google.com/drive/home/test.csv'
url_train = 'https://drive.google.com/drive/home/train.csv'


#df_test  = pd.read_csv(url_test)
#df_train = pd.read_csv(url_train)

# 2.- se guardar los datos en archivo pc si vienen de fuente externa
dir_test  = 'F:/CAPACITACION/MODELO MLEARNING/recursos/titanic_test.csv'
dir_train = 'F:/CAPACITACION/MODELO MLEARNING/recursos/titanic_train.csv'
#df_test.to_csv(dir_test)
#df_train.to_csv(dir_train)

# 3.- importar datos desde pc y asignar csv

df_test  = pd.read_csv(dir_test)
df_train = pd.read_csv(dir_train)

###NOT a la columna entrenamiento tiene una columna extra que indica 
# si el pasajero sobrevivio o no

#print(df_test.head())
#print(df_train.head())

#########ENTENDIMIENTO DE LA DATA ##############
#VErifico la cantidad de datos de cada dataset
print('Cantidad de datos')
print(df_train.shape)
print(df_test.shape)

#verifico que tipos de datos contienen ambos datasets

#opcion # 1
print('Tipos de datos')
print(df_train.info())
print(df_test.info())


#verifico qsi hay datos faltantes
#opcion #1
print('Datos Faltantes :')
print(df_train.isnull().sum())
print(df_test.isnull().sum())

#opcion #2
print('Datos Faltantes :')
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

#Verifico estadisticas de cada datasets
print('estadisticsa de cada datasets :')
print(df_train.describe())
print(df_test.describe())

#########PREPROCESAMIENTO DE LA DATA############

#sexo     : Cambio de datos columna sexo. string a numerico (1,0)
df_train.replace( ['female','male'], [0,1], inplace = True)
df_test.replace( ['female','male'], [0,1], inplace = True)

#Embarque : Cambio de los datos de embarque en numeros
df_train.replace( ['Q','S','C'], [0,1,2], inplace = True)
df_test.replace( ['Q','S','C'], [0,1,2], inplace = True)


#Edad     : Reemplazos los datos faltantes por el promedio
print(df_train['Age'].mean())
print(df_test['Age'].mean())

promedio = int(math.ceil((df_train['Age'].mean() + df_test['Age'].mean())/2))

#opcion # 1
df_train['Age']=df_train['Age'].fillna(promedio)
df_test['Age']=df_test['Age'].fillna(promedio)

#opcion # 2
df_train['Age']= df_train['Age'].replace(np.nan,promedio)
df_test['Age'] = df_test['Age'].replace(np.nan,promedio)

#Edad     : Creo rango de edades


#VGF analizo con bar e histogramas la distribucion de edades.
edades= df_train.groupby('Age').size() 
lista_edades = (list(edades.keys()))
cantidad_personas = list(edades.values)
plt.bar(lista_edades, cantidad_personas, color='blue')
plt.hist(df_train['Age'],bins='auto',color='blue', edgecolor='black')

bins= [0, 8, 15, 18, 25, 40, 60, 100]
#nombre_bins = ['1','2','3','4','5','6','7']
nombre_bins = [1 ,2 ,3 ,4 , 5, 6 , 7]

df_train['Age']= pd.cut(df_train['Age'], bins, labels = nombre_bins)
df_test['Age'] = pd.cut( df_test['Age'], bins, labels = nombre_bins)

#Cabin    : Se elimina columna. Muchos nulos.687 de 891. y 327 de 418
df_train.drop(['Cabin'],axis=1,inplace= True)
df_test.drop(['Cabin'],axis=1,inplace= True)

#  COLUMNAS       : se elimina por no ser necesarias para analisis
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)
df_test  = df_test.drop(['Name','Ticket'], axis=1)

#  FILAS       : se eiminan filsa con datos perdidos o nulos
df_train.dropna(axis=0, how = 'any' , inplace=True)
df_test.dropna( axis=0, how = 'any' , inplace=True)



#Verifico Datos

print(df_train.isnull().sum())
print(df_test.isnull().sum())

print(df_train.shape)
print(df_test.shape)

print(df_train.head())
print(df_test.head())

# Verifico Matriz de Correlación.

plt.figure(figsize=(12,10))
cor=df_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()



#########APLICACION DE ALGORITMOS DE ML#########
#########APLICACION DE ALGORITMOS DE ML#########
#########APLICACION DE ALGORITMOS DE ML#########
#########APLICACION DE ALGORITMOS DE ML#########
#########APLICACION DE ALGORITMOS DE ML#########
#########APLICACION DE ALGORITMOS DE ML#########
#########APLICACION DE ALGORITMOS DE ML#########
# 1 : separar la columna con l ainformacion de los sobrevivientes
# OJO : acá paso los datos de un DF a un ARRAY
#X = np.array(df_train.drop(['Survived'],axis=1))
#y = np.array(df_train['Survived'])

X = df_train.drop(['Survived'],axis=1)
y = df_train['Survived']
# 2 : Separo datos de train en train y 

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2)


#
# analizando la varianza
#
from sklearn.feature_selection import VarianceThreshold
var_thr=VarianceThreshold(threshold=0)
var_thr.fit(X_train)
var_thr.get_support()

columna_a_mantener = X_train.columns[var_thr.get_support()]
columna_a_mantener

columna_a_eliminar = [col for col in X_train.columns if col not in columna_a_mantener]
columna_a_eliminar

# Verifico Matriz de Correlación.

plt.figure(figsize=(12,10))
cor=X_train.corr()
heatmap_cor = sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)

heatmap_cor.set_xticklabels(heatmap_cor.get_xticklabels(), rotation=90, verticalalignment='top')
for i in range(len(cor)):
    for j in range(len(cor)):
       heatmap_cor.text(j + 0.5 , i + 0.5 ,  f'{cor.iloc[i,j]:.2f}' ,
                      ha="center", va="center", color="black", fontsize=10)

#plt.tight_layout()
#plt.show()









column_corr = set() # creates empty set
len_columns = len(cor.columns)
for i in range(len_columns):
        print('i : ',i)
        for j in range(i) : # not takes if i == j
            print('j : ',j)
            print('valor antes if if :', cor.iloc[i,j])
            if abs(cor.iloc[i,j]) > 0.3:
                print('valor despues if :', cor.iloc[i,j])
                colname=cor.columns[i]
                print('Columna :', colname)
                column_corr.add(colname)
column_corr

##############################################
##############################################
# 3 : Regresion Logistica (Supervisada)
algoritmo_lg = LogisticRegression()
algoritmo_lg.fit(X_train, y_train)
print('Precision Regresion Logidtica : ')
print(algoritmo_lg.score(X_train, y_train))

# 3.1 . Evaluando la performace del algoritmo ( MAE - MSE - RMSE)
# esta vlidacion se hcae con los subconjutnos split
from sklearn import metrics
prediccion_lg = algoritmo_lg.predict(X_test)
MAE =    metrics.mean_absolute_error(y_test,prediccion_lg)
MSE =    metrics.mean_squared_error(y_test,prediccion_lg)
RMSE =   np.sqrt(metrics.mean_squared_error(y_test,prediccion_lg))
VarScore = metrics.explained_variance_score(y_test,prediccion_lg)
print("LG : ", RMSE, MSE)

##############################################
##############################################
# 4 : Support Vector MAchines
#
algoritmo_svc = SVC()
algoritmo_svc.fit(X_train, y_train)
print('Precision Vector MAchines : ')
print(algoritmo_svc.score(X_train, y_train))

# 4.1 . Evaluando la performace del algoritmo ( MAE - MSE - RMSE)
# esta vlidacion se hcae con los subconjutnos split
from sklearn import metrics
prediccion_svc = algoritmo_svc.predict(X_test)
MAE =    metrics.mean_absolute_error(y_test,prediccion_svc)
MSE =    metrics.mean_squared_error(y_test,prediccion_svc)
RMSE =   np.sqrt(metrics.mean_squared_error(y_test,prediccion_svc))
VarScore = metrics.explained_variance_score(y_test,prediccion_svc)
print("SVC : ",RMSE , MSE)



##############################################
##############################################
# 5 : K neighbors
algoritmo_knn = KNeighborsClassifier(n_neighbors=3)
algoritmo_knn.fit(X_train, y_train)
print('Precision RK neighbors : ')
print(algoritmo_knn.score(X_train, y_train))

# 5.1 . Evaluando la performace del algoritmo ( MAE - MSE - RMSE)
# esta vlidacion se hcae con los subconjutnos split
from sklearn import metrics
prediccion_knn = algoritmo_knn.predict(X_test)
MAE =    metrics.mean_absolute_error(y_test,prediccion_knn)
MSE =    metrics.mean_squared_error(y_test,prediccion_knn)
RMSE =   np.sqrt(metrics.mean_squared_error(y_test,prediccion_knn))
VarScore = metrics.explained_variance_score(y_test,prediccion_knn)
print("KNN : ",RMSE , MSE)

#########PREDICCION UTILIZANDO LOS MODELOS######


ids = df_test['PassengerId']

# 3 : Regresion Logistica (Supervisada)
prediccion_lg = algoritmo_lg.predict(df_test.drop('PassengerId',axis=1))
out_lg        = pd.DataFrame( {'PassengerId' : ids,'Survived' : prediccion_lg} )
print('Prediccion Regresion Logistica : ')
print(out_lg)

# 4 : Support Vector MAchines
prediccion_svc = algoritmo_svc.predict(X_test.drop('PassengerId',axis=1))
out_svc        = pd.DataFrame( {'PassengerId' : ids,'Survived' : prediccion_svc} )
print('Prediccion Regresion Logistica : ')
print(out_svc)

# 5 : K neighbors

prediccion_knn = algoritmo_knn.predict(df_test.drop('PassengerId',axis=1))
out_knn       = pd.DataFrame( {'PassengerId' : ids,'Survived' : prediccion_knn} )
print('Prediccion Regresion Logistica : ')
print(out_knn)

print(df_train.columns)


#####
#### Genera natriz de correlacion rapido.

attributes = ["Survived","Pclass","Age","Sex","Embarked"]
scatter_matrix(df_train[attributes],figsize = (12,8))


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
#imputer.fit(df_train)



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Crear un imputador para manejar valores faltantes
imputer = SimpleImputer(strategy="median")

# Definir las columnas categóricas y numéricas
columnas_numericas   = ['Age', 'Pclass']
columnas_categoricas = ['Embarked', 'Sex']

# Definir los pasos del preprocesamiento
preprocesamiento = ColumnTransformer(
    transformers=[('numéricas'  , imputer        , columnas_numericas),
                  ('categóricas', OneHotEncoder(), columnas_categoricas)
                 ]
                                  )

# Crear un pipeline con el preprocesamiento y el imputador
pipeline = Pipeline(steps=[('preprocesamiento', preprocesamiento)])

# Ajustar y transformar los datos
datos_procesados = pipeline.fit_transform(df_train)


