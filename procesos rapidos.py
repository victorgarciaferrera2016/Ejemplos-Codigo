# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:23:12 2024

@author: victo
"""
"""
#genera graficos de forma intantanea pra entendimiento de los datos.

from pandas.plotting import scatter_matrix

attributes = ["Survived","Pclass","Age","Sex","Embarked"]
scatter_matrix(df_train[attributes],figsize = (12,8))


#rellena de una con la median los campos faltantes numericos.

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")
#imputer.fit(df_train)



# Se define un pipeline conmbinando 
# imputer variable numericas (puede ser media, median, u cualquier dato)
# hace OneHotencoder para aplnar la tabla y pasar de nuemricas a categoricas

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
