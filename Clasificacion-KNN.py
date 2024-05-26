# Importar algoritmos
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df_data = pd.read_excel('iris.xlsx')
print("-" * 30)
print("Cabecera")
print("-" * 30)
print(df_data.head())
print("-" * 30)
print("Información")
print("-" * 30)
print(df_data.info())

# Crear un array (dataframe) con las variables de entrada (X) y otro para la variable de salida (y)
X = df_data.drop('clase', axis=1).values
y = df_data['clase'].values

# Dividir datos en conjunto de "Training" (ej: 80%) y conjunto de "Test" (ej:20%)
# stratify asegura la proporcionalidad de la varible "y" para train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Instanciamos un objeto Clasificador (Ejemplo KNN)
# por defecto n_neighbors=5
knn = KNeighborsClassifier(n_neighbors=8)

# Entrenamos el modelo en base a los datos
knn.fit(X_train, y_train)

# Verificar precisión del modelo a partir del conjunto de TEST
# Precisión solo en un subconjunto del 20% sin hacer cross-validation
print("-" * 30)
print("Verificar precisión del 20% sin hacer cross-validation")
print("-" * 30)
print(knn.score(X_test, y_test))

# Predecir Resultados de salida (y_prediction) a partir de nuevos datos de entrada (X_new)
df_new = pd.read_excel('iris_nuevos_datos.xlsx')
X_new = df_new.values
print("-" * 30)
print("Mostramos los datos nuevos de entrada")
print("-" * 30)
print(X_new)
print("-" * 30)
print("tipo  de datos:")
print(type(X_new))

# Predecimos los nuevos datos de entrada
print("-" * 30)
print("Predecimos:")
y_prediction = knn.predict(X_new)
print(f"Prediccion: {y_prediction}")
print("-" * 30)
print("tipo  de datos:")
print(type(y_prediction))
print("-" * 30)

print("OPTIMIZANDO EL MODELO:")
print("-" * 30)

import numpy as np
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5) #CV = cross validation en 5 partes
knn_cv.fit(X, y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)
print("-" * 30)

print("FLUJO COMPLETO DE CLASIFICACION:")
print("-" * 30)

# Importar algoritmos
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

df_data = pd.read_excel('iris.xlsx')
print("-" * 30)
print("Cabecera")
print("-" * 30)
print(df_data.head())
print("-" * 30)
print("Información")
print("-" * 30)
print(df_data.info())


# Crear un array (dataframe) con las variables de entrada (X) y otro para la variable de salida (y)
X = df_data.drop('clase', axis=1).values
y = df_data['clase'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Construir Modelos en base a los diferentes algoritmos
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluar cada modelo

results = []
names = []
for name, model in models:
 kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
 cv_results = model_selection.cross_val_score(model, X, y, cv=kfold) #No sería incluso necesaria haber separado previamente en training y test
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)

# Seleccionar mejor modelo por observacion
svc = SVC()

# Optimizar modelo

import numpy as np
from sklearn.model_selection import GridSearchCV

Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
svc_cv = GridSearchCV(svc, param_grid, cv=5)
svc_cv.fit(X, y)
svc_cv.best_params_
svc_cv.best_score_

print(svc_cv.best_params_)
print(svc_cv.best_score_)

# Predecir Resultados de salida (y_prediction) a partir de nuevos datos de entrada (X_new)
df_new = pd.read_excel('iris_nuevos_datos.xlsx')
X_new = df_new.values

y_prediction = svc_cv.predict(X_new)
print("Prediccion: {}".format(y_prediction))