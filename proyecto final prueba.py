import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles, make_moons, make_classification
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
import matplotlib.patches as mpatches
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import multiclass
from sklearn.svm import SVC
import pickle


## PREPROCESAMIENTO DE DATOS
data_train = pd.read_csv('/Users/bustacara-js/Documents/IA codigos python/Proyecto/Resultados_de_An_lisis_de_Laboratorio_Suelos_en_Colombia (1).csv',sep=',',low_memory=False,)
print(sb.heatmap(data_train.corr(),annot=True))
data_train=data_train.drop(['numfila','FechaAnalisis','Estado','Secuencial'],axis=1)
columns=['Departamento','Municipio','Cultivo','Tiempo Establecimiento','Topografia','Drenaje','Riego','Fertilizantes aplicados']
columnsfill=['pH agua:suelo 2.5:1.0','Materia orgánica (MO) %','Fósforo (P) Bray II mg/kg','Azufre (S) Fosfato monocalcico mg/kg','Acidez (Al+H) KCL cmol(+)/kg','Aluminio (Al) intercambiable cmol(+)/kg','Magnesio (Mg) intercambiable cmol(+)/kg','Potasio (K) intercambiable cmol(+)/kg','Sodio (Na) intercambiable cmol(+)/kg','capacidad de intercambio cationico (CICE) suma de bases cmol(+)/kg','Couctividad el‚ctrica (CE) relacion 2.5:1.0 dS/m','Hierro (Fe) disponible olsen mg/kg','Cobre (Cu) disponible mg/kg','Manganeso (Mn) disponible Olsen mg/kg','Zinc (Zn) disponible Olsen mg/kg','Boro (B) disponible mg/kg','Hierro (Fe) disponible doble  cido mg/kg','Cobre (Cu) disponible doble acido mg/kg','Manganeso (Mn) disponible doble acido mg/kg','Zinc (Zn) disponible doble  cido mg/kg']
data_train[columns] = data_train[columns].fillna('')
data_train[columnsfill] = data_train[columnsfill].fillna(data_train[columnsfill].mean())


data_train_1 = data_train.copy()
ord_enc=OrdinalEncoder()


for x in columns:
    data_train_1[x] = ord_enc.fit_transform(data_train_1[[x]])

#####################

X=data_train_1.drop(['Cultivo'],axis=1)
y=data_train_1['Cultivo']

scaler=MinMaxScaler()
scaler.fit(X)
data_normalized=pd.DataFrame(scaler.transform(X),columns=X.columns)




#For para verificar mejor n_components

for i in range(1,29):
    pca = decomposition.PCA(n_components=i)
    pca.fit(data_normalized)
    print(i," varianza explicada por cada caracteristica después de PCA:",sum(pca.explained_variance_ratio_))

pca = decomposition.PCA(n_components=7)
pca.fit(data_normalized)
X=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#Seleccion de solver

solvers = {'newton-cg', 'lbfgs', 'sag', 'saga'}

for i in solvers:
    LR = LogisticRegression(penalty='l2',solver='saga',max_iter=200, C=1.0,random_state=None,n_jobs=8,multi_class='ovr',tol=1e-3).fit(X_train, y_train)
    Predicciones1=LR.predict(X_test)
    print('Score: ',LR.score(X_test, y_test)) 
    print('Matthews: ',matthews_corrcoef(y_test,Predicciones2))


LR = LogisticRegression(penalty='l2',solver='saga',max_iter=200, C=1.0,random_state=None,n_jobs=8,multi_class='ovr',tol=1e-3).fit(X_train, y_train)
Predicciones=LR.predict(X_train)
Predicciones2=LR.predict(X_test)
print('Score: ',LR.score(X_test, y_test)) 
print('Matthews: ',matthews_corrcoef(y_test,Predicciones2))
LR.predict(X_test)
filename2 = '/Users/bustacara-js/Documents/IA codigos python/Proyectofinalized_model_LR.sav'
pickle.dump(LR, open(filename2, 'wb'))


for j in ['euclidean','minkowski','manhattan','chebyshev']:
    for i in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        modelKNN = KNeighborsClassifier(n_neighbors=i, metric=j)
        modelKNN.fit(X_train,y_train)
        print("Model KNN ",i, " vecinos:", modelKNN.score(X_test,y_test))
    print("---------",j,"-----------")

modelKNN = KNeighborsClassifier(n_neighbors=5, metric='manhattan',weights='distance')
modelKNN.fit(X_train,y_train)
Predicciones=modelKNN.predict(X_train)
Predicciones2=modelKNN.predict(X_test)
print('Score: ',modelKNN.score(X_test, y_test))
print('Matthews: ',matthews_corrcoef(y_test,Predicciones2))
filename = 'finalized_modelKNN.sav'
pickle.dump(modelKNN, open(filename, 'wb'))

modelSVMlinear = SVC(kernel='sigmoid',decision_function_shape='ovr',cache_size=10000)
modelSVMlinear.fit(X_train,y_train)
print ("SVM sigmoide: ",modelSVMlinear.score(X_test,y_test))
filename = 'finalized_model_SVM_sigmoid.sav'
pickle.dump(modelSVMlinear, open(filename, 'wb'))

ANNModel=MLPClassifier(hidden_layer_sizes= 5000,random_state=1,activation='relu',solver='adam',batch_size=1000).fit(X_train, y_train)
print ("ANN Model 35 : ",ANNModel.score(X_test,y_test))
filename = 'finalized_model_ANN.sav'
pickle.dump(ANNModel, open(filename, 'wb'))
