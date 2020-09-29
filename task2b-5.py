import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets 
import pandas as pd
from sklearn import neighbors
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
import statistics 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
pd.options.mode.chained_assignment = None  # default='warn'


df = pd.read_csv("world.csv")
df2=pd.read_csv("life.csv")
dfnew=pd.merge(df,df2, how='right')

X=((dfnew.drop(['Life expectancy at birth (years)','Country','Time', 'Country Name', 'Country Code','Year'], axis = 1)))
y=dfnew.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, test_size=1/3, random_state=100)

column=[]
median=[]
mean=[]
variance=[]
X_train.iloc[:,:]=X_train.replace('..',  np.nan)
X_test.iloc[:,:]=X_test.replace('..',  np.nan)
median_imputer= SimpleImputer(missing_values=np.nan, strategy='median')
median_imputer=median_imputer.fit(X_train.iloc[:,:])
X_train.iloc[:,:]=median_imputer.transform(X_train.iloc[:,:])
X_test.iloc[:,:]=median_imputer.fit_transform(X_test.iloc[:,:])
scaler = preprocessing.StandardScaler().fit(X_train)
X_trainstd=scaler.transform(X_train)
X_teststd=scaler.transform(X_test)
le=preprocessing.LabelEncoder()
y_train=le.fit_transform(y_train)
le=preprocessing.LabelEncoder()

y_test=le.fit_transform(y_test)

multiply=[]
mms = MinMaxScaler()
mms.fit(X_train)
data_transformed = mms.transform(X_train)
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k/ number of cluster')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')

plt.savefig('task2bgraph1.png',bbox_inches='tight')

start=0
dalam=0
count=0
for i in X_train:
    dalam=0
    
    for y in X_train:  
        if dalam==len(X_train):
            starts=0
            break
        else:
            if dalam>start:
                count+=1
                
                multiply.append(np.multiply( X_train.iloc[:,start],X_train.iloc[:,dalam]))
            dalam+=1           
    start+=1
s=0
for i in X_train:
    multiply.append( X_train.iloc[:,s])
    s=s+1
a={}
for i in range(len(multiply)):
    a[i]=multiply[i]

articles1= pd.DataFrame.from_dict(a)
clusters=KMeans(n_clusters=3, random_state=100).fit(X_train)

print('below is the train set with 210 features')
print(articles1)
articles1['categories']=clusters.labels_
print('below is the train set with 211 features')
print(articles1)
fs = SelectKBest(score_func=mutual_info_classif, k='all')
fs.fit(articles1, y_train)
X_train_fs = fs.transform(articles1)


scores=[]
lo=[]
all_scores=[]
all_value=[]
for i in range(len(fs.scores_)):
    if fs.scores_[i]>0.71:
        scores.append(fs.scores_[i])
        lo.append(i)
    all_scores.append(fs.scores_[i])
    all_value.append(i)

plt.bar([u for u in all_value], all_scores)
plt.xlabel('column location')
plt.ylabel('kmeans')
plt.title('all features')
plt.show()
plt.savefig('task2bgraph2.png',bbox_inches='tight')
    
plt.bar([j for j in lo], scores)
plt.xlabel('column location')
plt.ylabel('kmeans')
plt.title('best 4 feature')

plt.savefig('task2bgraph3.png',bbox_inches='tight')
new_xtrain=(articles1.iloc[:,[149,60,153,117]])
X_testnew=[]


start=0
dalam=0
count=0
for i in X_test:
    dalam=0    
    for y in X_test:        
        if dalam==len(X_test):
            starts=0
            break
        else:
            if dalam>start:              
                count+=1               
                X_testnew.append(np.multiply( X_test.iloc[:,start],X_test.iloc[:,dalam]))
            dalam+=1
            
    start+=1
s=0
for i in X_test:
    X_testnew.append( X_test.iloc[:,s])
    s=s+1
b={}
for i in range(len(X_testnew)):
    b[i]=X_testnew[i]
articles2= pd.DataFrame.from_dict(b)
#print(articles2.describe())
mms = MinMaxScaler()
mms.fit(X_test)
data_transformed = mms.transform(X_test)
Sum_of_squared_distances = []
K = range(1,15)


clusters=KMeans(n_clusters=3, random_state=100).fit(X_test)
articles2['categories']=clusters.labels_
print('below is the test set column with 211 feature')
print(articles2)
new_xtest=(articles2.iloc[:,[149,60,153,117]])
#print(articles2)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    
knn.fit(new_xtrain, y_train) 

y_pred=knn.predict(new_xtest)



print("Accuracy of feature engineering: {}".format(round(accuracy_score(y_test, y_pred),3)))
scaler = preprocessing.StandardScaler().fit(X_train)
#print(X_train)
X_trainpca=scaler.transform(X_train.iloc[:,:])

X_testpca=scaler.transform(X_test.iloc[:,:])

pca = PCA(n_components=4)
principalComponents = pca.fit(X_trainpca)
X_trainpca = pca.transform(X_trainpca)
X_testpca = pca.transform(X_testpca)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_trainpca, y_train) 
y_pred=knn.predict(X_testpca)


per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)
labels=['pc'+ str(x) for x in range(1, len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel("percentage of explained variance")
plt.xlabel("principal component")
plt.title("scree plot")

plt.savefig('task2bgraph4.png',bbox_inches='tight')

print("Accuracy of PCA: {}".format(round(accuracy_score(y_test, y_pred),3)))
x4train=X_train.iloc[:,0:4]
x4test=X_test.iloc[:,0:4]
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

knn.fit(x4train, y_train) 
y_pred=knn.predict(x4test)

print("Accuracy of first four features: {}".format(round(accuracy_score(y_test, y_pred),3)))

print('below is the first 4 column starting from D-G')
print(x4train)