import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets 
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
import statistics 
pd.options.mode.chained_assignment = None  # default='warn'
def neighbour(k):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    
    knn.fit(X_train, y_train) 
    y_pred=knn.predict(X_test)
    
    return (accuracy_score(y_test, y_pred))
def tree_classifier(depth):
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    dt.fit(X_train, y_train)
    y_pred=dt.predict(X_test)
    return (accuracy_score(y_test, y_pred))

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

for i in X_train:
    column.append(i)
    counter = X_train[i]
    numbers=[float(j) for j in X_train[i] if j!='..' ]    
    median.append(' '+ str(round(statistics.median(numbers),3)))
    mean.append(' '+ str(round(statistics.mean(numbers),3)))
    variance.append(' '+ str(round(statistics.variance(numbers),3)))

data={'feature':column,' median':median,' mean':mean ,' variance': variance}
articles1= pd.DataFrame.from_dict(data)
articles1.to_csv('task2a.csv', index = False)    

X_train.iloc[:,:]=X_train.replace('..',  np.nan)
X_test=X_test.replace('..',  np.nan)
median_imputer= SimpleImputer(missing_values=np.nan, strategy='median')
median_imputer=median_imputer.fit(X_train.iloc[:,:])
X_train.iloc[:,:]=median_imputer.transform(X_train.iloc[:,:])
X_test=median_imputer.fit_transform(X_test)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train.iloc[:,:]=scaler.transform(X_train.iloc[:,:])

X_test=scaler.transform(X_test)
le=preprocessing.LabelEncoder()
y_train=le.fit_transform(y_train)
le=preprocessing.LabelEncoder()

y_test=le.fit_transform(y_test)






print("Accuracy of decision tree: {}".format(round(tree_classifier(4),3)))
print("Accuracy of k-nn (k=5): {}".format(round(neighbour(5),3)))
print("Accuracy of k-nn (k=10): {}".format( round(neighbour(10),3)))




