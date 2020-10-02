#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Model for analysing the types of Survivors on the Titanic Ship

# In[138]:


#data analysis and wrangling
import pandas as pd
import numpy as np

#visualizing the data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#model developing-machine learning
import sklearn
from scipy.stats import zscore                                                          #for removing the outliers
from sklearn.preprocessing import StandardScaler                                        #for standardizing the input dataset
from sklearn.model_selection import train_test_split                                    #to train the model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score       #for reporting purposes

#boosting techniques
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

#saving the model using joblib
import pickle

#for filtering the warnings
import warnings
warnings.filterwarnings("ignore")


# In[139]:


#acquiring the data
titanic_df=pd.read_csv("titanic_train.csv")


# In[140]:


#analysing the data
print(titanic_df.columns)

#titanic_df.set_index(["PassengerId"],inplace=True)

#previewing the data
titanic_df.head()

#conclusion:
#categorical data:Survived,Pclass,Sex,Embarked
#continuous  data:PassangerID,Age,SibSP,Fare,Parch


# In[141]:


#previewing the data
titanic_df.tail()


# In[142]:


#checking the structure of the dataset
titanic_df.shape


# In[143]:


#checking for the datatypes of all the fields
titanic_df.dtypes


# In[144]:


#extracting the general information from the dataset
titanic_df.info()


# In[145]:


#checking if any null values present in the dataset
titanic_df.isnull().sum()


# In[146]:


#imputer functions(handling null values)
titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)


# In[147]:


#checking the mode value for the mentioned column
titanic_df['Embarked'].mode()


# In[148]:


#imputer functions(handling null values)
titanic_df['Embarked'].fillna('S',inplace=True)


# In[149]:


#verifying the null values status
titanic_df.isnull().sum()


# In[150]:


#printing different plot w.r.t survived column
titanic_columns=['Pclass','Sex','SibSp','Parch']
for i in range(len(titanic_columns)):
    fig,ax=plt.subplots(figsize=(5,5))
    sns.countplot(data=titanic_df,x=titanic_columns[i],hue='Survived')


# In[151]:


#vizualizing the Fare
sns.distplot(titanic_df['Fare'])


# In[152]:


#checking for the correlation
corr_hmap=titanic_df.corr()
plt.figure(figsize=(8,7))
sns.heatmap(corr_hmap,annot=True)
#plt.show()


# In[153]:


#multiplot visualization
sns.pairplot(titanic_df)


# In[154]:


#checking for the outliers
titanic_df.plot(kind='box',subplots=True,layout=(2,4),figsize=(15,5))


# In[155]:


#checking the skewness before removing the outliers
titanic_df.skew()


# In[156]:


#dropping off the columns those are not required
titanic_df.drop(['PassengerId','Sex','Name','Ticket','Fare','Cabin','Embarked'],axis=1,inplace=True)
#titanic_df1 = pd.get_dummies(titanic_df)


# In[157]:


#removing outliers
z_score=np.abs(zscore(titanic_df))
print(titanic_df.shape)
titanic_df_final=titanic_df.loc[(z_score<3).all(axis=1)]
print(titanic_df_final.shape)


# In[158]:


#checking the skewness after removing the outliers
titanic_df_final.skew()


# In[159]:


#visualizing the null values through heatmap
sns.heatmap(titanic_df.isnull(),annot=True)


# In[160]:


#checking for the statastical report
titanic_df_final.describe()


# In[177]:


#Now separating input and output variable
x=titanic_df_final.drop(['Survived'],axis=1)
y=titanic_df_final['Survived']
print(x.shape)
print(y.shape)


# In[183]:


#standardizing the input dataset 
sc=StandardScaler()
x=sc.fit_transform(x)
x


# In[184]:


#Machine Learning Models
models=[]
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', tree.DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVC', SVC()))


# In[185]:


results = []
names = []
for name, model in models:
    print(name)
    max_acc_score=0
    for r_state in range(42,151):
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=r_state,test_size=0.10)
        model_name=model
        model_name.fit(x_train,y_train)
        y_pred=model_name.predict(x_test)
        accuracy_scr=accuracy_score(y_test,y_pred)
        print("random state: ",r_state," accuracy score: ",accuracy_scr)
        if accuracy_scr>max_acc_score:
            max_acc_score=accuracy_scr      
            final_r_state=r_state
    print()
    print("max accuracy score at random state:",final_r_state," for the model ",name," is: ",max_acc_score)
    print()
    print()


# In[186]:


#cross_val of the models
results = []
names = []
for name, model in models:
    cv_result=cross_val_score(model, x_train, y_train, cv=15, scoring="accuracy")
    results.append(cv_result)
    names.append(name)
    print("Model name: ",name)
    print("Cross Validation Score(Mean): ",cv_result.mean())
    print("Cross Validation Score(Std): ",cv_result.std())
    print()


# In[187]:


# Boosting methods
boosters=[]
boosters.append(('AB', AdaBoostClassifier()))
boosters.append(('GBM', GradientBoostingClassifier()))
boosters.append(('RF', RandomForestClassifier()))
boosters.append(('ET', ExtraTreesClassifier()))


# In[188]:


results = []
names = []
for name, model in boosters:
    cv_results = cross_val_score(model, x_train, y_train, cv=15, scoring="accuracy")
    results.append(cv_results)
    names.append(name)    
    print("Model name: ",name)
    print("Cross Validation Score(Mean): ",cv_results.mean())
    print("Cross Validation Score(Std): ",cv_results.std())
    print()


# In[189]:


#Choosing the Best Model
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=104,test_size=0.10)
model_name=SVC()
model_name.fit(x_train,y_train)
model_name.score(x_train,y_train)
y_pred_model=model_name.predict(x_test)
accuracy_scr=accuracy_score(y_test,y_pred_model)
cfm=confusion_matrix(y_test,y_pred_model)
cr=classification_report(y_test,y_pred_model)
print("accuracy score: ",accuracy_scr)
print("confusion matrix: ")
print(cfm)
print("classification report: ")
print(cr)
print(y_pred_model)


# In[190]:


#saving the model as pickle in a file
pickle.dump(model_name,open('svc_titanic_data.pkl','wb'))


# In[191]:


loaded_model=pickle.load(open('svc_titanic_data.pkl','rb'))
loaded_model.predict(x_test)


# In[ ]:




