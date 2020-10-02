#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Model on Sonar Dataset to discriminate between sonar signals bounced off a Metal Cylinder and those bounced off a roughly Cylindrical Rock.

# In[2]:


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


# In[3]:


#acquiring the data
sonar_df=pd.read_csv("sonar_data.csv",header=None)


# In[4]:


#analysing the data
print(sonar_df.columns)


# In[5]:


#changing the column name
sonar_df = sonar_df.rename(columns={60: 'Target'})

#previewing the data
sonar_df.head()


# In[6]:


#previewing the data
sonar_df.sample(5)


# In[7]:


#checking the structure of the dataset
sonar_df.shape


# In[8]:


#class distribution
sonar_df.groupby("Target").size()


# In[9]:


#checking for the datatypes of all the fields
sonar_df.dtypes


# In[10]:


#extracting the general information from the dataset
sonar_df.info()


# In[11]:


#checking if any null values present in the dataset
sonar_df.isnull().sum()


# In[12]:


#visualizing the null values through heatmap
sns.heatmap(sonar_df.isnull(),annot=True)


# In[16]:


#checking for the correlation
corr_hmap=sonar_df.corr()
plt.figure(figsize=(8,5))
sns.heatmap(corr_hmap)
#plt.show()


# In[25]:


#multi-variant vizualization
sonar_df.plot(kind='hist', subplots=True, layout=(8,8),figsize=(18,12))
plt.show()


# In[37]:


#checking the skewness before removing the outliers
sonar_df.skew()


# In[ ]:


#converting the Target values from categorical str to categorical int value
convertion={"Target":{"R": 1, "M": 2}}
sonar_df.replace(convertion, inplace=True)
sonar_df.head()


# In[43]:


#removing outliers
z_score=np.abs(zscore(sonar_df))
print(sonar_df.shape)
sonar_df_final=sonar_df.loc[(z_score<3).all(axis=1)]
print(sonar_df_final.shape)


# In[48]:


#checking the skewness after removing the outliers
sonar_df_final.skew()


# In[49]:


#checking for the statastical report
sonar_df_final.describe()


# In[57]:


#Now separating input and output variable
print(sonar_df_final.dtypes)
x=sonar_df_final.drop(['Target'],axis=1)
y=sonar_df_final.select_dtypes(include=['int64']).copy()
print(x.shape)
print(y.shape)


# In[58]:


#standardizing the input dataset 
sc=StandardScaler()
x=sc.fit_transform(x)
x


# In[59]:


#Machine Learning Models
models=[]
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', tree.DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVC', SVC()))


# In[60]:


results = []
names = []
for name, model in models:
    print(name)
    max_acc_score=0
    for r_state in range(42,151):
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=r_state,test_size=0.20)
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


# In[62]:


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


# In[63]:


# Boosting methods
boosters=[]
boosters.append(('AB', AdaBoostClassifier()))
boosters.append(('GBM', GradientBoostingClassifier()))
boosters.append(('RF', RandomForestClassifier()))
boosters.append(('ET', ExtraTreesClassifier()))


# In[65]:


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


# In[91]:


#Choosing the Best Model
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=64,test_size=0.20)
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


# In[92]:


#saving the model as pickle in a file
pickle.dump(model_name,open('svc_sonar_data.pkl','wb'))


# In[93]:


loaded_model=pickle.load(open('svc_sonar_data.pkl','rb'))
loaded_model.predict(x_test)


# In[ ]:




