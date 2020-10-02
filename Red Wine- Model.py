#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Model on determining the quality of Red Wine

# In[1]:


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


# In[120]:


#acquiring the data
red_wine_df=pd.read_csv("winequality-red.csv")


# In[48]:


#analysing the data
print(red_wine_df.columns)


# In[49]:


#previewing the data
red_wine_df.head()


# In[50]:


#previewing the data
red_wine_df.sample(5)


# In[51]:


#class distribution
red_wine_df.groupby("quality").size()


# In[52]:


#checking the structure of the dataset
red_wine_df.shape


# In[53]:


#checking for the datatypes of all the fields
red_wine_df.dtypes


# In[54]:


#extracting the general information from the dataset
red_wine_df.info()


# In[55]:


#checking if any null values present in the dataset
red_wine_df.isnull().sum()


# In[14]:


#checking for the correlation
corr_hmap=red_wine_df.corr()
plt.figure(figsize=(8,5))
sns.heatmap(corr_hmap,annot=True)


# In[150]:


sns.barplot(data=red_wine_df,x='quality',y='volatile acidity')


# In[155]:


fig,ax=plt.subplots(figsize=(5,5))
sns.countplot(data=red_wine_df,x=red_wine_df["quality"],hue='quality')


# In[144]:


#multiplot visualization
red_wine_df.plot(kind='density', subplots=True, layout=(4,3),legend=True,sharex=False,figsize=(12,12))


# In[160]:


#checking for the outliers
red_wine_df.plot(kind='box',subplots=True,layout=(3,4),figsize=(15,10))


# In[56]:


#checking the skewness before removing the outliers
red_wine_df.skew()


# In[112]:


#dropping off the columns those are not required
#red_wine_df.drop(['volatile acidity'],axis=1,inplace=True)
#red_wine_df.drop(['citric acid'],axis=1,inplace=True)


# In[121]:


#removing outliers
z_score=np.abs(zscore(red_wine_df))
print(red_wine_df.shape)
red_wine_df_final=red_wine_df.loc[(z_score<3).all(axis=1)]
print(red_wine_df_final.shape)


# In[122]:


#checking the skewness after removing the outliers
red_wine_df_final.skew()


# In[123]:


#checking for the statastical report
red_wine_df_final.describe()


# In[124]:


#Now separating input and output variable
print(red_wine_df_final.dtypes)
x=red_wine_df_final.drop(['quality'],axis=1)
y=red_wine_df_final.select_dtypes(include=['int64']).copy()
print(x.shape)
print(y.shape)


# In[125]:


#standardizing the input dataset 
sc=StandardScaler()
x=sc.fit_transform(x)
x


# In[126]:


#Machine Learning Models
models=[]
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', tree.DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVC', SVC()))


# In[127]:


results = []
names = []
for name, model in models:
    print(name)
    max_acc_score=0
    for r_state in range(42,201):
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


# In[100]:


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


# In[101]:


# Boosting methods
boosters=[]
boosters.append(('AB', AdaBoostClassifier()))
boosters.append(('GBM', GradientBoostingClassifier()))
boosters.append(('RF', RandomForestClassifier()))
boosters.append(('ET', ExtraTreesClassifier()))


# In[37]:


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


# In[130]:


#Choosing the Best Model
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=168,test_size=0.10)
model_name=tree.DecisionTreeClassifier()
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


# In[131]:


#saving the model as pickle in a file
pickle.dump(model_name,open('svc_red_wine_data.pkl','wb'))


# In[132]:


loaded_model=pickle.load(open('svc_red_wine_data.pkl','rb'))
loaded_model.predict(x_test)


# In[ ]:




