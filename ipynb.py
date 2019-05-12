#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import scipy
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[83]:


pwd


# In[84]:


os.path.isfile('/Users/ldawes/Desktop/emp_subset.csv')


# In[85]:


data = pd.read_csv('/Users/ldawes/Desktop/emp_subset.csv') 
data.head()


# In[86]:


data.describe()


# In[87]:


data.dtypes


# In[68]:


def check_nans(data):
    for column in data.columns.unique():
        print(column,":",data[column].isnull().sum()/data.shape[0]*100 ,"% is Nan")

check_nans(data)


# In[89]:


# Drop NA's from Union Code Column
data = data.dropna()


# In[69]:


# creating a dict file  
#dictionary = {'male': 1,'female': 2} 
  
# traversing through dataframe 
# Gender column and writing 
# values where key matches 
#data.Gender = [gender[item] for item in data.Gender] 
#print(data)


# In[90]:


from sklearn import linear_model
import seaborn as sns
from IPython.display import display
import math

pd.options.display.float_format = '{:.3f}'.format

# Suppress annoying harmless error.
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# In[91]:


data2 = data.round() 
data2.head()
#decimals = 2
#data['Salaries'] = data['Salaries'].apply(lambda x: round(x, decimals))
#print(data['Salaries'])


# In[109]:


#Split data first so that model does not test on the training set. X_train can be used to run first model
from sklearn.model_selection import train_test_split
# Use train_test_split to create the necessary training and test groups
X_train, X_test, y_train, y_test = train_test_split(data2['Health/Dental'], data2['Total Compensation'], test_size=0.2, random_state=20)
print('With 20% Holdout: ' + str(regr.fit(X_train, y_train).score(X_test, y_test).reshape(-1,1)))
print('Testing on Sample: ' + str(regr.fit(actual2, predicted2).score(actual2, predicted2).reshape(-1,1)))


# In[92]:


# Instantiate and fit our model.
# do not run model on entire df - use a subset - randomly pick up a set 80% (training)
regr = linear_model.LinearRegression()
Y = data['Total Compensation'].values.reshape(-1, 1)
X = data[['Health/Dental']] # check regression between one independent var and one of the dependent var - do that for all three; write a function that will pass through columns  
regr.fit(X, Y) #include columns and create a loops to calculate the R squared values 

# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')

#65% of the vairation of the total comp column is getting explained by the two columns union code and health/Dental


# In[93]:


#Extract predicted values.
predicted = regr.predict(X).ravel()
actual = Y.ravel()

predicted2 = predicted.astype(float)
actual2 = actual.astype(float)

#Calculate the error, also called the residual.
residual = actual2 - predicted2 # residual is showing negative so actual is smaller than predicted value - need to look for outliers
# Next plot a histogram/scatterplot of the actual to see outliers and re run


# Error histogram
plt.hist(residual)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()

# Error scatter plot
plt.scatter(predicted2, residual)
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.axhline(y=0)
plt.title('Residual vs. Predicted')
plt.show()


# In[99]:


#Holdout Groups # This code should be executed beforehand so that model does not test on the training set. X_train can be used to run first model
from sklearn.model_selection import train_test_split
# Use train_test_split to create the necessary training and test groups
X_train, X_test, y_train, y_test = train_test_split(actual2, predicted2, test_size=0.2, random_state=20)
print('With 20% Holdout: ' + str(regr.fit(X_train, y_train).score(X_test, y_test)).reshape(-1,1))
print('Testing on Sample: ' + str(regr.fit(actual2, predicted2).score(actual2, predicted2).reshape(-1,1)))


# In[100]:


model = regr.fit(X, Y)

from sklearn.model_selection import cross_val_score
cross_val_score(model, X, Y, cv=5)


# In[101]:


#Start of Nearest Neighboor 
from sklearn import neighbors

# Build our model.
knn = neighbors.KNeighborsRegressor(n_neighbors=10)
#X = pd.DataFrame(music.loudness) Edit out for data above
#Y = music.bpm
knn.fit(X, Y)

# Set up our prediction line.
T = np.arange(0, 50, 0.1)[:, np.newaxis]

# Trailing underscores are a common convention for a prediction.
Y_ = knn.predict(T)

plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('K=10, Unweighted')
plt.show()


# In[103]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(knn, X, Y, cv=5)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
score_w = cross_val_score(knn_w, X, Y, cv=5)
print("Weighted Accuracy: %0.2f (+/- %0.2f)" % (score_w.mean(), score_w.std() * 2))


# In[ ]:




