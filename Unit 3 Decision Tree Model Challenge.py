#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[89]:


#One way to understand how a city government works is by looking at who it employs and how its employees are 
#compensated. This data contains the names, job title, and compensation for San Francisco city employees on an 
#annual basis from 2011 to 2014. (From Kaggle)
#For this challenge I will predict how other income factors relate to how much retirement an invidual 
#recieves on an annual basis. This will help people to assess how income factors for city employees in SF relate to one another.


# In[90]:


## Import the file for Challenge
os.path.isfile('/Users/ldawes/Desktop/emp_subset.csv')
df = pd.read_csv('/Users/ldawes/Desktop/emp_subset.csv') 
df.head()


# In[91]:


# Check what type of data I have.
df.dtypes


# In[92]:


# Inspect variable correlation for feature selection
corr_matrix = df.corr()
print(corr_matrix)
print('Based on the correlation matrix, I will select only features with a positive correlation >0.10 to the target variable [Total Compensation]. Choosing features with positive and negative correlations to the predicted value would lead to noise in the prediction model.')


# In[93]:


# Define feature df
data = df.loc[:, ['Total Benefits', 'Retirement', 'Total Compensation', 'Total Salary', 'Health/Dental']]
# Trimmed correlation matrix
corr_matrix_data = data.corr()
print(corr_matrix_data)


# In[94]:


# Inspect distribution of features
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=False, sharey=False)

plt.plot(figsize=(20,40))

ax1.hist(df['Total Benefits'])
ax1.set_title('Total Benefits')

ax2.hist(df['Retirement'])
ax2.set_title('Retirement')

ax3.hist(df['Total Compensation'])
ax3.set_title('Total Compensation')

ax4.hist(df['Total Salary'])
ax4.set_title('Totlal Salary')

ax5.hist(df['Health/Dental'])
ax5.set_title('Health/Dental')

f.tight_layout()

plt.show()


# In[95]:


# Split data into train/test sets (70/30) (For regression models and KNN)
#x = df.loc[:, ['Total Benefits', 'Total Compensation', 'Total Salary', 'Health/Dental']]
#y = df.loc[:, ['Retirement']]

# Split data using sklearn
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=43)


# In[96]:


#Drop all columns that are not numeric
df = df.drop('Year Type', axis=1)
df = df.drop('Organization Group', axis=1)
df = df.drop('Department Code', axis=1)
df = df.drop('Department', axis=1)
df = df.drop('Union', axis=1)
df = df.drop('Job Family Code', axis=1)
df = df.drop('Job Family', axis=1)
df = df.drop('Job Code', axis=1)
df = df.drop('Job', axis=1)
df = df.dropna()

df.head()


# In[97]:


# Set up variables into different dataframes for model input 
X = df.drop('Retirement', axis=1)
Y = df['Retirement']


# In[107]:


#from sklearn import preprocessing
#from sklearn import utils - How can we use these packages to convert floats to intergers?
X = X.astype(int) # discuss how this affects your model
Y= Y.astype(int)


# In[109]:


# This is the model we'll be using.
from sklearn import tree

# A convenience for displaying visualizations.
from IPython.display import Image

# Packages for rendering our tree.
import pydotplus
import graphviz

# Initialize and train our tree.
decision_tree = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_features=4,
    max_depth=4,
)
decision_tree.fit(X, Y)

# Render our tree.
dot_data = tree.export_graphviz(
    decision_tree, out_file=None,
    feature_names=X.columns,
    class_names=Y,
)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# In[ ]:


# Cross validation 
from sklearn.model_selection import cross_val_score
tree_score = cross_val_score(regressor_tree, X, Y, cv=5)
print('Mean:', tree_score.mean(), '\nVariance:', tree_score.std()*2)


# In[ ]:


## Random Forest 
# Import model
from sklearn import ensemble

# Initialize and cross-validate forest
forest_regression = ensemble.RandomForestClassifier(
    max_depth=2,
    max_features=3
)

forest_score = cross_val_score(forest_regression, X, Y, cv=5)
print('Mean:', forest_score.mean(), '\nVariance:', forest_score.std()*2)


# In[ ]:


## Runtime
import time


# In[ ]:


# Test runtime of Tree
start = time.time()

regressor_tree = tree.DecisionTreeRegressor(
    criterion='mse',
    max_depth=4,
    max_features=3
)

tree_score = cross_val_score(regressor_tree, X, Y, cv=5)

end = time.time()
tree_time = end-start
print(tree_time)


# In[ ]:


# Test runtime of Forest

start = time.time()

forest_regression = ensemble.RandomForestClassifier(
    max_depth=4,
    max_features=3
)

forest_score = cross_val_score(forest_regression, X, Y, cv=5)

end = time.time()
forest_time = end-start
print(forest_time)


# In[ ]:


print('Forest is', round(forest_time/tree_time, 2), 'times slower than the tree')

