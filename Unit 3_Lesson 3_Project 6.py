#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

#PART 1
#The six distributions I will plot in this exercise include 1. Binomial, 2. Gamma, 3. Poisson,  4. Bernoulli, 
#5. Exponential, and 6. Standard_t


# In[18]:


#1. Binomial Distribution 
binomial = np.random.binomial(20, 0.166, 100)
plt.hist(binomial)
plt.axvline(binomial.mean(), color='g', linestyle ='solid', linewidth=2)
plt.axvline(binomial.mean() + binomial.std(), color='g', linestyle ='dashed', linewidth=2)
plt.axvline(binomial.mean() - binomial.std(), color='g', linestyle ='dashed', linewidth=2)
plt.show()


# In[20]:


#2. Gamma Distribution 
gamma = np.random.gamma(25, 25, 100)
plt.hist(gamma)
plt.axvline(gamma.mean(), color='g', linestyle ='solid', linewidth=2)
plt.axvline(gamma.mean() + gamma.std(), color='g', linestyle ='dashed', linewidth=2)
plt.axvline(gamma.mean() - gamma.std(), color='g', linestyle ='dashed', linewidth=2)
plt.show()


# In[25]:


#3. Poisson Distribution 
poisson = np.random.poisson(10, 100)
plt.hist(poisson)
plt.axvline(poisson.mean(), color='g', linestyle ='solid', linewidth=2)
plt.axvline(poisson.mean() + poisson.std(), color='g', linestyle ='dashed', linewidth=2)
plt.axvline(poisson.mean() - poisson.std(), color='g', linestyle ='dashed', linewidth=2)
plt.show()


# In[28]:


#4. Bernoulli Distribution 
bernoulli = np.random.binomial(1, 0.25, 100)
plt.hist(bernoulli)
plt.axvline(bernoulli.mean(), color='g', linestyle ='solid', linewidth=2)
plt.axvline(bernoulli.mean() + bernoulli.std(), color='g', linestyle ='dashed', linewidth=2)
plt.axvline(bernoulli.mean() - bernoulli.std(), color='g', linestyle ='dashed', linewidth=2)
plt.show()


# In[30]:


#5. Exponential Distribution 
expo = np.random.exponential(15, 100)
plt.hist(expo)
plt.axvline(expo.mean(), color='g', linestyle ='solid', linewidth=2)
plt.axvline(expo.mean() + expo.std(), color='g', linestyle ='dashed', linewidth=2)
plt.axvline(expo.mean() - expo.std(), color='g', linestyle ='dashed', linewidth=2)
plt.show()


# In[31]:


#6. Standard T Distribution 
standt = np.random.standard_t(15, 100)
plt.hist(standt)
plt.axvline(standt.mean(), color='g', linestyle ='solid', linewidth=2)
plt.axvline(standt.mean() + standt.std(), color='g', linestyle ='dashed', linewidth=2)
plt.axvline(standt.mean() - standt.std(), color='g', linestyle ='dashed', linewidth=2)
plt.show()


# In[33]:


#PART 2 
norm1 = np.random.normal(5, 0.5, 10)
norm2 = np.random.normal(10, 1, 10) 
addvar = norm1 + norm2
plt.hist(addvar)
plt.axvline(addvar.mean(), color='g', linestyle ='solid', linewidth=2)
plt.axvline(addvar.mean() + addvar.std(), color='g', linestyle ='dashed', linewidth=2)
plt.axvline(addvar.mean() - addvar.std(), color='g', linestyle ='dashed', linewidth=2)
plt.show()


# In[ ]:




