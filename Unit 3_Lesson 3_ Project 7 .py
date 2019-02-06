#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


pop1 = np.random.binomial(10, 0.2, 10000)
pop2 = np.random.binomial (10, 0.5, 10000)

#Sample Size 100 (mean & std) 

sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)
plt.hist(sample1, label='sample 1') #Why is alpha needed here? 
plt.hist(sample2, label="sample 2")
plt.legend(loc='upper right')
plt.show()
print (sample1.mean())
print (sample1.std())
print (sample2.mean())
print (sample2.std())


# In[11]:


#Question 1 - INCREASE 
#If we increase the size of the sample size I expect that the mean will stay pretty much the same 
sample3 = np.random.choice(pop1, 1000, replace=True)
sample4 = np.random.choice(pop2, 1000, replace=True)
plt.hist(sample3, label='sample 3')
plt.hist(sample4, label="sample 4")
plt.legend(loc='upper right')
plt.show()
print (sample3.mean())
print (sample3.std())
print (sample4.mean())
print (sample4.std())


# In[12]:


#Question 1 - DECREASE 
#If we decrease the size of the sample size I expect that the mean will stay pretty much the same 
sample4 = np.random.choice(pop1, 20, replace=True)
sample5 = np.random.choice(pop2, 20, replace=True)
plt.hist(sample4, label='sample 4')
plt.hist(sample5, label="sample 5")
plt.legend(loc='upper right')
plt.show()
print (sample4.mean())
print (sample4.std())
print (sample5.mean())
print (sample5.std())


# In[15]:


# Question 2 - Change of P Value 

newpop1 = np.random.binomial(10, 0.3, 10000)
pop2 = np.random.binomial (10, 0.5, 10000)

samplea = np.random.choice(newpop1, 100, replace=True)
sampleb = np.random.choice(pop2, 100, replace=True)

p_valuea = samplea.mean()/10
print(p_valuea)

p_valueb = sampleb.mean()/10
print(p_valueb)


# In[18]:


# Question 2 - T Statistic 

from scipy.stats import ttest_ind
print (ttest_ind(samplea, sampleb, equal_var=False)) #why does this print a negative value?
print (ttest_ind(sampleb, samplea, equal_var=False))


# In[19]:


# Question 2 - Repeat with p_value = 0.4 
popa = np.random.binomial(10, 0.4, 10000)
popb = np.random.binomial (10, 0.5, 10000)

samplec = np.random.choice(popa, 100, replace=True)
sampled = np.random.choice(popb, 100, replace=True)

p_valuec = samplec.mean()/10
print(p_valuec)

p_valued = sampled.mean()/10
print(p_valued)


# In[20]:


# Question 2 - p_value = 0.4  

print (ttest_ind(samplec, sampled, equal_var=False)) #why does this print a negative value?
print (ttest_ind(sampled, samplec, equal_var=False))


# In[21]:


# Question 4 Change Distribution

pop_norm = np.random.normal(10, 0.2, 10000)
pop_norm = np.random.normal (10, 0.5, 10000)

#Sample Size 100 (mean & std) 

sample_norm1 = np.random.choice(pop_norm, 100, replace=True)
sample_norm2 = np.random.choice(pop_norm, 100, replace=True)
plt.hist(sample_norm1, label='sample_norm 1')
plt.hist(sample_norm2, label="sample_norm 2")
plt.legend(loc='upper right')
plt.show()
print (sample_norm1.mean())
print (sample_norm1.std())
print (sample_norm2.mean())
print (sample_norm2.std())


# In[ ]:


#Original Values for Comparison 
#1.91
#1.1410083259994206
#4.96
#1.6119553343687907

