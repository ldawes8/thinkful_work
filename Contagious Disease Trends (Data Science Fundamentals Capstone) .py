#!/usr/bin/env python
# coding: utf-8

# In[165]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[166]:


#Part 1 
#Describe your dataset. Describe and explore your dataset in the initial section of your Report. 
#What does your data contain and what is its background? Where does it come from? Why is it interesting or significant? 
#Conduct summary statistics and produce visualizations for the particular variables from the dataset that you will use.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#The below analysis reviews two datasets from Project Tycho. Project Tycho is a research initiative in which global health 
#data was digitized by a team at the University of Pittsburgh to promote collaboration and research to advance human health. 
#There are various pre-compiled and standardized datasets as well as unstructured or “raw” data available for download.
#To date, Project Tycho data has been downloaded 15,721 times by 4,141 registered users. 

#For the purpose of this project I have chosen to use two csv files from a pre-compiled dataset concerning incidence of 
#contagious disease in the United States. One csv file contains infomration related to the number of cases at the state level
#for Hepatitis A while another csv file contained the same information for Measeles. All data was gathered from weekly National 
#Notifiable Disease Survellaillance system reports. Information in the dataset includes: the year, week, state, 
#number of cases, and the incidence of the contagious disease per capita. The incidence per 100,00 was based on historical
#population estimates.            

#This data is interesting because it shows incidence of disease in every state in the United States over a significant period
#of time. The Hepatitis A dataset contained data from 1966 to 2011 while the Measles dataset contained data from 1928 to 2002. 
#These data have been used by investigators at the University of Pittsburgh to estimate the impact of vaccination programs in 
#the United States. The findings from that study were published in the New England Journal of Medicine.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# In[167]:


#Part 1 Visualizations 
path_to_file ='C:/Users/etusk/Documents/measles.csv'
df = pd.read_csv(path_to_file)
df.describe()


# In[168]:


m_bar = df[['state', 'cases']]
m_bar2 = m_bar['state'].value_counts() # Shows total number of cases for each state from 1928 to 2002 
plt.figure(figsize=(15,10))#resize for this particular graph
plt.xlabel('State')
plt.ylabel('Total number of cases')
plt.title('Measles Cases in the US (1928-2002)')
m_bar2.plot(kind='bar')


# In[169]:


path_to_file2 ='C:/Users/etusk/Documents/hepatitis.csv'
df2 = pd.read_csv(path_to_file2)
df2.describe()


# In[170]:


hep_bar = df2[['state','cases']]
hep_bar2 = hep_bar['state'].value_counts() # Shows total number of cases for each state from 1966 to 2011
plt.figure(figsize=(15,10))
plt.xlabel('State')
plt.ylabel('Total number of cases')
plt.title('Hepatitis A in the US (1966-2011)')
hep_bar2.plot(kind='bar')


# In[7]:


#Part 2
#Ask and answer analytic questions. Ask three analytic questions and answer each one with a combination of statistics and 
#visualizations. These analytic questions can focus on individuals behaviors or comparisons of the population.


# In[171]:


#Part 2 - Question 1 
#What state had the most cases of Hepatitis A in a given week? Measles? 
df.loc[df['cases'].idxmax()]


# In[172]:


df2.loc[df2['cases'].idxmax()]


# In[173]:


#Part 2 - Question 2 
#What was the average number of measles cases per weel seen in the state of California? 
california = df.loc[lambda df: df['state'] =='CA', :]
california['cases'].mean()


# In[204]:


#Part 2 - Question 3
#Was there evidence of a Hepatitis A outbreak in week 1 on 1966? The World Health Organization describes and outbreak as "A disease outbreak is the occurrence of cases of disease in excess of what would normally be expected in a defined community, geographical area or season. An outbreak may occur in a restricted geographical area, or may extend over several countries. It may last for a few days or weeks, or for several years."
nineteen_sixty_six = df2.iloc[0:44:,0:5]
x = nineteen_sixty_six['state']
y = nineteen_sixty_six['cases']
plt.figure(figsize=(15,10))
plt.xlabel('State')
plt.ylabel('Total number of cases')
plt.title('Hepatitis A in Week 1 of 1966')
plt.plot(x,y)


# In[ ]:


#Part 3 
#Propose further research. Lastly, make a proposal for a realistic future research project on this dataset that would use some
#data science techniques you'd like to learn in the bootcamp. Just like your earlier questions, your research proposal should
#present one or more clear questions. Then you should describe the techniques you would apply in order to arrive at an answer.

#For a future research project I would like to evaluate the impact new diagnoses procedures and/or pharmaceutical products had on the incidence
#of Hepatitis A and Measles. Futhermore, I would like to utilize population and demographic data to make determiniations about the availability
#and accessibility of healthcare in the US.Techniques I would like to further explore and learn about include 
# utilizing the DateTime functionality to split out the week and year. This will allow me to conduct analyses by year and/or decade. 

