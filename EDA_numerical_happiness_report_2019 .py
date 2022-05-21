#!/usr/bin/env python
# coding: utf-8

# In[31]:


# STEP 1
# Import NumPy for numerical calculation, Pandas for handling data and visualization with seaborn and matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize']=(12,6)
sns.set

# STEP 2
#Import the dataset
happiness_data=pd.read_csv('world_happiness_report_2019.csv')

# STEP 3
# We can observe the dataset by checking a few of the rows using the head() function,
# which returns the first five records from the dataset.
happiness_data.head()


# In[32]:


# STEP 4
# Using shape function, we can observe the dimensions of the data
happiness_data.shape


# In[33]:


# There are 9 columns and 156 entries


# In[34]:


# STEP 5
# The info() method shows some of the characteristics of the data such as Column Name
happiness_data.info()


# In[35]:


# Feedback: We can observe that the data which we have doesn’t have any missing values. 


# In[36]:


#STEP 6
# We use describe() function, which shows basic statistical characteristics of each numerical feature
# number of non-missing values (count column), mean, standard deviation, range, median, 0.25, 0.50, 0.75 quartiles.
happiness_data.describe()


# In[37]:


# STEP 7
# We can check for duplicate values in our dataset as the presence of duplicate values will hamper the accuracy of our ML model.
duplicate_Values=happiness_data.duplicated()
print(duplicate_Values.sum())
happiness_data[duplicate_Values]


# In[38]:


# STEP 8
# Handling the outliers in the data, i.e. the extreme values in the data. 
# We can find the outliers in our data using a Boxplot.
happiness_data.boxplot(column=['Score'])
plt.show


# In[39]:


# Feedback: we can observe that our data is now free from outliers


# In[40]:


# STEP 9
# We can find the pairwise correlation between the different columns of the data using the corr() method. 
#(Note – All non-numeric data type column will be ignored.)

happiness_data.corr()


# In[41]:


# STEP 10
# we create a heatmap using Seaborn to visualize the correlation between the different columns of our data
sns.heatmap(happiness_data.corr(),annot=True,cmap='RdYlGn')


# In[42]:


# Feedback: we can see a high correlation between:
# GPD (economy) and Healthy life expenctancy = 0,84
# Score and GPD(economy) = 0,79
# Score and social support = 0,78


# In[43]:


# STEP 11
# using Seaborn, we will visualize the relation between Economy (GDP per Capita)and Happiness Score by using a regression plot


# In[44]:


sns.regplot(x='GDP per capita', y='Score', data=happiness_data)


# In[ ]:


# Feedback: As the Economy increases, the Happiness Score increases as well as denoting a positive relation.

