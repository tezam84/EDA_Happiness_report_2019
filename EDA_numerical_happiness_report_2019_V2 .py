#!/usr/bin/env python
# coding: utf-8

# # STEP 1

# In[73]:


#Import NumPy for numerical calculation, Pandas for handling data and visualization with Seaborn and Matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize']=(12,6)
sns.set


# # STEP 2

# In[74]:


#Import the dataset
happiness_data=pd.read_csv('world_happiness_report_2019.csv')


# # STEP 3

# In[75]:


#We can observe the dataset using the head()function, which returns the first five records from the dataset
happiness_data.head()


# # STEP 4

# In[80]:


# Using shape function, we can observe the dimensions of the data
happiness_data.shape


# In[81]:


# There are 9 columns and 156 entries


# # STEP 5

# In[82]:


# The info() method shows some of the characteristics of the data such as Column Name
happiness_data.info()


# In[83]:


# We don’t have any missing values and numerical variables can have type 'int64' or 'float64'


# # STEP 6

# In[84]:


# We use describe() function, which shows basic statistical characteristics of each numerical feature
# number of non-missing values (count column), mean, standard deviation, range, median, 0.25, 0.50, 0.75 quartiles.
happiness_data.describe()


# # STEP 7

# In[85]:


# We can check for duplicate values in our dataset as the presence of duplicate values will hamper the accuracy of our ML model.
duplicate_Values=happiness_data.duplicated()
print(duplicate_Values.sum())
happiness_data[duplicate_Values]


# # STEP 8

# In[44]:


# Handling the outliers in the data, i.e. the extreme values in the data. 
# We can find the outliers in our data using a Boxplot.
happiness_data.boxplot(column=['Score'])
plt.show


# In[45]:


# Our data is now free from outliers


# # STEP 9

# In[86]:


# We can find the pairwise correlation between the different columns of the data using the corr() method. 
#(All non-numeric data type column will be ignored.)
happiness_data.corr()


# # STEP 10

# In[87]:


# we create a heatmap using Seaborn to visualize the correlation between the different columns of our data
sns.heatmap(happiness_data.corr(),annot=True,cmap='RdYlGn')


# In[48]:


# There is a high correlation between:
# GPD (economy) and Healthy life expenctancy = 0,84
# Score and GPD(economy) = 0,79
# Score and social support = 0,78

# It's the Pearson correlation measure with a coefficient value between -1 and 1


# In[50]:


# We are lucky because not a lot of columns but it's not always the case. We can practice the corr() function
# with selected columns. For example 'generosity', 'overall rank' and 'corruption'
happiness_data[['Generosity', 'Overall rank', 'Perceptions of corruption']].corr()


# In[51]:


# The diagonal elements are always equal to 1. 


# # STEP 11

# In[89]:


# using Seaborn, we visualize the relation between Economy (GDP per Capita) and Happiness Score by using a regression plot
# We use "regplot" which plots the scartterplot plus the fitted regression libe for the data.


# In[90]:


sns.regplot(x='GDP per capita', y='Score', data=happiness_data)


# In[54]:


# As the Economy increases, the Happiness Score increases as well as denoting a positive correlation between 
# those two variables. GPD per capita seems to be a pretty good predictor of happiness score.
# The regression line is almost a perfect diagonal line


# In[63]:


happiness_data[['GDP per capita', 'Score']].corr()

# The correlation is approximately 0.79


# In[ ]:


# Be careful to distinguish correlation (interdependence) vs. causation (cause/effect relationship) 


# # Step 12

# In[91]:


# The P-value is to know the significance of the correlation estimate
# when p-value is
# < 0.001 leads to strong evidence 
# < 0.05 leads to moderate evidence 
# < 0.1 leads to weak evidence 
# > 0.1 leads to no evidence


# In[92]:


from scipy import stats


# In[93]:


pearson_coef, p_value = stats.pearsonr(happiness_data['GDP per capita'], happiness_data['Score'])
print ("The Pearson Correlation coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[94]:


# The linear relationship is strong (app. 0.79) and the correlation is statistically significant

