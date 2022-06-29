#!/usr/bin/env python
# coding: utf-8

# # STEP 1

# <div class="alert alert-block alert-info">
# <b>Load dataset and import librairies
# </div>

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize']=(12,6)
sns.set
import warnings
warnings.filterwarnings("ignore")


# # STEP 2

# <div class="alert alert-block alert-info">
# <b>Import the dataset
# </div>

# In[2]:


happiness_data=pd.read_csv('world_happiness_report_2019.csv')


# # STEP 3

# <div class="alert alert-block alert-info">
# <b>We can observe the dataset using the head()function, which returns the first five records from the dataset
# </div>

# In[4]:


happiness_data.head()


# # STEP 4

# <div class="alert alert-block alert-info">
# <b>Using the shape function, we can observe the dimensions of the data
# </div>

# In[3]:


happiness_data.shape


# <font color='green'>There are 9 columns and 156 observations</font>

# # STEP 5

# <div class="alert alert-block alert-info">
# <b>The info() method shows some of the characteristics of the data such as Column Name
# </div>

# In[5]:


happiness_data.info()


# <font color='green'>We don’t have any missing values and numerical variables can have type 'int64' or 'float64'</font>

# # STEP 6

# <div class="alert alert-block alert-info">
# <b>We use describe() function, which shows basic statistical characteristics of each numerical feature
# </div>

# In[6]:


happiness_data.describe()


# # STEP 7

# <div class="alert alert-block alert-info">
# <b>We can check for duplicate values in our dataset
# </div>

# In[8]:


duplicate_Values=happiness_data.duplicated()
print(duplicate_Values.sum())
happiness_data[duplicate_Values]


# <font color='green'>We don’t have any duplicate values </font>

# # STEP 8

# <div class="alert alert-block alert-info">
# <b>Let's have a look to the data type
# </div>

# In[9]:


cat_col = [col for col in happiness_data.columns if happiness_data[col].dtype ==
           'object']  # Categorical columns
num_col = [col for col in happiness_data.columns if happiness_data[col].dtype !=
           'object']  # Numerical columns

print('Categorical Columns: ', cat_col)
print('Numerical Columns: ', num_col)

# As it can be seen no column has been left behind
print(f'Total columns: {len(cat_col) + len(num_col)}')


# <font color='green'>It's mainly numerical variables and only 2 categorical variables </font>

# # STEP 8

# <div class="alert alert-block alert-info">
# <b>Handling the outliers in the data, i.e. the extreme values in the data. <br>We can find the outliers in our data using a Boxplot
# </div>

# In[10]:


happiness_data.boxplot(column=['Score'])
plt.show


# <font color='green'>Our dataframe is now free from outliers</font>

# # STEP 9

# <div class="alert alert-block alert-info">
# <b>We can find the pairwise correlation between the different columns of the data using the corr() method. <br>  All non-numeric data type column will be ignored
# </div>

# In[11]:


happiness_data.corr()


# # STEP 10

# <div class="alert alert-block alert-info">To have a better outlook, we create a heatmap using Seaborn to visualize the correlation 
# </div>

# In[14]:


sns.heatmap(happiness_data.corr(),annot=True,cmap='RdYlGn')


# <font color='green'>There is a high correlation between:<br>GPD (economy) and Healthy life expenctancy = 0,84 <br>
# Score and GPD(economy) = 0,79 <br> Score and social support = 0,78</font>

# #### We can practice the corr() function with selected columns <br>For example 'generosity', 'overall rank' and 'corruption'

# In[50]:


happiness_data[['Generosity', 'Overall rank', 'Perceptions of corruption']].corr()


# # STEP 11

# <div class="alert alert-block alert-info">Using Seaborn, we visualize the relation between Economy (GDP per Capita) and Happiness Score by using a regression plot <br>We use "regplot" which plots the scartterplot plus the fitted regression libe for the data.
# </div>

# In[90]:


sns.regplot(x='GDP per capita', y='Score', data=happiness_data)


# <font color='green'> As the Economy increases, the Happiness Score increases as well as denoting a positive correlation between 
# those two variables <br> GPD per capita seems to be a pretty good predictor of happiness score <br>The regression line is almost a perfect diagonal line</font>

# In[63]:


happiness_data[['GDP per capita', 'Score']].corr()

# The correlation is approximately 0.79


# <font color='green'>Be careful to distinguish correlation (interdependence) vs. causation (cause/effect relationship)  </font>

# # Step 12

# <div class="alert alert-block alert-info">The P-value is to know the significance of the correlation estimate
# when p-value is <br> < 0.001 leads to strong evidence <br> < 0.05 leads to moderate evidence <br> < 0.1 leads to weak evidence <br> > 0.1 leads to no evidence </div>

# In[92]:


from scipy import stats


# In[93]:


pearson_coef, p_value = stats.pearsonr(happiness_data['GDP per capita'], happiness_data['Score'])
print ("The Pearson Correlation coefficient is", pearson_coef, " with a P-value of P =", p_value)


# <font color='green'>The linear relationship is strong (app. 0.79) and the correlation is statistically significant</font>
# 
