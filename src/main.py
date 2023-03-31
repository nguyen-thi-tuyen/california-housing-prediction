#!/usr/bin/env python
# coding: utf-8

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1z0U84GYqhbWWpCenFajh8_8XFRGyOc3U" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h1> Live Coding 1: Simple Prediction Notebook</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center><h4>Propulsion Academy, 2022</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm;color:#303030">
#         <p><strong>Goal:</strong> Revision on a simple prediction model using Scikit Learn</p>
#         <strong> Outline:</strong>
#         <a id='P0' name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href='#SU'>Set up</a></li>
#             <li> <a style="color:#303030" href='#P1'>Data Exploration and Cleaning</a></li>
#             <li> <a style="color:#303030" href='#P2'>Modeling</a></li>
#             <li> <a style="color:#303030" href='#P3'>Model Evaluation</a></li>
#             <li> <a style="color:#303030" href='#CL'>Conclusion</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Notebook Layout, Data Cleaning, Modelling and Model Evaluation
#     </div>
# </div>
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/ds-materials/07_MLEngineering/index.html" title="momentum"> Module 7, Machine Learning Engineering </a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/ds-materials/07_MLEngineering/day1/index.html" title="momentum">Day 1, Data Science Project Development </a>|
#         <a style="color:#00BAE5" href="https://drive.google.com/file/d/1SOCQu9Gv3jNNXxvJSszBC3fYNsM0df2F/view?usp=sharing" title="momentum"> Live Coding 1, Simple Prediction Notebook</a>
# </strong></nav>

# <a id='I' name="I"></a>
# ## [Introduction](#P0)

# This Notebook is a minimal example of a regression experiment on the California Housing Dataset. It is inspired from the exercise from day 2 of the [Machine Learning Module](https://monolith.propulsion-home.ch/backend/api/momentum/materials/ds-materials/04_MachineLearning/day2/pages/07_exercises.html).
# 
# The modeling and data cleaning are very simple, so that you can focus on MLOps concepts

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# ### Packages

# In[1]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline


# ### User Dependent Variables

# In[2]:


data_path = "../data/raw/california_housing_0.csv"


# <a id='P1'></a>
# ## [Data Preparation](#P0)

# In[3]:


data = pd.read_csv(data_path)


# ### Data Exploration

# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.isna().sum()


# ### Data Cleaning

# In[7]:


data = data.drop(columns="total_bedrooms")


# ### Train-Test Split

# In[8]:


data_train, data_test = train_test_split(data, test_size=0.33, random_state=0)


# In[9]:


data_train.shape, data_test.shape


# In[10]:


# # Select X and y values (predictor and outcome)
X_train = data_train.drop(columns="median_house_value")
y_train = data_train["median_house_value"]


# In[11]:


X_test = data_test.drop(columns="median_house_value")
y_test = data_test["median_house_value"]


# In[12]:


X_train.shape, X_test.shape


# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# ### Pipeline Definition

# In[13]:


sc = StandardScaler()
lin_reg = LinearRegression()
pipeline_mlr = Pipeline([("data_scaling", sc), ("estimator", lin_reg)])


# ### Model Fit

# In[14]:


pipeline_mlr.fit(X_train, y_train)


# <a id='P3' name="P3"></a>
# ## [Model Evaluation](#P0)

# In[15]:


predictions_mlr = pipeline_mlr.predict(X_test)


# In[16]:


# Test score
pipeline_mlr.score(X_test, y_test)


# In[17]:


print("MAE", metrics.mean_absolute_error(y_test, predictions_mlr))
print("MSE", metrics.mean_squared_error(y_test, predictions_mlr))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, predictions_mlr)))
print("Explained Var Score", metrics.explained_variance_score(y_test, predictions_mlr))


# <a id='CL'></a>
# ## [Conclusion](#P0)
# 
# This Notebook Shows a simple modelling experiment. We will use this base for building our machine Learning Project.

# <div style="border-top:0.1cm solid #EF475B"></div>
#     <strong><a href='#Q0'><div style="text-align: right"> <h3>End of this Notebook.</h3></div></a></strong>
