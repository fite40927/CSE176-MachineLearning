#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn


# In[23]:


# Load the dataset from the text file
data = np.loadtxt('Desktop/YearPredictionMSD.txt', delimiter=',')


# In[27]:


mask = data[:,0] >= 2000
A = data[np.invert(mask)]
B = data[mask]
B_prime , _ = train_test_split(B, test_size=0.5)
data = np.vstack((A,B_prime))

# Split the dataset into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2)
test_data, val_data = train_test_split(test_data, test_size=0.5)

# Separate the features and target variable in the train set
X_train = train_data[:, 1:]  # Exclude the first column (year)
y_train = train_data[:, 0]  # First column is the year

# Separate the features and target variable in the test set
X_test = test_data[:, 1:]
y_test = test_data[:, 0]

X_val = val_data[:,1:]
y_val = val_data[:,0]

print(np.mean(y_train),np.mean(y_val),np.mean(y_test))
plt.hist(y_train)


# In[28]:


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.01,0.1, 1],
    'min_samples_leaf': [2],
    'l2_regularization': [0,0.05],
    'max_iter': [1000]
}


# In[29]:


# Create the HistGradientBoostingRegressor model
model = HistGradientBoostingRegressor()

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(model, param_grid, scoring='r2')
grid_search.fit(X_val, y_val)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_


# In[30]:


# Train the best model
best_model.fit(X_train, y_train)


# In[58]:


# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Print the best hyperparameters and R-squared score
print("Best Hyperparameters:", best_params)
print("Best R-squared score:", r2)

# Create scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')

# Save the scatter plot as an image file
plt.savefig('scatter_plot.png')

