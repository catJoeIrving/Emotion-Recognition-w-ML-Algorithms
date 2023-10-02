"""main.py: Starter file for assignment on Decision Trees, SVM, and K-NN """

__author__ = "Bryan Tuck"
__version__ = "1.0.0"
__copyright__ = "All rights reserved.  This software  \
                should not be distributed, reproduced, or shared online, without the permission of the author."

# Data Manipulation and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Model Evaluation and Hyperparameter Tuning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.metrics import accuracy_score, precision_score, recall_score

__author__ = "Please enter your name here"
__version__ = "1.1.0"

'''
Github Username: 
PSID:
'''

# Reading of training and testing files
df_train = pd.read_json('emotion_train.json', lines=True)
df_test = pd.read_json('emotion_test.json', lines=True)

# Task 1: Decision Trees

''' Task 1A: Build Decision Tree Models with Varying Depths '''
# Using all attributes, train Decision Tree models with maximum depths of 3, 7, 11, and 15.

''' Task 1B: 5-Fold Cross-Validation for Decision Trees '''
# Perform 5-fold cross-validation on each Decision Tree model. Compute and store the mean accuracy, precision, and recall for each depth. Generate the table.

''' Task 1C: Interpret Decision Tree Depths '''
# Provide explanations on how the tree depth impacts overfitting and underfitting.

''' Task 1D: Interpret Decision Tree Metrics '''
# Explain the significance of differences in accuracy, precision, and recall if any notable differences exist.

# Task 2: K-NN

''' Task 2A: Build k-NN Models with Varying Neighbors '''
# Train K-NN models using 3, 9, 17, and 25 as the numbers of neighbors.

''' Task 2B: 5-Fold Cross-Validation for K-NN '''
# Perform 5-fold cross-validation on each K-NN model. Compute and store the mean accuracy, precision, and recall for each neighbor size. Generate the table.

''' Task 2C: Interpret K-NN Neighbor Sizes '''
# Discuss how the number of neighbors impacts overfitting and underfitting.

''' Task 2D: Interpret K-NN Metrics '''
# Explain any significant differences in accuracy, precision, and recall among the different neighbor sizes if any notable differences exist..

# Task 3: SVM

''' Task 3A: Build SVM Models with Varying Kernel Functions '''
# Train SVM models using linear, polynomial, rbf, and sigmoid kernels. Store each trained model.

''' Task 3B: 5-Fold Cross-Validation for SVM '''
# Perform 5-fold cross-validation on each SVM model. Compute and store the mean accuracy, precision, and recall for each kernel. Generate the table.

''' Task 3C: Interpret SVM Kernel Functions '''
# Discuss the impact of different kernel functions on the performance of the SVM models.

''' Task 3D: Interpret SVM Metrics '''
# Explain any significant differences in accuracy, precision, and recall among the different kernels.

# Task 4: Interpretation and Comparison

''' Task 4: Interpret Tables and Model Comparison '''
# Compare the performance metrics (accuracy, precision, and recall) of the Decision Tree, K-NN, and SVM models. Discuss which model performs better and why.

''' Recommendations for Model Improvement '''
# Provide suggestions on how you might improve each model’s performance.

''' Conclusion '''
# Summarize the key findings and insights from this assignment.