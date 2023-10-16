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

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


__author__ = "Please enter your name here"
__version__ = "1.1.0"

'''
Github Username: catJoeIrving
PSID: 1766731
'''

# Reading of training and testing files
df_train = pd.read_json('emotion_train.json', lines=True)
df_test = pd.read_json('emotion_test.json', lines=True)
df_train = df_train.drop('text', axis=1)
df_test = df_test.drop('text', axis=1)

# Extract Features and Target Variable
X_train = df_train.drop('label', axis=1)
y_train = df_train['label']

X_test = df_test.drop('label', axis=1)
y_test = df_test['label']


# Task 1: Decision Trees

''' Task 1A: Build Decision Tree Models with Varying Depths '''
# Using all attributes, train Decision Tree models with maximum depths of 3, 7, 11, and 15.
depths = [3, 7, 11, 15]
results = {'Max Depths': [], 'Accuracy': [], 'Precision': [], 'Recall': []}


for depth in depths:
    # I opted to go with gini for the model as accuracy, precision, and recall scores across all depths were higher
    # than entropy
    Tree = DecisionTreeClassifier(criterion="gini", max_depth=depth)

    ''' Task 1B: 5-Fold Cross-Validation for Decision Trees '''
    # Perform 5-fold cross-validation on each Decision Tree model. Compute and store the mean accuracy, precision,
    # and recall for each depth. Generate the table.

    # 5-fold cross-validation
    accuracies = cross_val_score(Tree, X_train, y_train, cv=5, scoring='accuracy')
    # Added lamba function to prevent zero division errors
    precisions = cross_val_score(Tree, X_train, y_train, cv=5,
                                 scoring=lambda estimator, X, y: precision_score(y, estimator.predict(X),
                                                                                 average='weighted', zero_division=0))
    recalls = cross_val_score(Tree, X_train, y_train, cv=5, scoring='recall_weighted')

    # Calculate means and append the results to the dictionary
    results['Max Depths'].append(depth)
    results['Accuracy'].append(np.mean(accuracies))
    results['Precision'].append(np.mean(precisions))
    results['Recall'].append(np.mean(recalls))

# Display results
df_results = pd.DataFrame(results)
print("Results for Decision Tree Models with Varying Depths")
print(df_results)

''' Task 1C: Interpret Decision Tree Depths '''
# Provide explanations on how the tree depth impacts overfitting and underfitting.
# See included PDF report for explanation

''' Task 1D: Interpret Decision Tree Metrics '''
# Explain the significance of differences in accuracy, precision, and recall if any notable differences exist.
# See included PDF report for explanation


# Task 2: K-NN

''' Task 2A: Build k-NN Models with Varying Neighbors '''
# Train K-NN models using 3, 9, 17, and 25 as the numbers of neighbors.
neighbors = [3, 9, 17, 25]
results_knn = {'Neighbors': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

for n in neighbors:
    # Create KNN model
    # After testing various distance metrics to include euclidean, manhattan, hamming,
    # and chebyshev, I opted to stick with euclidean distance as it produced the highest accuracy, precision,
    # and recall scores
    knn = KNeighborsClassifier(n_neighbors=n, p=2, metric='minkowski')

    ''' Task 2B: 5-Fold Cross-Validation for K-NN '''
    # Perform 5-fold cross-validation on each K-NN model. Compute and store the mean accuracy, precision, and recall
    # for each neighbor size. Generate the table.

    accuracies = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    # Added lambda function to prevent zero division errors
    precisions = cross_val_score(knn, X_train, y_train, cv=5,
                                 scoring=lambda estimator, X, y: precision_score(y, estimator.predict(X),
                                                                                 average='weighted', zero_division=0))
    recalls = cross_val_score(knn, X_train, y_train, cv=5, scoring='recall_weighted')

    # Calculate means and append the results to the dictionary
    results_knn['Neighbors'].append(n)
    results_knn['Accuracy'].append(np.mean(accuracies))
    results_knn['Precision'].append(np.mean(precisions))
    results_knn['Recall'].append(np.mean(recalls))

# Display results
df_results_knn = pd.DataFrame(results_knn)
print("Results for KNN Models with Varying Neighbors")
print(df_results_knn)


''' Task 2C: Interpret K-NN Neighbor Sizes '''
# Discuss how the number of neighbors impacts overfitting and underfitting.
# See included PDF report for explanation

''' Task 2D: Interpret K-NN Metrics '''
# Explain any significant differences in accuracy, precision, and recall among the different neighbor sizes if any notable differences exist..
# See included PDF report for explanation


# Task 3: SVM

''' Task 3A: Build SVM Models with Varying Kernel Functions '''
# Train SVM models using linear, polynomial, rbf, and sigmoid kernels. Store each trained model.
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results_svm = {'Kernel': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

# Set common parameters

# I tested several values for C to include 0.1, 0.5, 1, 3 , 5, and 10. I opted to go with 0.5
# as it produced the highest accuracy, precision, and recall scores for the linear kernel function. At a C of 3 rbf
# and sigmoid kernels also had competitive scores but overall, the linear kernel at 0.5 had the best numbers across
# all 3 metrics.
C_value = 0.5
gamma_value = 0.2

for kernel in kernels:
    # Initialize SVM model with common parameters for each kernel type.
    svm = SVC(kernel=kernel, C=C_value, gamma=gamma_value, random_state=0)

    ''' Task 3B: 5-Fold Cross-Validation for SVM '''
    # Perform 5-fold cross-validation on each SVM model. Compute and store the mean accuracy, precision, and recall
    # for each kernel. Generate the table.

    accuracies = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
    # Added lambda function to prevent zero division errors
    precisions = cross_val_score(svm, X_train, y_train, cv=5,
                                 scoring=lambda estimator, X, y: precision_score(y, estimator.predict(X),
                                                                                 average='weighted', zero_division=0))
    recalls = cross_val_score(svm, X_train, y_train, cv=5, scoring='recall_weighted')

    # Append the results to the dictionary
    results_svm['Kernel'].append(kernel)
    results_svm['Accuracy'].append(np.mean(accuracies))
    results_svm['Precision'].append(np.mean(precisions))
    results_svm['Recall'].append(np.mean(recalls))

# Display results
df_results_svm = pd.DataFrame(results_svm)
print("Results for SVM Models with Varying Kernel Functions")
print(df_results_svm)

''' Task 3C: Interpret SVM Kernel Functions '''
# Discuss the impact of different kernel functions on the performance of the SVM models.
# See included PDF report for explanation

''' Task 3D: Interpret SVM Metrics '''
# Explain any significant differences in accuracy, precision, and recall among the different kernels.
# See included PDF report for explanation

# Task 4: Interpretation and Comparison

''' Task 4: Interpret Tables and Model Comparison '''
# Compare the performance metrics (accuracy, precision, and recall) of the Decision Tree, K-NN, and SVM models. Discuss which model performs better and why.
# See included PDF report for explanation

''' Recommendations for Model Improvement '''
# Provide suggestions on how you might improve each model’s performance.
# See included PDF report for explanation

''' Conclusion '''
# Summarize the key findings and insights from this assignment.
# See included PDF report for explanation