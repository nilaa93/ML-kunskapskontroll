#!/usr/bin/env python
# coding: utf-8

# # Kunskapskontroll del II
# Namn: Nil Abukar

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[2]:


mnist = fetch_openml('mnist_784', version = 1, cache = True, as_frame = False)
print(mnist.DESCR)


# # Splitting the data in train and test and flatten/reshape it

# In[3]:


# Extract data and labels
X = mnist ['data']
y = mnist ['target'].astype(np.uint8)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Normalize the pixel values to range between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the input data to 4D tensor shape. The numbers represent the follwing -1 is the placeholder that allows Numpy to automatically infer the size of the first dimension. 28 represents the width dimension and the other 28 represnts hight dimension. 1 represents the number of channels in the data. 
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape (-1, 28, 28, 1)


# In[6]:


# In order to train the model there needs to be a flatten/reshape data to  2D

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# # Train, test and evaluate Random Forest Classifier (RFC) and Support Vector Machine Classifier (SVMC)

# In[7]:


# Importing RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier


# In[8]:


# Import Support Vector Machine 
from sklearn.svm import SVC


# In[9]:


# Creating a RandomForestClassifier 'RF'
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the RF classifier and cross validate
rf_classifier.fit(X_train_flat, y_train)
scores_rf = cross_validate(rf_classifier, X_train_flat, y_train, cv=3, scoring='accuracy')['test_score']

# Test the classifier
test_accuracy = rf_classifier.score(X_test_flat, y_test)
print("Test Accuracy for RF:", test_accuracy)


# In[10]:


# Create a SVM classifer 
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Train the classifier and cross validate
svm_classifier.fit(X_train_flat, y_train)
scores_svm = cross_validate(svm_classifier, X_train_flat, y_train, cv=3, scoring='accuracy')['test_score']

# Test the classifier
test_accuracy_svm = svm_classifier.score(X_test_flat, y_test)
print(f'Test accuracy for SVM: {test_accuracy_svm}')                           


# In[11]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[17]:


# Create a DT classifer 
dt_classifier = DecisionTreeClassifier(random_state=42)

#Train and cross validate
dt_classifier.fit(X_train_flat, y_train)
scores_dt = cross_validate(dt_classifier, X_train_flat, y_train, cv=3, scoring='accuracy')['test_score']

y_pred_dt = dt_classifier.predict(X_test_flat)

test_accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Test Accuracy for Decision Tree:", test_accuracy_dt)


# In[18]:


from sklearn.tree import plot_tree


# In[19]:


plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, max_depth=3, feature_names=mnist.feature_names)
plt.show()


# In[20]:


test_accuracy_rf = 0.9675  
test_accuracy_svm = 0.9351428571428572
# test_accuracy_dt = 0.8695714285714286

# Define classifiers and their test accuracies
classifiers = ['Random Forest', 'SVM']
test_accuracies = [test_accuracy_rf, test_accuracy_svm]

# Create chart plot
plt.figure(figsize=(8, 6))
plt.plot(classifiers, marker='o', linestyle='-', color='blue')
plt.plot(test_accuracies, marker='o', linestyle='-', color='orange')
plt.xlabel('Classifier')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Classifiers')
plt.ylim(0, 1)
plt.show()


# # Making predictions with RFC and SVMC

# In[21]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# In[22]:


# Make predictions using the trained Random Forest classifier
y_pred_rf = rf_classifier.predict(X_test_flat)
print(f'Predicitions for Random Forest Classifier: {y_pred_rf}')

correct_predictions = (y_pred_rf == y_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')  
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')  
f1_rf = f1_score(y_test, y_pred_rf, average='weighted') 
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print("Accuracy:", accuracy_score)
print("Precision:", precision_score)
print("Recall:", recall_score)
print("F1-score:", f1_score)


# In[24]:


# Make predictions using the trained SVM classifier
y_pred_svm = svm_classifier.predict(X_test_flat)
print(f'Predicitions for Support Vector Machine Classifier: {y_pred_svm}')

correct_predictions = (y_pred_svm == y_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')  
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')  
f1_svm = f1_score(y_test, y_pred_svm, average='weighted') 
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

print("Accuracy:", accuracy_score)
print("Precision:", precision_score)
print("Recall:", recall_score)
print("F1-score:", f1_score)


# # An analysis with visualization by using confusion matrix

# In[25]:


# Plot confusion matrix for RF
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[26]:


# Plot confusion matrix for SVM
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title('Confusion Matrix for SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[27]:


# A confusion matrix for RF using TP, FP, TN and FN
TP_rf = conf_matrix_rf[1, 1]
FP_rf = conf_matrix_rf[0, 1]
TN_rf = conf_matrix_rf[0, 0]
FN_rf = conf_matrix_rf[1, 0]

# Creating a bar plot
labels = ['True Positive (TP)', 'False Negative (FN)', 'False Positive (FP)', 'True Negative (TN)']
values_rf = [TP_rf, FN_rf, FP_rf, TN_rf]

plt.figure(figsize=(8, 6))
plt.bar(labels, values_rf, color=['green', 'red', 'orange', 'blue'])
plt.xlabel('Predicted Label')
plt.ylabel('Count')
plt.title('True Positives (TP), False Negatives (FN), False Positives (FP), and True Negatives (TN) for Random Forest')
plt.show()


# In[28]:


# A confusion matrix for SVM using TP, FP, TN and FN
TP_svm = conf_matrix_svm[1, 1]
FP_svm = conf_matrix_svm[0, 1]
TN_svm = conf_matrix_svm[0, 0]
FN_svm = conf_matrix_svm[1, 0]

# Create a bar plot for SVM
values_svm = [TP_svm, FN_svm, FP_svm, TN_svm]

plt.figure(figsize=(8, 6))
plt.bar(labels, values_svm, color=['red', 'yellow', 'green', 'purple'])
plt.xlabel('Predicted Label')
plt.ylabel('Count')
plt.title('True Positives (TP), False Negatives (FN), False Positives (FP), and True Negatives (TN) for SVM')
plt.show()


# In[29]:


random_indices = np.random.choice(len(X_test), size=10, replace=False)

for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f"True: {y_test[idx]}, RF Predicted: {y_pred_rf[idx]}, SVM Predicted: {y_pred_svm[idx]}", fontsize=4.5)

plt.tight_layout()
plt.show()


# In[30]:


# Identify instances with different predictions
differing_predictions_indices = np.where(y_pred_rf != y_pred_svm)[0]


# In[31]:


# Visualize instances with different predictions
num_instances_to_visualize = min(len(differing_predictions_indices), 10)  # Limit to 10 instances
plt.figure(figsize=(12, 6))
for i, idx in enumerate(differing_predictions_indices[:num_instances_to_visualize]):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f"True: {y_test[idx]}\nRF: {y_pred_rf[idx]}\nSVM: {y_pred_svm[idx]}", fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[32]:


# Analyze differences in predictions
data = []
for idx in differing_predictions_indices[:num_instances_to_visualize]:
    instance_info = {
        'Index': idx,
        'True Label': y_test[idx],
        'RF Prediction': y_pred_rf[idx],
        'SVM Prediction': y_pred_svm[idx]
    }
    data.append(instance_info)

# Create a DataFrame from the list
df = pd.DataFrame(data)

# Display the DataFrame
print(df)


# In[33]:


# Identify misclassified instances
misclassified_indices_rf = np.where(y_pred_rf != y_test)[0]
misclassified_indices_svm = np.where(y_pred_svm != y_test)[0]

# Visualize misclassified instances for Random Forest
print("Misclassified Instances for Random Forest:")
for idx in misclassified_indices_rf[:10]:  # Display first 10 misclassified instances
    plt.imshow(X_test[idx], cmap='gray', interpolation='nearest')
    plt.title(f"True Label: {y_test[idx]}, Predicted Label: {y_pred_rf[idx]}")
    plt.show()

# Visualize misclassified instances for SVM
print("Misclassified Instances for SVM:")
for idx in misclassified_indices_svm[:10]:  # Display first 10 misclassified instances
    plt.imshow(X_test[idx], cmap='gray', interpolation='nearest')
    plt.title(f"True Label: {y_test[idx]}, Predicted Label: {y_pred_svm[idx]}")
    plt.show()


# In[34]:


from sklearn.metrics import roc_curve, auc


# In[35]:


# Generate predicted probabilities from RFC
y_score_rf = rf_classifier.predict_proba(X_test_flat)

# Calculate ROC curve and AUC for each class
n_classes = len(np.unique(y_test))
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test == i, y_score_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr_rf[i], tpr_rf[i], lw=2, label='ROC curve (class %d, AUC = %0.2f)' % (i, roc_auc_rf[i]))

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
plt.legend(loc='lower right')
plt.show()


# In[36]:


# Generate predicted probabilities from SVM
y_score_svm = svm_classifier.decision_function(X_test_flat)

# Calculate ROC curve and AUC for each class
n_classes = len(np.unique(y_test))
fpr_svm = dict()
tpr_svm = dict()
roc_auc_svm = dict()
for i in range(n_classes):
    fpr_svm[i], tpr_svm[i], _ = roc_curve(y_test == i, y_score_svm[:, i])
    roc_auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr_svm[i], tpr_svm[i], lw=2, label='ROC curve (class %d, AUC = %0.2f)' % (i, roc_auc_svm[i]))

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Support Vector Machine')
plt.legend(loc='lower right')
plt.show()

