#!/usr/bin/env python
# coding: utf-8

# ** Data Exploration **

# In[120]:


import pandas as pd

# Load the dataset
data = pd.read_csv('diabetes.csv')


# In[121]:


print(data.head(10))


# In[122]:


data.groupby('Outcome').mean()


# In[107]:


# Display information about the types of features and missing values
print("\nInformation about the dataset:")
print(data.info())


# In[108]:


# Display summary statistics of numerical features
print("\nSummary statistics of numerical features:")
print(data.describe())


# In[59]:


data.


# In[56]:


# Display the distribution of the target variable (Outcome)
print("\nDistribution of the target variable (Outcome):")
import matplotlib.pyplot as plt
a = data['Outcome'].value_counts()
a.plot.bar()


# ** Data Preprocessing**

# In[109]:


#dropping duplicate values - checking if there are any duplicate rows and dropping if any
data=data.drop_duplicates()


# In[110]:


#check for missing values, count them and print the sum of that count for every column
data.isnull().sum() #conclusion :- there are no null values in this dataset
# Age & DiabetesPedigreeFunction do not have have minimum 0 value so no need to replace , also no. of pregnancies as 0 is possible as observed in df.describe


# In[111]:


#checking for 0 values in 5 columns , 
print(f"BloodPressure:",data[data['BloodPressure']==0].shape[0])                                                    #.shape[0] then gives the number of rows in the resulting DataFrame, which corresponds to the number of instances where 'BloodPressure' is 0.
print(f"Glucose:",data[data['Glucose']==0].shape[0])
print(f"skinThickness:",data[data['SkinThickness']==0].shape[0])
print(f"Insulin:",data[data['Insulin']==0].shape[0])
print(f"BMI:",data[data['BMI']==0].shape[0])


# In[112]:


# Replacing 0 values with median of that column
data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())  # normal distribution
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].mean())  # normal distribution
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].median())  # skewed distribution
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].median())  # skewed distribution
data['BMI'] = data['BMI'].replace(0, data['BMI'].median())  # skewed distribution


# In[91]:


num_cols = len(columns)
num_rows = (num_cols + 2) // 3  # Adjust the number of rows based on the number of columns
fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, num_rows * 4))
fig.suptitle('Histograms for Selected Columns')

# Plot histograms for each column
for i, col in enumerate(columns):
    data[col].plot(kind='hist', ax=axes[i // 3, i % 3], title=col)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plots
plt.show()


# ** Data Visualization **

# In[60]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('diabetes.csv')
data.head()
x = data.drop('Outcome', axis=1)
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(x_test, y_test)))
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["No", "Yes"],
 feature_names=x.columns, impurity=False, filled=True)
import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
def plot_feature_importances_custom(model, feature_names):
    n_features = len(feature_names)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    
custom_feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
plot_feature_importances_custom(tree, custom_feature_names)


# In[119]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Create histograms for key features
plt.figure(figsize=(16, 10))
plt.suptitle('Distribution of Key Features', fontsize=16)

plt.subplot(2, 3, 1)
sns.histplot(data['Glucose'], kde=True, color='skyblue')
plt.title('Glucose Distribution')

plt.subplot(2, 3, 2)
sns.histplot(data['BloodPressure'], kde=True, color='salmon')
plt.title('Blood Pressure Distribution')

plt.subplot(2, 3, 3)
sns.histplot(data['SkinThickness'], kde=True, color='green')
plt.title('Skin Thickness Distribution')

plt.subplot(2, 3, 4)
sns.histplot(data['Pregnancies'], kde=True, color='orange')
# plt.title('Insulin Distribution')

plt.subplot(2, 3, 5)
sns.histplot(data['BMI'], kde=True, color='purple')
# plt.title('BMI Distribution')

plt.subplot(2, 3, 6)
sns.histplot(data['Age'], kde=True, color='purple')
# plt.title('BMI Distribution')

# plt.subplot(2, 3, 7)
# sns.histplot(data['Insulin'], kde=True, color='purple')
# # plt.title('BMI Distribution')


plt.show()


# In[13]:


# Create pair plot
sns.pairplot(data, hue='Outcome', diag_kind='kde', markers=["o", "s"], palette="husl")
plt.suptitle('Pair Plot of Health Indicators by Diabetes Outcome', y=1.02, fontsize=22)
plt.show()


# ** Features Selection **

# In[23]:


# Create correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Health Indicators', fontsize=16)
plt.show()


# In[24]:


data_selected=data.drop(['BloodPressure','Insulin','DiabetesPedigreeFunction'],axis='columns')


# In[97]:


from sklearn.preprocessing import QuantileTransformer

# Assuming df_selected is a subset of the columns from the 'data' DataFrame
x = data[['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome','BloodPressure','Insulin','DiabetesPedigreeFunction']]

quantile = QuantileTransformer()
X = quantile.fit_transform(x)
df_new = pd.DataFrame(X, columns=['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome','BloodPressure','Insulin','DiabetesPedigreeFunction'])

df_new.head()


# In[99]:


plt.figure(figsize=(16, 12))
sns.set_style(style='whitegrid')

plt.subplot(3, 3, 1)
sns.boxplot(x='Glucose', data=df_new)

plt.subplot(3, 3, 2)
sns.boxplot(x='BMI', data=df_new)

plt.subplot(3, 3, 3)
sns.boxplot(x='Pregnancies', data=df_new)

plt.subplot(3, 3, 4)
sns.boxplot(x='Age', data=df_new)

plt.subplot(3, 3, 5)
sns.boxplot(x='SkinThickness', data=df_new)

plt.subplot(3, 3, 6)
sns.boxplot(x='BloodPressure', data=df_new)


plt.subplot(3, 3, 7)
sns.boxplot(x='Insulin', data=df_new)

plt.subplot(3, 3, 8)
sns.boxplot(x='DiabetesPedigreeFunction', data=df_new)
plt.show()


# In[100]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Assume df_new contains the preprocessed data

# Separate features and target variable
X = df_new.drop('Outcome', axis=1)
y = df_new['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for SVM and Neural Network models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 
# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))



# In[123]:


import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Assuming rf_model is already trained
rf_predictions = rf_model.predict(X_test)

# Plot the confusion matrix
disp = plot_confusion_matrix(rf_model, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix - Random Forest')

plt.show()
from sklearn.metrics import recall_score

# Calculate recall for each class
recall_rf = recall_score(y_test, rf_predictions, average=None)

# Plot recall values
plt.bar(range(len(recall_rf)), recall_rf, tick_label=['Class 0', 'Class 1'])
plt.title('Recall for Random Forest')
plt.xlabel('Class')
plt.ylabel('Recall')
plt.show()


# In[124]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming lr_model and y_test have been defined as in your code

conf_matrix = confusion_matrix(y_test, lr_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="GnBu", cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[125]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Assume df_new contains the preprocessed data

# Separate features and target variable
X = df_new.drop('Outcome', axis=1)
y = df_new['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for SVM and Neural Network models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# Support Vector Machine (SVM)
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
print("\nSupport Vector Machine Accuracy:", accuracy_score(y_test, svm_predictions))
print("Support Vector Machine Classification Report:")
print(classification_report(y_test, svm_predictions))

# Neural Network
nn_model = MLPClassifier(max_iter=1000)
nn_model.fit(X_train_scaled, y_train)
nn_predictions = nn_model.predict(X_test_scaled)
print("\nNeural Network Accuracy:", accuracy_score(y_test, nn_predictions))
print("Neural Network Classification Report:")
print(classification_report(y_test, nn_predictions))


# In[45]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

# Function to plot learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curves for each model
title = "Learning Curves"
cv = 5  # Number of cross-validation folds

# Logistic Regression
plot_learning_curve(lr_model, title=, X, y, cv=cv)
plt.show()

# Decision Tree
plot_learning_curve(dt_model, title, X, y, cv=cv)
plt.show()

# Random Forest
plot_learning_curve(rf_model, title, X, y, cv=cv)
plt.show()

# Support Vector Machine (SVM)
plot_learning_curve(svm_model, title, X_train_scaled, y_train, cv=cv)
plt.show()

# Neural Network
plot_learning_curve(nn_model, title, X_train_scaled, y_train, cv=cv)
plt.show()


# In[40]:


from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Logistic Regression
plot_confusion_matrix(lr_model, X_test, y_test, cmap='Blues', display_labels=['0', '1'])
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Decision Tree
plot_confusion_matrix(dt_model, X_test, y_test, cmap='Blues', display_labels=['0', '1'])
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Random Forest
plot_confusion_matrix(rf_model, X_test, y_test, cmap='Blues', display_labels=['0', '1'])
plt.title('Random Forest Confusion Matrix')
plt.show()

# Support Vector Machine (SVM)
plot_confusion_matrix(svm_model, X_test_scaled, y_test, cmap='Blues', display_labels=['0', '1'])
plt.title('SVM Confusion Matrix')
plt.show()

# Neural Network
plot_confusion_matrix(nn_model, X_test_scaled, y_test, cmap='Blues', display_labels=['0', '1'])
plt.title('Neural Network Confusion Matrix')
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Assume df_new contains the preprocessed data

# Separate features and target variable
X = df_new.drop('Outcome', axis=1)
y = df_new['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for SVM and Neural Network models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("Precision:", precision_score(y_test, lr_predictions))
print("Recall:", recall_score(y_test, lr_predictions))
print("F1 Score:", f1_score(y_test, lr_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, lr_predictions))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("Precision:", precision_score(y_test, dt_predictions))
print("Recall:", recall_score(y_test, dt_predictions))
print("F1 Score:", f1_score(y_test, dt_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, dt_predictions))

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Precision:", precision_score(y_test, rf_predictions))
print("Recall:", recall_score(y_test, rf_predictions))
print("F1 Score:", f1_score(y_test, rf_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, rf_predictions))

# Support Vector Machine (SVM)
svm_model = SVC(probability=True)  # probability=True for ROC-AUC score
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
print("\nSupport Vector Machine:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Precision:", precision_score(y_test, svm_predictions))
print("Recall:", recall_score(y_test, svm_predictions))
print("F1 Score:", f1_score(y_test, svm_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, svm_predictions))

# Neural Network
nn_model = MLPClassifier(max_iter=1000)
nn_model.fit(X_train_scaled, y_train)
nn_predictions = nn_model.predict(X_test_scaled)
print("\nNeural Network:")
print("Accuracy:", accuracy_score(y_test, nn_predictions))
print("Precision:", precision_score(y_test, nn_predictions))
print("Recall:", recall_score(y_test, nn_predictions))
print("F1 Score:", f1_score(y_test, nn_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, nn_predictions))


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV to the data
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Print the best parameters
print("Best Hyperparameters:", best_params)

# Evaluate the model with the best parameters on the test set
best_rf_model = grid_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)

# Print performance metrics with the best model
print("\nRandom Forest with Best Hyperparameters:")
print("Accuracy:", accuracy_score(y_test, best_rf_predictions))
print("Precision:", precision_score(y_test, best_rf_predictions))
print("Recall:", recall_score(y_test, best_rf_predictions))
print("F1 Score:", f1_score(y_test, best_rf_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, best_rf_predictions))


# In[1]:


import pandas as pd

# Example dictionary of model results
model_results = {
    'Logistic Regression': {'Accuracy': 0.7468, 'Precision': 0.6538, 'Recall': 0.6182, 'F1 Score': 0.6355, 'ROC-AUC': 0.7182},
    'Decision Tree': {'Accuracy': 0.6494, 'Precision': 0.5091, 'Recall': 0.5091, 'F1 Score': 0.5091, 'ROC-AUC': 0.6182},
    'Random Forest': {'Accuracy': 0.7597, 'Precision': 0.65, 'Recall': 0.7091, 'F1 Score': 0.6783, 'ROC-AUC': 0.7485},
    'Support Vector Machine': {'Accuracy': 0.7662, 'Precision': 0.6727, 'Recall': 0.6727, 'F1 Score': 0.6727, 'ROC-AUC': 0.7455},
    'Neural Network': {'Accuracy': 0.7532, 'Precision': 0.6667, 'Recall': 0.6182, 'F1 Score': 0.6415, 'ROC-AUC': 0.7232},
    'Random Forest (Tuned)': {'Accuracy': 0.7468, 'Precision': 0.629, 'Recall': 0.7091, 'F1 Score': 0.6667, 'ROC-AUC': 0.7384}
}

# Convert the dictionary to a pandas DataFrame
results_df = pd.DataFrame.from_dict(model_results, orient='index')

# Display the results DataFrame
print(results_df)


# In[ ]:




