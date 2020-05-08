# --------------
# Data loading and splitting
#The first step - you know the drill by now - load the dataset and see how it looks like.   Additionally, split it into train and test set.


# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
warnings.filterwarnings('ignore')

# Code starts here

# Load dataset using pandas read_csv api in variable df and give file path as path.

file_path = path
print(file_path)

df = pd.read_csv(path)
print(df)


# Display first 5 columns of dataframe df.

df.head(5)


# Store all the features(independent values) in a variable called X

X = df[["age" , "sex" , "bmi"	, "children" , 	"smoker" , 	"region" , "charges" ]] 

print(X)

# Store the target variable (dependent value) in a variable called y

y = df["insuranceclaim"]
print(y)


# Split the dataframe into X_train,X_test,y_train,y_test using train_test_split() function. Use test_size = 0.2 and random_state = 6

train , test = train_test_split(df , test_size = 0.2 , random_state = 6)

X_train = train.drop(["insuranceclaim"] , axis = 1)
y_train = train["insuranceclaim"]

X_test = test.drop(["insuranceclaim"] , axis = 1)
y_test = test["insuranceclaim"]








# Code ends here


# --------------
# Outlier Detection
# Let's plot the box plot to check for the outlier.



import matplotlib.pyplot as plt


# Code starts here

# Plot the boxplot for X_train['bmi'].



plt.boxplot(X_train["bmi"])


# Set quantile equal to 0.95for X_train['bmi']. and store it in variable q_value.

q_value = X_train["bmi"].quantile(0.95)
print(q_value)


# Check the value counts of the y_train

y_train.value_counts()



# Code ends here


# --------------
# Code starts here

# Correlation Check !
#Let's check the pair_plot for feature vs feature. This tells us which features are highly  correlated with the other feature and help us predict its better logistic regression model.

# Find the correlation between the features which are stored in 'X_train' and store the result in a variable called 'relation'.

relation = X_train.corr()
print(relation)


# plot pairplot for X_train.

sns.pairplot(X_train)





# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt


# Predictor check!
#Let's check the count_plot for different features vs target variable insuranceclaim. This  tells us which features are highly correlated with the target variable insuranceclaim and help  us predict it better.

# Code starts here

# Create a list cols store the columns 'children','sex','region','smoker' in it.

cols = ['children','sex','region','smoker']
print(cols)
type(cols)


# Create subplot with (nrows = 2 , ncols = 2) and store it in variable's fig ,axes

fig , axes = plt.subplots(nrows=2 , ncols=2 , figsize=(30,30))


# Create for loop to iterate through row.

# Create another for loop inside for to access column.

# create variable col and pass cols[ i * 2 + j].

# Using seaborn plot the countplot where x=X_train[col], hue=y_train, ax=axes[i,j]

for i in range(0,2):
    for j in range(0,2):
        col = cols[i * 2 + j]
        sns.countplot(x=X_train[col],hue=y_train,ax=axes[i,j])


# Code ends here


# --------------



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Is my Insurance claim prediction right?
# Now let's come to the actual task, using logistic regression to predict the insuranceclaim.   We will select the best model by cross-validation using Grid Search.


# You are given a list of values for regularization parameters for the logistic regression model.

# parameters for grid search

parameters = {'C':[0.1,0.5,1,5]}
print(parameters)


# Instantiate a logistic regression model with LogisticRegression() and pass the parameter as random_state=9 and save it to a variable called 'lr'.


lr = LogisticRegression(random_state=9)


# Inside GridSearchCV() pass estimator as the logistic model, param_grid=parameters. to do grid search on the logistic regression model store the result in variable grid.


grid = GridSearchCV(estimator=lr , param_grid=parameters)


# Fit the model on the training data X_train and y_train.

grid.fit(X_train,y_train)


# Make predictions on the X_test features and save the results in a variable called 'y_pred'.

y_pred = grid.predict(X_test)


# Calculate accuracy for grid and store the result in the variable accuracy

accuracy = accuracy_score(y_test , y_pred)


# print accuracy

print(accuracy)


# Code starts here



# Code ends here


# --------------
# Performance of a classifier !
# Now let's visualize the performance of a binary classifier. Check the performance of the classifier using roc auc curve.


from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Calculate the roc_auc_score and store the result in variable score.

score = roc_auc_score(y_test , y_pred)
print(score)


# Predict the probability using grid.predict_proba on X_test and take the second column and store the result in y_pred_proba.

y_pred_proba = grid.predict_proba(X_test)
print(y_pred_proba)

y_pred_proba = y_pred_proba[:,1]
print(y_pred_proba)



# Use metrics.roc_curve to calculate the fpr and tpr and store the result in variables fpr, tpr, _.

fpr , tpr , _ = metrics.roc_curve(y_test , y_pred_proba)


# Calculate the roc_auc score of y_test and y_pred_proba and store it in variable called roc_auc.


roc_auc = roc_auc_score(y_test , y_pred_proba)
print(roc_auc)


# Plot auc curve of 'roc_auc' using the line plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc)).


plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))
plt.legend(loc = 4)
plt.show()


# Code starts here







# Code ends here


