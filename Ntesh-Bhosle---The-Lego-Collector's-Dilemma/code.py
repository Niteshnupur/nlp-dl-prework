# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here

print(path)

df = pd.read_csv(path)

print(df.head(5))

# splitting the features (independent variables)
X = df[['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country']]

print(X)


# splitting the Target (dependent variable)

y = df['list_price']

print(y)



# splitting dataframe into 70% train data and (0.3) 30% test data


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 6)



# code ends here



# --------------
# Let's check the scatter_plot for different features vs target variable list_price. This tells us which features are highly correlated with the target variable list_price and help us predict it better.


import matplotlib.pyplot as plt

# code starts here        

# Create variable cols store all the X_train columns in it.

print(X_train)

X_train.head(5)


cols = X_train.columns
print(cols)


# Create for loop to iterate through row.
#Create a nested for loop to access column.
#Create variable col and pass cols[ i * 3 + j].
#Plot the scatter plot of each column vs. the list_price


fig , axes = plt.subplots(nrows = 3 , ncols = 3 , figsize=(20,20))

for i in range(0,3):
     for j in range(0,3):
        col = cols[i*3 + j]
        axes[i,j].scatter(X_train[col],y_train)
        axes[i,j].xlabel = X_train[col]
        axes[i,j].ylabel = y_train


# code ends here



# --------------
# Features highly correlated with each other adversely affect our lego pricing model. Thus we keep an inter-feature correlation threshold of 0.75. If two features are correlated and with a value greater than 0.75, remove one of them.



# Code starts here

# Find the correlation between the features which are stored in 'X_train' and store the result in a variable called 'corr'. Print the correlation table

import seaborn as sns

corr = X_train.corr()

#Load the dataset.
#Create a Python Numpy array.
#Create a Pivot in Python.
#Create an Array to Annotate the Heatmap.
#Create the Matplotlib figure and define the plot.
#Create the Heatmap.

sns.heatmap(data = corr)



# We can see that the features of play_star_rating, val_star_rating and star_ratin have a correlation of greater than 0.75. We should drop two of these features to make our model better.
# Remove play_star_rating and val_star_rating from X_train.
# Remove play_star_rating and val_star_rating from X_test.

X_train.drop("play_star_rating", axis = 1 , inplace=True)

X_test.drop("play_star_rating" ,  axis = 1 , inplace=True)

X_train.drop("val_star_rating", axis = 1 , inplace=True)

X_test.drop("val_star_rating" ,  axis = 1 , inplace=True)




# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here

# Instantiate a linear regression model with LinearRegression() and save it to a variable called 'regressor'

regressor = LinearRegression()


# Fit the model on the training data X_train and y_train.

regressor.fit(X_train,y_train)

# Make predictions on the X_test features and save the results in a variable called 'y_pred'.

y_pred = regressor.predict(X_test)
y_pred

# Find the mean squared error and store the result in a variable called 'mse'. Print the value of mse.

mse = mean_squared_error(y_test , y_pred)
print(mse)


# Find the r^2 score and store the result in a variable called 'r2'. Print the value of r2.

r2 = r2_score(y_test , y_pred)
print(r2)





# Code ends here


# --------------
# Code starts here
# Based on the distance between the true target y_test and predicted target y_pred, also known as the residual the cost function is defined. Let's look at the residual and visualize the errors in the model.

# Calculate the residual for true value vs predicted value and store the result into a new variable 'residual'.

residual = y_test - y_pred
print(residual)


# Plot the histogram of the residual.

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.hist(residual , color = "orange" )
plt.xlabel('Residual')
plt.show()





# Code ends here


