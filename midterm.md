# Midterm 
## A. Import necessary libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```
## B. Create DoKFold
```

def DoKFold(model, X, y, k, standardize = False, random_state = 146):
    import numpy as np
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = k, shuffle = True, random_state = random_state)
    
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

    # add an object for training and testing scores
    train_scores = []
    test_scores = []
    
    #add an object for training and testing MSE
    train_mse = []
    test_mse = []
    
    # for loop to create idxTrain & idxTest using kf.split with features
    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain,:]
        Xtest = X[idxTest,:]
        ytrain = y[idxTrain]
        ytest = y[idxTest]
        
        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)
        
        # to fit model using training data
        model.fit(Xtrain, ytrain)
        
        # use feature and target training and testing data to calculate model score and append it to training and testing score object
        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))
        
        # use model to predict target values with training and testing data
        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)
        
        # use target values and predicted target values from training and testing data to calculate MSE and append them to training and testing MSE object
        train_mse.append(np.mean((ytrain - ytrain_pred)**2))
        test_mse.append(np.mean((ytest - ytest_pred)**2))
        
    return train_scores, test_scores, train_mse, test_mse
    
```
## C. Import the California Housing Data
```

from sklearn.datasets import fetch_california_housing as cal_data
data = cal_data()

```
## D. Create dataframe
```

# set up X as your features from data.data
X = data.data

# create a names object from data.feature_names
X_names = data.feature_names

# set up y as your target from data.target
y = data.target

# use pandas to create a data frame from your features and names object
cal_housing = pd.DataFrame(X, columns = X_names)

```
## 15. Which of the below features is most strongly correlated with the target?
MedInc (median income)
AveRooms (average number of rooms)
AveBedrms (average number of bedrooms)
HouseAg (average house age)
```
# create a data frame with all the features as well as the target
housing_copy = cal_housing.copy()
housing_copy['y'] = y
# calculate correlations of all the variables
housing_copy.corr()

```
correlations between features and targets
- MedInc (median income) = 0.688075
- AveRooms (average number of rooms) = 0.151948
- AveBedrms (average number of bedrooms) = -0.046701
- HouseAg (average house age) = 0.105623
### answer: based on the results above, median income is most strongly correlated with the target. A perfect correlation would be 1, so the closest 1 one is median income (0.69) 

## 16. If the features are standardized, the correlations from the previous question do not change
```
from sklearn.preprocessing import StandardScaler as SS 
ss = SS()

# fit transform the features
Xtransform = ss.fit_transform(X)

# create a data frame with the transformed data
Xtransform_df = pd.DataFrame(X, columns = X_names)
Xtransform_copy = Xtransform_df.copy()

# add target to the data frame
Xtransform_copy['y'] = y

# calculate correlations amongst all variables 
Xtransform_copy.corr()

```
correlations between standardized features and targets
- MedInc (median income) = 0.688075
- AveRooms (average number of rooms) = 0.151948	
- AveBedrms (average number of bedrooms) = -0.046701
- HouseAg (average house age) = 0.105623	

### answer: after standardizing the features, the correlations from the previous question did not change

## 17. If we were to perform a linear regression using on the feature identified in question 15 (median income), what would be the coefficient of determination? Enter answer to two decimal places (ex: 0.12).
```

np.round(np.corrcoef(X_df['MedInc'], y)[0][1]**2, 2)

```
### answer: 0.47 

## 18. Performing different regression methods on the data
Start with linear regresssion. what is the mean R2 value on the test folds? enter answer to 5 decimal places (ex: 0.12345)
standardize data
perform K-fold validation using:
    k = 20
    shuffle = True
    random_state = 146
```

# define k
k = 20

# use DoKFold with LR()
from sklearn.linear_model import LinearRegression as LR
lin_reg = LR()

train_scores, test_scores, train_mse, test_mse = DoKFold(LR(), X, y, k, True)

print(np.mean(train_scores), np.mean(test_scores))
print(np.mean(train_mse), np.mean(test_mse))

```
### answers: 
- training scores: 0.60630
- testing scores: 0.60198
- training MSE: 0.52423
- testing MSE: 0.52880

## 19. Ridge regression
Look at 101 equally spaced values between 20 and 30 for alpha. Use same settings for K-fold validation as in previous question. 
For the optimal value of alpha in this range, what is the mean R2 value on the test folds? Enter answer to 5 decimal places (ex: 0.12345)
```

from sklearn.linear_model import Ridge, Lasso

# define range
rid_a_range = np.linspace(20, 30, 101)

# create an object to append calculated mean values from ridge training and testing data
rid_tr = []
rid_te = []

# create an object to append calculated mean MSE values from ridge training and testing data
rid_mse_tr = []
rid_mse_te = []

for a in rid_a_range:
    mdl = Ridge(alpha=a)
    train, test, train_mse, test_mse = DoKFold(mdl, X, y, k, True)
    
    rid_tr.append(np.mean(train))
    rid_te.append(np.mean(test))
    rid_mse_tr.append(np.mean(train_mse))
    rid_mse_te.append(np.mean(test_mse))

idx = np.argmax(rid_te)
print(rid_a_range[idx], rid_tr[idx], rid_te[idx], rid_mse_tr[idx], rid_mse_te[idx])
plt.plot(rid_a_range, rid_te, 'or')
plt.xlabel('$\\alpha$')
plt.ylabel('Avg $R^2$')
plt.show()

```
### answers: 
- rid_a_range[idx] = 25.8
- rid_tr[idx] = 0.60627
- rid_te[idx] = 0.60201 
- rid_mse_tr[idx] = 0.52427
- rid_mse_te[idx] = 0.52876
![plot](https://meredithjolly.github.io/data146/midterm1.png)

## 20. Lasso regression
Look at 101 equally spaced values between 0.001 and 0.003 for alpha. Use same settings for K-fold validation as in previous question. 
For the optimal value of alpha in this range, what is the mean R2 value on the test folds? Enter answer to 5 decimal places (ex: 0.12345)
```

from sklearn.linear_model import Ridge, Lasso

#define range
las_a_range = np.linspace(0.001, 0.003, 101)

# create an object to append calculated mean values from ridge training and testing data
las_tr = []
las_te = []

# create an object to append calculated mean MSE values from ridge training and testing data
las_mse_tr = []
las_mse_te = []

for a in las_a_range:
    mdl = Lasso(alpha=a)
    train, test, train_mse, test_mse = DoKFold(mdl, X, y, k, True)
    
    las_tr.append(np.mean(train))
    las_te.append(np.mean(test))
    las_mse_tr.append(np.mean(train_mse))
    las_mse_te.append(np.mean(test_mse))

idx = np.argmax(las_te)
print(las_a_range[idx], las_tr[idx], las_te[idx], las_mse_tr[idx], las_mse_te[idx])
plt.plot(las_a_range, las_te, 'or')
plt.xlabel('$\\alpha$')
plt.ylabel('Avg $R^2$')
plt.show()

```
### answers: 
- las_a_range[idx] = 0.00186
- las_tr[idx] = 0.60616
- las_te[idx] = 0.60213 
- las_mse_tr[idx] = 0.52442
- las_mse_te[idx] = 0.52860
![plot](https://meredithjolly.github.io/data146/midterm2.png)

## 21. Refit a linear, ridge and lasso regression to the entire (standardized) dataset
Which of these models estimates the smallest coefficient for the variable that is least correlated (in terms of absolute value of the correlation coefficient) with the target?
```

# identify the index number for the variable that is least correlated (average number of bedrooms)
print(X_names[5])

# create models
lin = LR()
rid = Ridge(alpha = 25.8)
las = Lasso(alpha = 0.00186)

# fit models using transformed features and target
lin.fit(Xtransform, y)
rid.fit(Xtransform, y)
las.fit(Xtransform, y)

# extract coeficcients from each model 
print(lin.coef_[5])
print(rid.coef_[5])
print(las.coef_[5])

```
### answers:
- lin.coef_[5] = -0.03933 
- rid.coef_[5] = -0.03941
- las.coef_[5] = -0.03762

## 22. Which of the above models estimates the smallest coefficient for the variable that is most correlated (median income)?
```

#identify index number for variable that is most correlated (median income)
print(X_names[0])

#extract coefficients from each model
print(lin.coef_[0])
print(rid.coef_[0])
print(las.coef_[0])

```
### answers: 
- lin.coef_[0] = 0.82962
- rid.coef_[0] = 0.82889
- las.coef_[0] = 0.82001

## 23. If we had looked at MSE instead of R2 when doing Ridge regression (question 19), would we have determined the same optimal value for alpha?
```

idx = np.argmin(rid_mse_te)
print(rid_a_range[idx], rid_tr[idx], rid_te[idx], rid_mse_tr[idx], rid_mse_te[idx])

```
```

plt.plot(rid_a_range, rid_mse_te,'or')
plt.xlabel('$\\alpha$')
plt.ylabel('Avg MSE')
plt.show()

```
### answers: I got a slightly higher optimal value for alpha using MSE instead of R2 while doing ridge regression.
- rid_a_range[idx] = 26.1
- rid_tr[idx] = 0.60627
- rid_te[idx] = 0.60201 
- rid_mse_tr[idx] = 0.52427
- rid_mse_te[idx] = 0.52876
![plot](https://meredithjolly.github.io/data146/midterm3.png)

## 24. If we had looked at MSE instead of R2 when doing Lasso regression (question 20), what would the optimal value for alpha be?
```

idx = np.argmin(las_mse_te)
print(las_a_range[idx], las_tr[idx], las_te[idx], las_mse_tr[idx], las_mse_te[idx])

```
```

plt.plot(las_a_range, las_mse_te,'or')
plt.xlabel('$\\alpha$')
plt.ylabel('Avg MSE')
plt.show()

```
### answers: 
- las_a_range[idx] = 0.00186
- las_tr[idx] = 0.60616
- las_te[idx] = 0.602133
- las_mse_tr[idx] = 0.52442
- las_mse_te[idx] = 0.52860
![plot](https://meredithjolly.github.io/data146/midterm4.png)

## Reflection:
I struggled a lot with these questions on the midterm, and I think the main source of my confusion was with the MSE. My DoKFold ended with appending the model scores produced using the feature and target training and testing data to the training and testing scores object. That meant that later on, I was stuck trying to figure out how to predict the taget values with the training and testing data. This is how I was trying to find the predicted target values:
```

prices = Y
predicted = y_pred

summation = 0
n = len(predicted)
for i in range (0,n):
  difference = prices[i] - predicted[i]
  squared_difference = difference**2
  summation = summation + squared_difference

MSE = summation/n
print ("The Mean Squared Error is: " , MSE)

```
It was producing a MSE of 0.61568. This was after I was playing around with it for a while, so I was trying various combinations of inputs after looking things up, but I was ultimately confused about the whole set up so I was never confident if it was the right output.

To produce a ridge regression, I got stuck after not creating a separate object to append the calculated mean training and testing values, along with the MSE values from training and testing. Here is what I was trying to run: 
```

from sklearn.linear_model import Ridge

a_range = np.linspace(20, 30, 101)

k = 20

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,Y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Optimal alpha value: ' + format(a_range[idx], '.3f'))
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))

```

The outcome was:
Optimal alpha value: 30.000
Training score for this value: 0.538
Testing score for this value:0.532
so the alpha value was only about 5 off from what it should be, and the testing and training scores were close to the mean MSE values from the testing and training data, so I think I was trying to calculate the mean testing and training scores but ended up calculating the mean MSE values. 

I tried to calculate the mean MSE values from the ridge testing and training data separately by running: 
```

prices = Y
predicted = avg_te_score

summation = 0
n = len(predicted)
for i in range (0,n):
  difference = prices[i] - predicted[i]
  squared_difference = difference**2
  summation = summation + squared_difference

MSE = summation/n
print ("The Mean Squared Error is: " , MSE)

```
The result was:
The Mean Squared Error is:  1.625559019495714
which is not close to what the mean MSE value should be for the ridge testing and training data. 
I ran into the same issues going forward. Here is what I tried to run for the Lasso regression:
```

from sklearn.linear_model import Lasso

a_range = np.linspace(0.001, 0.003, 101)

k = 20

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    las_reg = Lasso(alpha=a)
    train_scores,test_scores = DoKFold(las_reg,X,Y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))
    
idx = np.argmax(avg_te_score)
print('Optimal alpha value: ' + format(a_range[idx], '.3f'))
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))

```
again, I did not create a separate object for the MSE training and testing data. 

I did not get much futher than that because I kept running into the same problems. The session this past Wednesday really helped because you went over exactly how to approach each problem. 

In my many hours long struggle to try to find answers online to some of the problems I ran into, I did find some useful graphs and charts to produce. I believe we did create a heatmap in class, but one website I looked at was focusing on how to find correlation values. 
I imported the seaborn library and ran this:
```

import seaborn as sns
correlation_matrix = california.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

```
This is the heatmap produced:
![plot](https://meredithjolly.github.io/data146/midterm5.png)

I just think the heatmap was a cool way of looking at the correlation between features and targets. I got the same results in the heatmap as I did in the corrected midterm answers above (question 15).







