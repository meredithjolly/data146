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
## D. create dataframe
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
### answers: 
- rid_a_range[idx] = 26.1
- rid_tr[idx] = 0.60627
- rid_te[idx] = 0.60201 
- rid_mse_tr[idx] = 0.52427
- rid_mse_te[idx] = 0.52876







