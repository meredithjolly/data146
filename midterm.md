# Midterm 
## A. import necessary libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```
## B. create DoKFold
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
## C. import the California Housing Data
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
## 15. which of the below features is most strongly correlated with the target?
MedInc (median income), AveRooms (average number of rooms), AveBedrms (average number of bedrooms), HouseAg (average house age)
```
# create a data frame with all the features as well as the target
housing_copy = cal_housing.copy()
housing_copy['y'] = y
# calculate correlations of all the variables
housing_copy.corr()

```
variable correlations with target
- MedInc (median income) = 0.688075
- AveRooms (average number of rooms) = 0.151948
- AveBedrms (average number of bedrooms) = -0.046701
- HouseAg (average house age) = 0.105623
### answer: based on the results above, **median income** is most strongly correlated with the target. A perfect correlation would be 1
