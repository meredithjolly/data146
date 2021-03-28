### Midterm 
A. import the libraries you will need 
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```
B. create your DoKFold
```

def DoKFold(model, X, y, k, standardize = False, random_state = 146):
    import numpy as np
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = k, shuffle = True, random_state = random_state)
    
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

```
add an object for your training and testing scores 
```

    train_scores = []
    test_scores = []

```
add an object for your training and testing MSE
```

    train_mse = []
    test_mse = []

```
add your for loop where you create idxTrain & idxTest using kf.split with your features 
```

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain,:]
        Xtest = X[idxTest,:]
        ytrain = y[idxTrain]
        ytest = y[idxTest]
        
        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)

```
fit your model on this line using your training data 
```

        model.fit(Xtrain, ytrain)

```
use your feature and target training data to calculate your model score and append it to the train score object
```

        train_scores.append(model.score(Xtrain, ytrain))
        
```
use your feature and target testing data to calculate your model score and append it to the test score object
```

        test_scores.append(model.score(Xtest, ytest))

```
use your model to predict target values with the training and testing data
```

        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)

```
use your target values and predicted target values from the training data to calculate your mean MSE and append them to your training MSE object
```

        train_mse.append(np.mean((ytrain - ytrain_pred)**2))

```
use your target values and predicted target values from the testing data to calculate your mean MSE and append them to your testing MSE object
```

        test_mse.append(np.mean((ytest - ytest_pred)**2))

```
```

    return train_scores, test_scores, train_mse, test_mse

```

    
        
