### Download the dataset charleston_ask.csv and import it into your PyCharm project workspace. Specify and train a model the designates the asking price as your target variable and beds, baths and area (in square feet) as your features. Train and test your target and features using a linear regression model. Describe how your model performed. What were the training and testing scores you produced? How many folds did you assign when partitioning your training and testing data? Interpret and assess your output.
```

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

```
```

homes = pd.read_csv('charleston_ask.csv')

```
```

lin_reg = LinearRegression()

```
```

X = np.array(homes.iloc[:, 1:4])
y = np.array(homes.iloc[:, 0])

```
```

kf = KFold(n_splits = 10, shuffle=True)
train_scores=[]
test_scores=[]

for idxTrain,idxTest in kf.split(X):
    Xtrain = X[idxTrain,:]
    Xtest = X[idxTest,:]
    ytrain = y[idxTrain]
    ytest = y[idxTest]
    lin_reg.fit(Xtrain,ytrain)
    train_scores.append(lin_reg.score(Xtrain,ytrain))
    test_scores.append(lin_reg.score(Xtest,ytest))
    
np.mean(train_scores)

np.mean(test_scores)
    
```
The training score produced from this model using 10 folds averaged 0.019 and the testing score averaged -0.038. The low numbers mean that the model performed poorly. One reason for this could be that the three features are not all on a similar scale. The number of bedrooms and bathrooms are similar, but the area is measured in square feet and it relatively unrelated to the number of bathrooms and bedrooms. Another reason could be that the number of bedrooms, bathrooms and area of a house are not the best predictors of price. In Charleston, the location of the house could have a bigger impact on the price. 
### Now standardize your features (again beds, baths and area) prior to training and testing with a linear regression model (also again with asking price as your target). Now how did your model perform? What were the training and testing scores you produced? How many folds did you assign when partitioning your training and testing data? Interpret and assess your output.
```

def DoKFold(model, X, y, k, standardize=False):
    from sklearn.model_selection import KFold
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()
    
    kf = KFold(n_splits=k, shuffle=True)
    
    train_scores = []
    test_scores = []
    
    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]
        
        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)
        
        model.fit(Xtrain, ytrain)
        
        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

    return train_scores, test_scores

```
```

train_scores, test_scores = DoKFold(lin_reg,X,y,10,True)

np.mean(train_scores)

np.mean(test_scores)

```
The training score produced after standardizing the features averaged 0.019 and the testing score produced after standardizing the features averaged 0.003. This shows that even after standardization, the model still performed poorly because the numbers did not improve. I used the same number of folds (10).
### Then train your dataset with the asking price as your target using a ridge regression model. Now how did your model perform? What were the training and testing scores you produced? Did you standardize the data? Interpret and assess your output.
```

from sklearn.linear_model import Ridge 
a_range = np.linspace(0, 100, 100)
k = 10

avg_tr_score = []
avg_te_score = []

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores, test_scores = DoKFold(rid_reg, X, y, k, standardize=True)
avg_tr_score.append(np.mean(train_scores))
avg_te_score.append(np.mean(test_scores))

```
The training score produced using the ridge regression model on the standardized data averaged 0.019. The testing score produced using standardized data averaged 0.001. I also tried using the ridge regression model on the data before standardization. 
```

from sklearn.linear_model import Ridge 
a_range = np.linspace(0, 100, 100)
k = 10

avg_tr_score = []
avg_te_score = []

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores, test_scores = DoKFold(rid_reg, X, y, k, standardize=False)
avg_tr_score.append(np.mean(train_scores))
avg_te_score.append(np.mean(test_scores))

```
The training score produced from the un-standaedized data using the ridge regression model averaged 0.019. The testing score produced using the un-standardized data averaged -0.002. This shows that the model still performs poorly with the Charleston housing data. This suggests that the number of bedrooms, bathrooms and area is likely not a good predictor of housing cost. 
### Next, go back, train and test each of the three previous model types/specifications, but this time use the dataset charleston_act.csv (actual sale prices). How did each of these three models perform after using the dataset that replaced asking price with the actual sale price? What were the training and testing scores you produced? Interpret and assess your output.
The training score produced from the un-standardized actual sales price data averaged 0.004. The testing score of the un-standardized data averaged -0.018. I then standardized the actual sales price data, and the training score averaged 0.004 and the testing score averaged -0.012. Using ridge regression and the un-standardized data, the training score averaged 0.004 and the testing score averaged -0.04. Using ridge regression on the standardized data produced an average of 0.004 for the training score and an average of -0.035 for the testing score. All of these scores were about the same as the scores produced from the asking housing price data. This shows that the model still performed poorly, even when using the actual price data. 

