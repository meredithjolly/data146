### Question 1
## Download the anonymized dataset describing persons.csv from a West African county and import it into your PyCharm project workspace (right click and download from the above link or you can also find the data pinned to the slack channel). First set the variable wealthC as your target. It is not necessary to set a seed.
```

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import StandardScaler as SS

pns = pd.read_csv("https://raw.githubusercontent.com/tyler-frazier/intro_data_science/main/data/persons.csv", sep=",")

check_nan = pns['age'].isnull().values.any()
pns.dropna(inplace=True)

display(pns.dtypes)

pns['age'] = pns['age'].astype(int)
pns['edu'] = pns['edu'].astype(int)

X = np.array(pns.drop(["wealthC", "wealthI"], axis=1))
y = np.array(pns.wealthC)

X_df = pd.DataFrame(X, columns = ["size","gender","age","edu","county..data_1","county..data_2","county..data_3","county..data_4","county..data_5","county..data_6","county..data_7","county..data_8","county..data_9","county..data_10","county..data_11","county..data_12","county..data_13","county..data_14","county..data_15","settle_density..data_2","settle_density..data_3","settle_density..data_4","potable_water..data_Piped into dwelling","potable_water..data_Piped to yard/plot","potable_water..data_Public tap/standpipe","potable_water..data_Tube well or borehole","potable_water..data_Protected well","potable_water..data_Unprotected well","potable_water..data_Protected spring","potable_water..data_Unprotected spring","potable_water..data_River/dam/lake/ponds/stream/canal/irrigation channel","potable_water..data_Rainwater","potable_water..data_Tanker truck","potable_water..data_Cart with small tank","potable_water..data_Bottled water","toilet_type..data_Flush to piped sewer system","toilet_type..data_Flush to septic tank","toilet_type..data_Flush to pit latrine","toilet_type..data_Flush to somewhere else","toilet_type..data_Flush, don't know where","toilet_type..data_Ventilated Improved Pit latrine (VIP)","toilet_type..data_Pit latrine with slab","toilet_type..data_Pit latrine without slab/open pit","toilet_type..data_No facility/bush/field","toilet_type..data_Composting toilet","toilet_type..data_Bucket toilet","toilet_type..data_Hanging toilet/latrine","toilet_type..data_Other","has_electricity..data_No","has_electricity..data_Yes","has_car..data_No","has_car..data_Yes","cooking_type..data_Electricity","cooking_type..data_LPG","cooking_type..data_Kerosene","cooking_type..data_Charcoal","cooking_type..data_Wood","cooking_type..data_No food cooked in house","has_bank..data_No","has_bank..data_Yes"])

# Create DoKFold function

def DoKFold(model, X, y, k, standardize = False, random_state = 146):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()
           
    train_scores = []
    test_scores = []

    train_mse = []
    test_mse = []
    
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

        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)

        train_mse.append(np.mean((ytrain - ytrain_pred)**2))
        test_mse.append(np.mean((ytest - ytest_pred)**2))

    return train_scores,test_scores,train_mse,test_mse

```
### Question 2
## Perform a linear regression and compute the MSE. Standardize the features and again computer the MSE. Compare the coefficients from each of the two models and describe how they have changed.
```

from sklearn.model_selection import train_test_split as tts
Xtrain, Xtest, ytrain, ytest = tts(X, y, test_size=0.4)

train_scores, test_scores, train_mse, test_mse = DoKFold(LR(), X, y, 20, False)

[np.mean(train_scores), np.mean(test_scores)]
[np.mean(train_mse), np.mean(test_mse)]

```
Training score before standardization: 0.73584
Testing score before standardization: 0.73505
Training MSE before standardization: 0.44279
Testing MSE before standardization: 0.44376

```

train_scores, test_scores, train_mse, test_mse = DoKFold(LR(), X, y, 20, False)

[np.mean(train_scores), np.mean(test_scores)]
[np.mean(train_mse), np.mean(test_mse)]

```
Training score after standardization: 0.73582
Testing score after standardization: 0.73498
Training MSE after standardization: 0.44282
Testing MSE after standardization: 0.44388

The training and testing MSE values increased only slightly after standardizing the features. The training and testing scores decreased slightly after standardizing the features. 

### Question 3
## Run a ridge regression and report your best results.



