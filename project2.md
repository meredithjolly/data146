### Describe continuous, ordinal and nominal data. Provide examples of each. Describe a model of your own construction that incorporates variables of each type of data. You are perfectly welcome to describe your model using english rather than mathematical notation if you prefer. Include hypothetical variables that represent your features and target.
Continuous data is data that can take on any value. It is a measurement so it can take on any value in a range, but it cannot be counted. An example of continuous data is a person's height or the outside temperature. Ordinal data represents discrete and ordered values. The order matters but the difference between values does not matter. An example of ordinal data is education level. Nominal data is discrete values used to label variables. Nominal data has no value and there is no order to the values. 

Model example: Predicting satisfaction rating from a customer based on age, sex and income. The dependent variable (target) is satisfaction rating and the independent variables (features) are age, sex and income. Satisfaction rating and income are ordinal data. Age is continuous data and sex is nominal data. 
### Comment out the seed from your randomly generated data set of 1000 observations and use the beta distribution to produce a plot that has a mean that approximates the 50th percentile. Also produce both a right skewed and left skewed plot by modifying the alpha and beta parameters from the distribution. Be sure to modify the widths of your columns in order to improve legibility of the bins (intervals). Include the mean and median for all three plots.
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 1000
a = 5
b = 5
#np.random.seed(10)
data = np.random.beta(a, b, size=n)

```
Here is the plot that has a mean that approximates the 50th percentile:
![plot](https://meredithjolly.github.io/data146/prj2_1.png) 

The mean for the plot above is 0.502 and the median is 0.499

To produce a right skewed plot:
```
n = 1000
a = 0.5
b = 1
#np.random.seed(10)
rightskew = np.random.beta(a, b, size=n)

plt.figure(figsize = (8, 8))
plt.hist(rightskew, rwidth = 0.8)
plt.show()

```
Here is the right skewed plot:
![right skewed](https://meredithjolly.github.io/data146/prj2_2.png) 

The mean for the right skewed plot is 0.323 and the median is 0.242

