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

To produce a left skewed plot:
```
n = 1000
a = 1
b = 0.5
#np.random.seed(10)
leftskew = np.random.beta(a, b, size=n)

plt.figure(figsize = (8, 8))
plt.hist(leftskew, rwidth = 0.8)
plt.show()

```
Here is the left skewed plot:
![left skewed](https://meredithjolly.github.io/data146/prj2_3.png) 

The mean for the left skewed plot is 0.666 and the median is 0.760
### Using the gapminder data set, produce two overlapping histograms within the same plot describing life expectancy in 1952 and 2007. Plot the overlapping histograms using both the raw data and then after applying a logarithmic transformation (np.log10() is fine). Which of the two resulting plots best communicates the change in life expectancy amongst all of these countries from 1952 to 2007?
Here is the histogram showing the life expectancy in 1952 and 2007:
![life exp](https://meredithjolly.github.io/data146/prj2_4.png)

Here is the histogram with the logarithmic transformation:
![life exp log](https://meredithjolly.github.io/data146/prj2_5.png)

The plot with the logarithmic transformation is a better representation of the change in life expectancy amongst all of the countries from 1952 to 2007. 
### Using the seaborn library of functions, produce a box and whiskers plot of population for all countries at the given 5-year intervals. Also apply a logarithmic transformation to this data and produce a second plot. Which of the two resulting box and whiskers plots best communicates the change in population amongst all of these countries from 1952 to 2007?
Here is the plot of population for all countries:
![pop box and whiskers plot](https://meredithjolly.github.io/data146/prj2_6.png)

Here is the plot of population for all countries with the logarithmic transformation:
![pop log box and whiskers plot](https://meredithjolly.github.io/data146/prj2_7.png)

The plot with the logarithmic transformation best communicates the change in population amongst all of the countries. 

