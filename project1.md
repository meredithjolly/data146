# Project 1

### Describe what is a package? Also, describe what is a library? What are the two steps you need to execute in order to install a package and then make that library of functions accessible to your workspace and current python work session? Provide examples of how you would execute these two steps using two of the packages we have used in class thus far. Be sure to include an alias in at least one of your two examples and explain why it is a good idea to do so.
A package is a directory that contains various Python modules. A library is a collection of packages that can perform various functions. It is reusable code that you can use by importing it in the program. You can use the pip tool to install a package, then you import the library to the workspace to make the library functions accessable. While importing a library, you can rename it so you only have to use the alias in the workspace instead of typing out the full library name. Here is an example:
```
pip install pandas
pip install numpy 

import pandas as pd
import numpy as np

```
### Describe what is a data frame? Identify a library of functions that is particularly useful for working with data frames. In order to read a file in its remote location within the file system of your operating system, which command would you use? Provide an example of how to read a file and import it into your work session in order to create a new data frame. Also, describe why specifying an argument within a read_() function can be significant. Does data that is saved as a file in a different type of format require a particular argument in order for a data frame to be successfully imported? Also, provide an example that describes a data frame you created. How do you determine how many rows and columns are in a data frame? Is there an alternate terminology for describing rows and columns?
A dataframe is a two-dimensional structure that contains rows, columns and data. The Pandas library is a useful library to use while working with dataframes. To read a file from a remote location and import the data to your workspace, you would specify the directory where the file is located and the use the pandas .read() function. It is important to specify how the values are separated within the .read() function because different types of files use different ways to separate values. Two examples are comma separated values (csv) and tab separated values (tsv). Since my workspace was in the same location as the file, I just put the filename instead of the whole path to data in the directory. Here is my example of reading in data from a file to my workspace:
```

data = pd.read_csv('gapminder.tsv', sep = '\t')

```
And here is what the dataframe I created looks like: 
![gapminder dataframe](https://meredithjolly.github.io/data146/gapminder_dataframe.png) 

To determine how many rows and columns are in my dataframe:
```

data.shape

```
An alternative name for a row is an observation, and an alternative name for a column is a variable. 
### Import the gapminder.tsv data set and create a new data frame. Interrogate and describe the year variable within the data frame you created. Does this variable exhibit regular intervals? If you were to add new outcomes to the raw data in order to update and make it more current, which years would you add to each subset of observations? Stretch goal: can you identify how many new outcomes in total you would be adding to your data frame?
To see what years are included in the dataframe:
```

data['year'].unique()

```
There are 12 different years in the dataframe. The years start from 1952 and end with 2007. The years increase in intervals of 5. To make the dataframe up to date, I would add data from the years 2012 and 2017. There are 142 different countries in the dataframe so if I added data for those two years for every country, there would be 284 new rows. 
### Using the data frame you created by importing the gapminder.tsv data set, determine which country at what point in time had the lowest life expectancy. Conduct a cursory level investigation as to why this was the case and provide a brief explanation in support of your explanation.
To see what the lowest value was for life expectancy and then to determine the country and year:
```

data['lifeExp'].min()
data.loc[data['lifeExp'] == 23.599]

```
The lowest life expectancy was 23.599 years and this occured in Rwanda in 1992. One possible reason for this was that 1992 was right in the middle of the Rwandan Civil War. 
### Using the data frame you created by importing the gapminder.tsv data set, multiply the variable pop by the variable gdpPercap and assign the results to a newly created variable. Then subset and order from highest to lowest the results for Germany, France, Italy and Spain in 2007. Create a table that illustrates your results (you are welcome to either create a table in markdown or plot/save in PyCharm and upload the image). Stretch goal: which of the four European countries exhibited the most significant increase in total gross domestic product during the previous 5-year period (to 2007)?
To multiply the variable pop by the variable gdpPercap and add this to the dataframe as a new column:
```

i = 0
gdp = []
for pop in data['pop']:
    a = (pop * (data['gdpPercap'].iloc[i]))
    gdp.append(a)
    i += 1 
data['gdp'] = gdp

```
To select a subset of the dataframe:
```

countries = ['Germany', 'France', 'Italy', 'Spain']
new = data.loc[(data['year'] == 2007) & data['country'].isin(countries)]

```
To order the subset from highest to lowest:
```

new.sort_values(by = ['gdp'], ascending = False)

```
Here is a screenshot of the result:
![gapminder dataframe](https://meredithjolly.github.io/data146/gapminder_subset.png)




