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

### You have been introduced to four logical operators thus far: &, ==, | and ^. Describe each one including its purpose and function. Provide an example of how each might be used in the context of programming.
& is the bitwise AND operator. Bitwise operators perform boolean logic on individual bits. If both bits in the pair being evaluated are 1, it will return a 1. Otherwise, it will return a 0. The bitwise & operator can be used for filtering a dataframe based on more than one comparison. In the example above, the & operator was used to select a subset of the gapminder dataset based on year and country. 
== means equal to. It returns True if the two values equal eachother, and False if they do not. An example would be a = 1 , b = 1 , print(a == b) returns: True. It returned true because a and b are the same. 
| is the bitwise OR operator. If the two bits in the pair are both 0, the output is 0. If at least one of the bits in a pair is 1, it will return 1. For example, if I replaced & with | in the gapminder subset example, it would subset all values that are in the year 2007 or the countries specified, not just the values that satisfy both conditions. 
^ is the bitwise exclusive OR operator. If one of the two bits in the pair is 1, the output will be 1. If both of the bits in the pair are 1 or 0, it will return 0. If I replaced & with ^ in the gapminder subset example, it would subset all values that are either in the year 2007 or the countries specified. It will only subset the values that satisfy one of conditions, not values that satisfy both or none of the conditions. 
### Describe the difference between .loc and .iloc. Provide an example of how to extract a series of consecutive observations from a data frame. Stretch goal: provide an example of how to extract all observations from a series of consecutive columns.
.loc and .iloc are two different data selection methods. .loc is label based, so you select rows and columns based on their row or column label. .iloc is integer index based, so you select rows and columns based on their integer location. 
To select consecutive observations using .loc and .iloc:
```

new.loc[539:1427]
new.iloc[0:4]

```
### Describe how an api works. Provide an example of how to construct a request to a remote server in order to pull data, write it to a local file and then import it to your current work session.
API stands for Application Programming Interface. It is a software intermediary that allows two applications to talk to each other. 
```

import requests 
url = "https://api.covidtracking.com/v1/states/daily.csv"
r = requests.get(url)
filename = 'data_folder'
with open(file_name, 'wb') as f:
    f.write(r.content)
import pandas as pd
df = pd.read_csv(file_name)

```
### Describe the apply() function from the pandas library. What is its purpose? Using apply) to various class objects is an alternative (potentially preferable approach) to writing what other type of command? Why do you think apply() could be a preferred approach?
The apply() function allows you to use lambda functions to iterate on columns and rows in dataframes. The apply() function is an alternative to using a loop to interate over a dataframe. It is often faster and uses less lines of code, ultimately simplifying the code. 
### Also describe an alternative approach to filtering the number of columns in a data frame. Instead of using .iloc, what other approach might be used to select, filter and assign a subset number of variables to a new data frame?
One alternative approach to filtering a dataframe would be to use the filter() function. This function subsets rows or columns based on the labels of the specified index. Using the gapminder dataset as an example, if I wanted to subset the country, year and pop columns, I would use:
```

data.filter(["country", "year", "pop"])

```





