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
[gapminder dataframe](Screen Shot 2021-02-23 at 9.01.13 PM.png)


