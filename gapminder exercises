# Gapminder Exercises (2/8/21)

1. Get a list of all the years in this data, without any duplicates. How many unique values are there, and what are they?
  
  There are 12 years in the data. The years are 1952, 1957, 1962, 1972, 1977, 1982, 1987, 1992, 1997, 2002, 2007
  
  data['year'].unique()
  
  len(data['year'].unique())
  
2. What is the largest value for population (pop) in this data? When and where did this occur?
 
  The largest value for population in this data is 1318683096. This occured in China in 2007. 
 
  max_pop = data['pop'].max()
  max_pop_idx = data['pop'].idxmax()
  data.loc[max_pop_idx]
 
3. Extract all the records for Europe. In 1952, which country had the smallest population, and what was the population in 2007?
  
  In 1952, Iceland had the smallest population (147962). The population in Iceland in 2007 was 301931. 
  
  idx_europe = data['continent']== 'Europe'
  data_europe = data[idx_europe]
  year_data = data_europe['year'] == 1952
  europe_year = data_europe[year_data]
  min_pop = europe_year['pop'].min()
  min_pop_idx = europe_year['pop'].idxmin()
  europe_year.loc[min_pop_idx]
  
  idx_iceland = data['country']== 'Iceland'
  data_iceland = data[idx_iceland]
  year = data_iceland['year']==2007
  iceland_year = data_iceland[year]
