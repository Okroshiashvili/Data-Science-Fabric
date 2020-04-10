Title: Lost in Pandas: #1
Date: 2019-04-10 19:14
Category: Data Science
Tags: pandas, python, data analysis, pandas tips and tricks
Author: Nodar Okroshiashvili
Summary: Practical Pandas tips and tricks from everyday use


Many have been told and written about Pandas and its capabilities. 
I could not imagine data scientist or data analyst who had not heard about Pandas or had not used it at least once. 
We all use it. Every day, every week. It does not matter how many times. It's a great tool. 
I use it all the time when I want to do data analysis, either it is simple calculations or complex data transformations, and it surprises me. 
Pandas is so simple in its form. However, imagine, how much you can do with some simple method chaining.


Saying all of these, this blog aims to share my experience and amaizment with Pandas. 
This series is not meant for beginner users and will not be short in length. These series will be based on my experience and 
I will try to give a detailed explanation for every step from problem definition to solving it. That's enough for now. Let get down to business.

### Problem Statement

We have data. This data comes from the HR department of the company. The data contains two columns, company name, and information about its employees. 
Each row of the employee information column is a list of lists. The lists inside the outer list can be duplicated. 
It also can have duplicate values, and inner lists have at most two values. 

> Disclaimer: Any name, phone, email, and the title is a pure coincidence. Data is random and fake.


Here is our data.


```python
from collections import defaultdict

import pandas as pd
import numpy as np
```




```python
data = {"company_name": ['A', 'B', 'C', 'D', 'E', 'F'],
        "info": [
[['Name', 'David Jones'],['Title', 'CEO'],['Phone', '207-685-1626'],['Email', 'djones@example.org']],

[['Name', 'Kate Brown'],['Title', 'Senior Lawyer'],['Phone', '316-978-7791'],
['Email', 'Kate.Brown@example.edu'],['Name', 'Darin White'],['Title', 'Associate Vice President'],
['Phone', '316-978-3887'],['Email', 'Darin.White@example.edu']],

[['Name', 'Scott Lamb'],['Title', 'Actuary'],['Phone', '316-978-3804'],
['Email', 'scott.lamb@example.edu'],['Name', 'Scott Lamb'],['Title', 'Senior Officer'],
['Title', 'Application Developer'],['Title', 'Blockchain Architect'],['Title', 'Director of External Affairs'],['Name', 'Scott'],
['Name', 'Scott'],['Title', 'Director of Medicine'],['Title', 'Product Owner'],
['Name', 'Mike'],['Title', 'Domain Expert'],['Title', 'Growth Hacker'],
['Title', 'Engineering Head'],['Title', 'Event Manager'],['Name', 'Joe'],
['Name', 'Mike'],['Title', 'Fundraising'],['Title', 'VP of Customers'],
['Name', 'Mike'],['Title', 'Venture Capital Analyst'],['Title', 'UX Designer'],
['Name', 'Mike'],['Name', 'Susan'],['Name', 'Bryan'],['Name', 'Mia'],['Title', 'Songwriter']],

[['Name', 'Rose Smith Rose Smith'],['Title', 'Vice President'],
['Title', 'Finance and Operations Head'],['Phone', '316-978-3810'],
['Email', 'rose.smith@example.edu'],['Name', 'Rose Smith'],
['Title', 'Foundation'],['Name', 'Susan'],['Title', 'Foundation'],
['Title', 'Accountant'],['Title', 'Accountant'],['Title', 'Executive'],
['Title', 'Director'],['Title', 'Executive'],['Name', 'Ray'],
['Title', 'Strategic Planning'],['Title', 'Financial Analyst'],['Title', 'Foundation'],
['Title', 'Foundation'],['Name', 'Susan'],['Title', 'member of the board'],
['Title', 'board of directors'],['Title', 'president'],['Title', 'board of directors']],

[['Name', 'Carl Clark'],['Title', 'Chief '],['Title', 'Operating Officer'],
['Title', 'PhD'],['Phone', '413-534-2745'],['Email', 'Clark_Carl@example.com']],

[['Title', 'Board Member'],['Name', 'Taylor Garcia'],['Phone', '307-733-2164'],
['Phone', '307-733-4568'],['Email', 'Garcia@example.org']]
        ]
}
```



Let convert this dictionary into Pandas DataFrame and see what data we have.



```python
df = pd.DataFrame(data)

# Print DataFrame head
df.head()
```

```
  company_name                                               info
0            A  [[Name, David Jones], [Title, CEO], [Phone, 20...
1            B  [[Name, Kate Brown], [Title, Senior Lawyer], [...
2            C  [[Name, Scott Lamb], [Title, Actuary], [Phone,...
3            D  [[Name, Rose Smith Rose Smith], [Title, Vice P...
4            E  [[Name, Carl Clark], [Title, Chief ], [Title, ...
```



We see that the first column seems okay, but the second one not. Here we have one big list containing smaller two-element lists.



```python
df['info'].iloc[2]
```

```
[['Name', 'Scott Lamb'],
 ['Title', 'Actuary'],
 ['Phone', '316-978-3804'],
 ['Email', 'scott.lamb@example.edu'],
 ['Name', 'Scott Lamb'],
 ['Title', 'Senior Officer'],
 ['Title', 'Application Developer'],
 ['Title', 'Blockchain Architect'],
 ['Title', 'Director of External Affairs'],
 ['Name', 'Scott'],
 ['Name', 'Scott'],
 ['Title', 'Director of Medicine'],
 ['Title', 'Product Owner'],
 ['Name', 'Mike'],
 ['Title', 'Domain Expert'],
 ['Title', 'Growth Hacker'],
 ['Title', 'Engineering Head'],
 ['Title', 'Event Manager'],
 ['Name', 'Joe'],
 ['Name', 'Mike'],
 ['Title', 'Fundraising'],
 ['Title', 'VP of Customers'],
 ['Name', 'Mike'],
 ['Title', 'Venture Capital Analyst'],
 ['Title', 'UX Designer'],
 ['Name', 'Mike'],
 ['Name', 'Susan'],
 ['Name', 'Bryan'],
 ['Name', 'Mia'],
 ['Title', 'Songwriter']]
```



As we figured out the data structure, let define the aim. 

**We need to unpack lists from the second column and flatten them in tabular format in the way to preserve the order. 
Meaning that, from the above example, ```Scott Lamb``` has to have title ```Actuary``` and not other titles are allowed. 
Long story short, we need to flatten list of list and make proper DataFrame from it.**


### How I approached this problem

The first idea that came to my mind was to use Pandas DataFrame ```.explode()``` method to unpack list of lists, 
which returned lists containing two elements. After that, I extracted these two elements into two different columns.


```python
df_exploded = df.explode('info')

df_exploded.head()
```

```
  company_name                         info
0            A          [Name, David Jones]
0            A                 [Title, CEO]
0            A        [Phone, 207-685-1626]
0            A  [Email, djones@example.org]
1            B           [Name, Kate Brown]
```


```python
# Add two new columns
df_exploded.loc[:, 'tag'] = df_exploded['info'].map(lambda x: x[0])

df_exploded.loc[:, 'result'] = df_exploded['info'].map(lambda x: x[1])

df_exploded.head()
```

```
  company_name                         info    tag              result
0            A          [Name, David Jones]   Name         David Jones
0            A                 [Title, CEO]  Title                 CEO
0            A        [Phone, 207-685-1626]  Phone        207-685-1626
0            A  [Email, djones@example.org]  Email  djones@example.org
1            B           [Name, Kate Brown]   Name          Kate Brown
```



Despite unpacking the list of lists, that is not the format I wanted. So, I need to do an extra transformation.


```python
df_exploded_final = (df_exploded.groupby(['company_name', 'tag'])['result']
                                 .apply(lambda x: pd.Series(x.values))
                                 .unstack(1)
                                 .reset_index()
                                 .drop(['level_1'], axis=1))

df_exploded_final.head()
```

```
tag company_name                    Email         Name         Phone                      tag
0              A       djones@example.org  David Jones  207-685-1626                      CEO
1              B   Kate.Brown@example.edu   Kate Brown  316-978-7791             Senior Lawyer
2              B  Darin.White@example.edu  Darin White  316-978-3887  Associate Vice President
3              C   scott.lamb@example.edu   Scott Lamb  316-978-3804                   Actuary
4              C                      NaN   Scott Lamb           NaN            Senior Officer
```



It seems we did a good job. However, this approach is prone to errors. Namely, it does not preserve the order 
of the list values and may assign a different email to a different person. That was not what I need.

So, I decided to use another way to solve this problem. Notably, as the data contained millions of rows, 
it seemed impossible to be too precise, but I wanted to reduce the error of non-matching cases. To achieve this, I iterated over 
the values of ```info``` column and converted it to dict of lists, where keys are tags (identifiers) and values are actual employee information.


```python
out = []

for x in df['info'].tolist():
    groups = defaultdict(list)
    for g, v in x:
        groups[g].append(v)
    out.append(dict(groups))


df.loc[:,'new_info'] = out

df['new_info'].iloc[0]
```

```
{'Name': ['David Jones'],
 'Title': ['CEO'],
 'Phone': ['207-685-1626'],
 'Email': ['djones@example.org']}
```



That's a step forward. After this, I was interested in counting the values for each key in dicts for each row. 
I made small changes in the above code and applied it to the ```new_info``` column.


```python
out = []

for x in df['new_info']:
    groups = defaultdict(int)
    for g, v in x.items():       
        groups[g] = len(list(filter(None, v)))
    out.append(dict(groups))


df.loc[:,'new_info_stats'] = out

df['new_info_stats'].iloc[0]
```

```
{'Name': 1, 'Title': 1, 'Phone': 1, 'Email': 1}
```



As we calculated value counts for each dict, now we need to add three helper columns to the dataset for further usage. 
These helper columns will help to differentiate matching cases and non-matching cases.


```python
df['_max'] = df['new_info_stats'].apply(lambda x: max(x.values()))

df['_min'] = df['new_info_stats'].apply(lambda x: min(x.values()))

df.loc[:, 'max_equal_min'] = pd.Series(np.where((df['_max'] == df['_min']), 1, 0))
```



The column ```max_equal_min``` is a dummy variable and helps us to differentiate matching and non-matching cases. 
The value 1 indicates we have a matching case and value 0 - non-matching case. According to this column, 
I split data into two parts. The first only contains matching examples, and the second will have only non-matching cases.


```python
df_first = (df[df['max_equal_min'] > 0].reset_index(drop=True))

df_second = (df[df['max_equal_min'] == 0].reset_index(drop=True))
```



The pre-processing of the first DataFrame is over and is ready to flatten. To do so, I iterate over 
```new_info``` column and transform each row into Pandas DataFrame. After this step, data will be flat.


```python
new_data = []

for j in df_first['new_info']:
    new_data.append(pd.DataFrame(j))


df_first_final_i = (pd.concat(new_data, axis=0, sort=False)
                  .drop_duplicates()
                  .dropna(subset=['Name'])
                  .reset_index(drop=True))

df_first_final_i.head()
```

```
          Name                     Title         Phone                    Email
0  David Jones                       CEO  207-685-1626       djones@example.org
1   Kate Brown             Senior Lawyer  316-978-7791   Kate.Brown@example.edu
2  Darin White  Associate Vice President  316-978-3887  Darin.White@example.edu
```

Woohoo, it works! However, imagine having millions of rows how slow this approach will be. 
For this reason, I tried another method and found it much faster. Here it is.


```python
df_first_final_ii = (df_first['new_info'].apply(pd.Series)
                                        .apply(lambda x: x.explode())
                                        .drop_duplicates()
                                        .reset_index(drop=True))

df_first_final_ii.head()
```

```
          Name                     Title         Phone                    Email
0  David Jones                       CEO  207-685-1626       djones@example.org
1   Kate Brown             Senior Lawyer  316-978-7791   Kate.Brown@example.edu
2  Darin White  Associate Vice President  316-978-3887  Darin.White@example.edu
```



Faster and cleaner solution. But, what about the second DataFrame? It turned out that the above solution 
did not fit the second DataFrame and gave me an error. The error was ```ValueError: cannot reindex from a duplicate axis```. 
Before finding the solution for this error, let take a look at the data.


```python
df_second.head()
```

```
  company_name                                               info  \
0            C  [[Name, Scott Lamb], [Title, Actuary], [Phone,...
1            D  [[Name, Rose Smith Rose Smith], [Title, Vice P...
2            E  [[Name, Carl Clark], [Title, Chief ], [Title, ...
3            F  [[Title, Board Member], [Name, Taylor Garcia],...

                                            new_info  \
0  {'Name': ['Scott Lamb', 'Scott Lamb', 'Scott',...
1  {'Name': ['Rose Smith Rose Smith', 'Rose Smith...
2  {'Name': ['Carl Clark'], 'Title': ['Chief ', '...
3  {'Title': ['Board Member'], 'Name': ['Taylor G...

                                      new_info_stats  _max  _min  max_equal_min
0  {'Name': 12, 'Title': 16, 'Phone': 1, 'Email': 1}    16     1              0
1   {'Name': 5, 'Title': 17, 'Phone': 1, 'Email': 1}    17     1              0
2    {'Name': 1, 'Title': 3, 'Phone': 1, 'Email': 1}     3     1              0
3    {'Title': 1, 'Name': 1, 'Phone': 2, 'Email': 1}     2     1              0
```



In the second and third row, we have one ```Name``` and three ```Title``` and two ```Phone```, 
respectively, for the ```new_info_stats``` column. This may be due to the data entry or extraction reason. 
Not 100% sure that this is the case, but the likelihood is very high. So, we have to handle this problem properly. 
One solution is to concatenate strings for the values of ```Title``` and ```Phone``` keys.


```python
def process_info(record: dict) -> dict:

    if (len(record.keys()) == 4 and
            len(record.get("Name")) == 1 and
            len(record.get('Title')) > 1 and
            len(record.get('Email')) == 1 and
            len(record.get('Phone')) == 1):
        record['Title'] = [' '.join(record.get('Title'))]

    elif (len(record.keys()) == 4 and
          len(record.get("Name")) == 1 and
          len(record.get('Title')) == 1 and
          len(record.get('Email')) > 1 and
          len(record.get('Phone')) == 1):
        record['Email'] = [','.join(record.get('Email'))]

    elif (len(record.keys()) == 4 and
          len(record.get("Name")) == 1 and
          len(record.get('Title')) == 1 and
          len(record.get('Email')) == 1 and
          len(record.get('Phone')) > 1):
        record['Phone'] = [','.join(record.get('Phone'))]

    else:
        pass
    return record



df_second['new_info'] = df_second['new_info'].apply(process_info)
```



This is a simple logic to check if we are correctly concatenating string. 
After applying this function, the second DataFrame is ready to flatten. 
As I mentioned above the good old method did not give me 
the desired result for this case and then I came up to the following:


```python
def flatten(df, column):

    data = []

    for i in df[column]:
        data.append(pd.DataFrame(dict([(k, pd.Series(v)) for k, v in i.items()])))
    
    new_df = (pd.concat(data, axis=0, sort=False)
                .drop_duplicates()
                .dropna(subset=['Name'])
                .drop_duplicates(subset=['Name', 'Title', 'Phone', 'Email'])
                .reset_index(drop=True))

    return new_df


df_second_final = flatten(df_second, 'new_info')

df_second_final.head()
```

```
         Name                         Title         Phone                   Email
0  Scott Lamb                       Actuary  316-978-3804  scott.lamb@example.edu
1  Scott Lamb                Senior Officer           NaN                     NaN
2       Scott         Application Developer           NaN                     NaN
3       Scott          Blockchain Architect           NaN                     NaN
4        Mike  Director of External Affairs           NaN                     NaN
```


Applied this function to the second DataFrame flattened it, and combining first and second DataFrames will give the final result.


To sum up, that was only a tiny part of this data pre-processing process. 
However, it was a great journey and lots to learn. What do you think? Did you find a more elegant solution? Please share it in the comments.
