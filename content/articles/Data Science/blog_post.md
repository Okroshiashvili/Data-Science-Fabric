Title: Test post
Date: 2018-08-29 02:14
Tags: Data Science
Author: Nodar Okroshiashvili


> Written with [StackEdit](https://stackedit.io/).

# Churn Rate Modeling



    import pandas as pd
    df = pd.read_excel('nodo.xlsx')
    from sklearn import preprocessing

> This is my new definition of data science


```python
import numpy as np
import matplotlib.pyplot as plt
```




```python
x = np.linspace(0, 2*np.pi)
plt.plot(x, np.sin(x))
```

```
[<matplotlib.lines.Line2D at 0x7f032d0992e8>]
```

![picture](content/article/images/blog_post_figure2_1.png)\




```python
def sqaure(x):
    return x ** 2

sqaure(5)
```

```
25
```




pweave -f markdown blog_post.pmd

