Title: Test post II
Date: 2018-08-29 02:14
Category: Data Science
Tags: Data Science, Linear Algebra, Python
Author: Nodar Okroshiashvili
Summary: Summary abut the blog post




```python
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 2*np.pi)
plt.plot(x, np.sin(x))
```

```
[<matplotlib.lines.Line2D at 0x7fcabd56b5c0>]
```

![picture]({static}../../images/Blog_Post_Template_figure1_1.png)




```python
def sqaure(x):
    return x ** 2

sqaure(5)
```

```
25
```




### Some Latex/Katex


$$
\mathbb F(x) \approx
\mathbb F(a) + \mathbb F^{'}(a)\cdot(x - a) + \frac{1}{2!}\cdot\mathbb F^{''}(a)\cdot(x - a)^{2} + 
\frac{1}{3!}\cdot\mathbb F^{3}(a)\cdot(x - a)^{3} + \cdots + \frac{1}{n!}\cdot\mathbb F^{n}(a)\cdot(x - a)^{n}
$$


pweave -f markdown Blog_Post_Template.pmd

