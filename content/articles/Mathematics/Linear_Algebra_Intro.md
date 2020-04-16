Title: Introduction to Linear Algebra with Python
Date: 2019-03-01 12:14
Category: Mathematics
Tags: Linear Algebra
Keywords: linear algebra, linear algebra with python, introduction to linear algebra with python, python, numpy, scipy
Author: Nodar Okroshiashvili
Summary: This is introduction into the Linear Algebra Series



These are the series of linear algebra mainly based on *Numpy* and *Scipy*. 
The series will follow the sequence:


1. **Introduction**
2. Basics of linear algebra
3. Intermediate linear algebra: Part I and Part II
4. Advances in linear algebra: Part I and Part II


**In these series I will attempt to demystify linear algebra concepts to beginners 
and combine it with Python for practical use.**


Linear algebra is one of the building block of data science among others. Its importance is huge, 
as all supervised, unsupervised and semi-supervised algorithms use it with some degree. 
One great example is Google's famous [Page Rank](https://en.wikipedia.org/wiki/PageRank) algorithm, 
which heavily relies on it. And that is not all. We can find as many usage cases of linear 
algebra as many individuals exist in the data science field.

The purpose of this blog series is to introduce you the ways how to use linear algebra 
in data science. More precisely, by this series, I intend to help aspiring data scientist to 
refresh their linear algebra knowledge with Python and gain some hands-on experience. 
Moreover, this may serve you as a starting point to dig deeper into an amazing world of linear 
algebra with Python.


Let start by defining what is linear algebra

> <div class="alert alert-block alert-info">
&nbsp;&nbsp;&nbsp;&nbsp;Linear algebra is a branch of mathematics that is concerned 
with mathematical structures, closed under addition and scalar multiplication operations 
and that includes the theory of systems of linear equations, matrices, determinants, 
vector spaces, and linear transformations.
</div>


Let's start explaining word by word the definition of linear algebra. 
Loosely speaking, [mathematical structure](https://en.wikipedia.org/wiki/Mathematical_structure) 
is a set, together with a family of operations and relations defined on that set. 
Now divide "Closed under addition and scalar multiplication" into two parts. 
First is "closed under addition" which means that a set is "closed under addition" if the sum 
of any two members of this set belongs to this set again. For example, imagine the set of 
even integers. Then, take any to integer and add them up. The result is an even integer 
belonging to the initial set. Here is a mathematical definition.

$$
A=\{x \in \mathbb{Z}~\vert~mod~2 =0\}
$$

Second is "closed under scalar multiplication". This means that the product of any member of 
the set and any scalar $\alpha$ such that $\alpha~\in \mathbb{R}$ is also in the set. 
The above-mentioned set is also closed under scalar multiplication. 
Generally, the sets $\mathbb{N}, ~ \mathbb{Z}, ~ \mathbb{Q}$ and $\mathbb{R}$ are closed 
under both addition and multiplication.

$$
A = (0,1)
$$

is closed under multiplication, but not addition. $(0.6 + 0.7 = 1.3 > 1)$

The set of all half integers 
$$
\frac{\mathbb{Z}}{2} = 
\{x : \exists~{y} \in \mathbb{Z}~(x = \frac{y}{2})\}
$$
is closed under addition, but not under multiplication. $(0.5 * 0.5 = 0.25~\notin~\frac{\mathbb{Z}}{2})$


The system of linear equations is a collection of two or more linear equations involving 
the same set of variables. The example is the following:
$$
\begin{cases}
3x + 2y - z = 1 \\
2x - 2y + 4z = -2 \\
-x + \frac{1}{2}y - z = 0
\end{cases}
$$


[Matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)) is just a rectangular array of numbers, 
symbols, or expressions, arranged in rows and columns. There is no plain English definition of 
the determinant, but forthcoming blogs in this series will cover it in detail. 
Vector space is a set of objects (vectors) closed under finite vector addition and scalar multiplication.


Linear transformation or linear map is a mapping between two vector spaces that preserves 
the addition and scalar multiplication rule. More mathematically, a linear transformation between 
two vector space $V$ and $W$ is a map $T : V~\rightarrow~W$ such that the following hold:


* $T(v_1 + v_2) = T(v_1) + T(v_2)$  for any vectors $v_1$ and $v_2$ in $V$


* $T(\alpha v_1) = \alpha T(v_1)$  for any scalar $\alpha$



### Conclusion
To sum up,  this post is an introduction towards linear algebra series, where I will introduce you 
linear algebra concepts intuitively and programmatically in Python. 
The main idea of this series is to feel comfortable in the field and to give you the direction where to dig deeper.


