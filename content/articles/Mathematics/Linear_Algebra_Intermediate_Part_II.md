Title: Intermediates of Linear Algebra with Python - Part II
Date: 2019-03-03 12:14
Category: Mathematics
Tags: Linear Algebra, Intermediate Topics
Keywords: intermediate linear algebra with python, matrices and matrix operations in python, matrix norms in python, python, numpy, scipy, sympy, linear algebra for machine learning, linear algebra, machine
Author: Nodar Okroshiashvili
Summary: This is the first part of intermediate Linear Algebra with Python

This is the second part of third post in blog series about linear algebra.


1. Introduction
2. Basics of linear algebra
3. **Intermediate linear algebra**: Part I and **Part II**
4. Advances in linear algebra


This is the continuation of the part one and in this post I will introduce you the following topics:


- [Matrix](#matrix)
  - [Types of Matrices](#types-of-matrices)
  - [Trace of a Matrix](#trace-of-a-matrix)
  - [Determinant of a Matrix](#determinant-of-a-matrix)
  - [Minor of a Matrix](#minor-of-a-matrix)
  - [Cofactor of a Matrix](#cofactor-of-a-matrix)
  - [Determinant of a Matrix continuation](#determinant-of-a-matrix-continuation)
  - [Matrix Division](#matrix-division)
  - [Inverse of a Matrix](#inverse-of-a-matrix)
  - [Matrix Division continuation](#matrix-division-continuation)
  - [Solving System of Equations with Matrices](#solving-system-of-equations-with-matrices)
  - [Elementary Row Operations](#elementary-row-operations)
  - [Rank of a Matrix](#rank-of-a-matrix)
  - [Power of a Matrix](#power-of-a-matrix)
  - [Norm of a Matrix](#norm-of-a-matrix)
- [Numerical Representation](#numerical-representation)
  - [Types of Matrices](#types-of-matrices-1)
  - [Trace of a Matrix](#trace-of-a-matrix-1)
  - [Determinant of a Matrix](#determinant-of-a-matrix-1)
  - [Inverse of a Matrix](#inverse-of-a-matrix-1)
  - [Matrix Division](#matrix-division-1)
  - [Solving System of Equations with Matrices](#solving-system-of-equations-with-matrices-1)
  - [Rank of a Matrix](#rank-of-a-matrix-1)
  - [Power of a Matrix](#power-of-a-matrix-1)
  - [Norm of a Matrix](#norm-of-a-matrix-1)
- [Conclusion](#conclusion)
- [References](#references)


## Matrix

### Types of Matrices


During years of linear algebra evolution, there appeared different types of matrices. Some of them were fundamentals, 
some of them appeared lately. In this part, I will introduce some basic types of matrices and give you reference to find 
some other useful ones. Previously, I talked about the identity matrix, which operates as number 1 in matrix multiplication 
and is denoted by capital letter $I$.

A **square** matrix is a matrix with the same number of rows and columns. 

$$
A =
\begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

A **diagonal** matrix is a matrix in which the entries on principal diagonal are non-zero and all the others are zeros.

$$
A =
\begin{bmatrix}
    1 & 0 & 0 \\
    0 & 2 & 0 \\
    0 & 0 & 3
\end{bmatrix}
$$

Scalar multiple of the identity matrix is called **scalar** matrix that is also diagonal. This means on the main diagonal all elements are equal.

$$
A =
\begin{bmatrix}
    2 & 0 & 0 \\
    0 & 2 & 0 \\
    0 & 0 & 2
\end{bmatrix}
$$

A square matrix is called **triangular** matrix if all of its elements above the main diagonal are zero 
(**lower triangular matrix**) or all of its elements below the main diagonal are zero (**upper triangular matrix**).

$$
A =
\begin{bmatrix}
    1 & 0 & 0 \\
    4 & 5 & 0 \\
    7 & 8 & 9
\end{bmatrix}
\quad
A =
\begin{bmatrix}
    1 & 2 & 3 \\
    0 & 5 & 6 \\
    0 & 0 & 9
\end{bmatrix}
$$

These matrices are lower and upper triangular matrices, respectively.

A **null** or **zero** matrix is a matrix with all elements equal to zero.

$$
A =
\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0
\end{bmatrix}
$$

A matrix of **ones** is where all elements equal to 1.

$$
A =
\begin{bmatrix}
    1 & 1 & 1 \\
    1 & 1 & 1 \\
    1 & 1 & 1
\end{bmatrix}
$$

**Symmetric** matrix is a square matrix that is equal to its own transpose or $A = A^T$. For example,

$$
A =
\begin{bmatrix}
    1 & 2 & 3 \\
    2 & 4 & 5 \\
    3 & 5 & 6
\end{bmatrix}
$$


is a symmetric matrix. Furthermore, matrix elements are symmetric with respect to main diagonal or are equal.

A **skew-symmetric** matrix is a square matrix whose transpose equals its negative or $A^T = -A$. For example,

$$
A =
\begin{bmatrix}
    0 & 3 & 4 \\
    -3 & 0 & 7 \\
    -4 & -7 & 0
\end{bmatrix}
\quad
A^T =
\begin{bmatrix}
    0 & -3 & -4 \\
    3 & 0 & -7 \\
    4 & 7 & 0
\end{bmatrix}
\quad
-A =
\begin{bmatrix}
    0 & -3 & -4 \\
    3 & 0 & -7 \\
    4 & 7 & 0
\end{bmatrix}
$$

**Involutory** matrix is a square matrix that is equal to its own inverse. More precisely, it is the matrix whose square is the identity matrix.

$$
A =
\begin{bmatrix}
    -5 & -8 & 0 \\
    3 & 5 & 0 \\
    1 & 2 & -1
\end{bmatrix}
$$

then 

$$
A^2 =
\begin{bmatrix}
    -5 & -8 & 0 \\
    3 & 5 & 0 \\
    1 & 2 & -1
\end{bmatrix}
\cdot
\begin{bmatrix}
    -5 & -8 & 0 \\
    3 & 5 & 0 \\
    1 & 2 & -1
\end{bmatrix}
\ =
\begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
\end{bmatrix}
\ =
I
$$

A square matrix is called **idempotent** matrix, if multiplied by itself yields itself. Equivalently, $A \cdot A = A$

$$
A =
\begin{bmatrix}
    2 & -2 & -4 \\
    -1 & 3 & 4 \\
    1 & -2 & -3
\end{bmatrix}
\cdot
\begin{bmatrix}
    2 & -2 & -4 \\
    -1 & 3 & 4 \\
    1 & -2 & -3
\end{bmatrix}
\ =
\begin{bmatrix}
    2 & -2 & -4 \\
    -1 & 3 & 4 \\
    1 & -2 & -3
\end{bmatrix}
$$

A **nildepotent** matirix is such that $A^k = 0$  for some positive integer $k$. This means, for some positive $k$, 
multipling matrix $A$ by $k$ times gives zero matrix. For matrix $A$ and for $k=2$ we have:

$$
A =
\begin{bmatrix}
    5 & -3 & 2 \\
    15 & -9 & 6 \\
    10 & -6 & 4
\end{bmatrix}
$$

$$
\quad
$$

$$
A^2 =
\begin{bmatrix}
    5 & -3 & 2 \\
    15 & -9 & 6 \\
    10 & -6 & 4
\end{bmatrix}
\cdot
\begin{bmatrix}
    5 & -3 & 2 \\
    15 & -9 & 6 \\
    10 & -6 & 4
\end{bmatrix}
\ =
\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0
\end{bmatrix}
$$

---
So, as I said there are much much more matrices, but I restricted here due to limited space. If you think you need more, 
definitely check this [wikipedia page](https://en.wikipedia.org/wiki/List_of_matrices).


### Trace of a Matrix

The trace of $n\times n$ square matrix $A$ is the sum of all elements on the main diagonal. 
It is defined only for square matrices and the formula is:

$$
tr(A)=\sum_{i=1}^{n}a_{ii} = a_{11} + a_{22} + \cdots + a_{nn}
$$

Where $a_{ii}$ denotes the entry on $i$-th row and $j$-th column of matrix A.

For example, Let $A$ be a matrix,

$$
A =
\begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33}
\end{bmatrix}
\ =
\begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix}
$$

Then the trace is:

$$
tr(A)=\sum_{i=1}^{3}a_{ii} = a_{11} + a_{22} + a_{33} = 1 + 5 + 9 = 15
$$


### Determinant of a Matrix

There is not plain English definition of determinant but I'll try to explain it by examples to catch the main idea behind 
of that special number. However, we can consider determinant as a function which as an input accepts $n \times n$ matrix 
and output real or a complex number, that is called determinant of input matrix and is denoted by $det(A)$ or $|A|$.

For any $2 \times 2$ square matrix $A$ determinant is calculated by:

$$
A =
\begin{bmatrix}
    a & b \\
    c & d \\
\end{bmatrix}
$$

$$
det(A) = ad - bc
$$

It seems easy to calculate the determinant of any $2 \times 2$ matrix right? Now think about how to calculate determinant 
for higher dimensional matrices...did you find a way? If no, let me explain it step by step. 
If we have, say $3 \times 3$ matrix $A$ and want to calculate determinant we need some other notions such as **minors** 
and **co-factors** of that matrix.


### Minor of a Matrix

A minor of matrix $A$ is the determinant of some smaller square matrix. Precisely, the minor $M_{i,j}$ is the **determinant** of 
matrix $A$ with row $i$ and column $j$ omitted. Minor of matrix $A$ is denoted by $M_{ij}$, where $i$ and $j$ denotes element 
of $i$-th row and $j$-th column. Let have general matrix $A$:

$$
A =
\begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

We can take rows or columns to find all minors. It's up to you which one you take, rows or columns. Let take the columns. 
We take the first element of our matrix $a_{11}$ and delete row and column along it. As the first element is $a_{11}$, 
we have to delete first row and first column. After that, we take the second element of the first column which is $a_{21}$ 
and do same or delete second row and first column. After that, we take the third element of the first column $a_{31}$ and 
delete third row and first column. We have to do these for three columns. After all of that, we have:

$$
M_{11} =
\begin{bmatrix}
    \square & \square & \square \\
    \square & a_{22} & a_{23} \\
    \square & a_{32} & a_{33}
\end{bmatrix}
\ =
\begin{bmatrix}
    a_{22} & a_{23} \\
    a_{32} & a_{33}
\end{bmatrix}
\ =
a_{22}a_{33} - a_{23}a_{32}
$$
$$
\quad
$$
$$
M_{21} =
\begin{bmatrix}
    \square & a_{12} & a_{13} \\
    \square & \square & \square \\
    \square & a_{32} & a_{33}
\end{bmatrix}
\ =
\begin{bmatrix}
    a_{12} & a_{13} \\
    a_{32} & a_{33}
\end{bmatrix}
\ =
a_{12}a_{33} - a_{13}a_{32}
$$
$$
\quad
$$
$$
M_{31} =
\begin{bmatrix}
    \square & a_{12} & a_{13} \\
    \square & a_{22} & a_{23} \\
    \square & \square & \square
\end{bmatrix}
\ =
\begin{bmatrix}
    a_{12} & a_{13} \\
    a_{22} & a_{23}
\end{bmatrix}
\ =
a_{12}a_{23} - a_{13}a_{22}
$$
$$
\quad
$$
$$
M_{12} =
\begin{bmatrix}
    \square & \square & \square \\
    a_{21} & \square & a_{23} \\
    a_{31} & \square & a_{33}
\end{bmatrix}
\ =
\begin{bmatrix}
    a_{21} & a_{23} \\
    a_{31} & a_{33}
\end{bmatrix}
\ =
a_{21}a_{33} - a_{23}a_{31}
$$
$$
\quad
$$
$$
M_{22} =
\begin{bmatrix}
    a_{11} & \square & a_{13} \\
    \square & \square & \square \\
    a_{31} & \square & a_{33}
\end{bmatrix}
\ =
\begin{bmatrix}
    a_{11} & a_{13} \\
    a_{31} & a_{33}
\end{bmatrix}
\ =
a_{11}a_{33} - a_{13}a_{31}
$$
$$
\quad
$$
$$
M_{32} =
\begin{bmatrix}
    a_{11} & \square & a_{13} \\
    a_{21} & \square & a_{23} \\
    \square & \square & \square
\end{bmatrix}
\ =
\begin{bmatrix}
    a_{11} & a_{13} \\
    a_{21} & a_{13}
\end{bmatrix}
\ =
a_{11}a_{13} - a_{13}a_{21}
$$
$$
\quad
$$
$$
M_{13} =
\begin{bmatrix}
    \square & \square & \square \\
    a_{21} & a_{22} & \square \\
    a_{31} & a_{32} & \square
\end{bmatrix}
\ =
\begin{bmatrix}
    a_{21} & a_{22} \\
    a_{31} & a_{32}
\end{bmatrix}
\ =
a_{21}a_{32} - a_{22}a_{31}
$$
$$
\quad
$$
$$
M_{23} =
\begin{bmatrix}
    a_{11} & a_{12} & \square \\
    \square & \square & \square \\
    a_{31} & a_{32} & \square
\end{bmatrix}
\ =
\begin{bmatrix}
    a_{11} & a_{12} \\
    a_{31} & a_{32}
\end{bmatrix}
\ =
a_{11}a_{32} - a_{12}a_{31}
$$
$$
\quad
$$
$$
M_{33} =
\begin{bmatrix}
    a_{11} & a_{12} & \square \\
    a_{21} & a_{22} & \square \\
    \square & \square & \square
\end{bmatrix}
\ =
\begin{bmatrix}
    a_{11} & a_{12} \\
    a_{21} & a_{22}
\end{bmatrix}
\ =
a_{11}a_{22} - a_{12}a_{21}
$$

These nine scalars are minors of matrix $A$. Once again, minor is not smaller matrix, it is **determinant** of a smaller matrix.


### Cofactor of a Matrix

We left one more step to compute determinant of $3 \times 3$ matrix $A$. This step is cofactor of matrix $A$. The cofactor of matrix $A$ is the minor, multiplied by $(-1)^{i+j}$ and is denoted by $C_{ij}$

$$
C_{ij} =
(-1)^{i+j} \cdot M_{ij}
$$

where $i$ is the number of row and $j$ is the number of column of matrix $A$.

In the above case our co-factors are:

$$
C_{11} =
(-1)^{1+1} \cdot M_{11}
$$

$$
C_{21} =
(-1)^{2+1} \cdot M_{21}
$$

$$
C_{31} =
(-1)^{3+1} \cdot M_{31}
$$

$$
C_{12} =
(-1)^{1+2} \cdot M_{12}
$$

$$
C_{22} =
(-1)^{2+2} \cdot M_{22}
$$

$$
C_{32} =
(-1)^{3+2} \cdot M_{32}
$$

$$
C_{13} =
(-1)^{1+3} \cdot M_{13}
$$

$$
C_{23} =
(-1)^{2+3} \cdot M_{23}
$$

$$
C_{33} =
(-1)^{3+3} \cdot M_{33}
$$

So, the sum of $i$ and $j$ in the power of $(-1)$ switch the sign of every minor.


### Determinant of a Matrix continuation

We are ready to compute the determinant of our $3 \times 3$ matrix $A$. We need to expand this matrix along one of 
the row or one of the column to compute the determinant. It's up to you which one you take, row or column. Let take the 
first column. Now, what does expansion means? We have to fix either $i$ if we choose a row, or $j$ if we choose column. 
At first glance it seems confusing but an example will make sense. This expansion is called **Laplace Expansion** and is 
used to compute the determinant of any $n \times n$ matrix.

$$
det(A) = \sum_{j\prime=1}^{n}a_{ij\prime}C_{ij\prime}
\ =
\sum_{i\prime=1}^{n}a_{i\prime j}C_{i\prime j}
$$

where $i\prime$ means we fixed index $i$ or row and we change only column index. In case of $j\prime$ we fixed index $j$ or 
columns and change the only row. So, when $i$ is fixed it is called row expansion and when $j$ is fixed it's called column 
expansion. $C_{ij}$ is our co-factor.

To continue the above example, let expand our initial matrix $A$ by the first column, meaning I fix $j$ to be 1 and only 
change row index $i$ from 1 to 3. In this particular case above formula is:

$$
det(A) = \sum_{j\prime=1}^{3}a_{ij\prime}C_{ij\prime}
\ =
a_{11}C_{11} + a_{21}C_{21} + a_{31}C_{31}
$$

Instead, if I choose first row, I have to fix row index $i$ and change column index $j$ from 1 to 3 and determinant formula is:

$$
det(A) = \sum_{i\prime=1}^{3}a_{i\prime j}C_{i\prime j}
\ =
a_{11}C_{11} + a_{12}C_{12} + a_{13}C_{13}
$$

Below, you will see numerical example.


### Matrix Division

We can't actually divide matrix by a matrix; but when we want to divide matrices, we can take advantage of the fact 
that division by a given number is the same as multiplication by the reciprocal of that number.

For matrix division, we use a related idea; we multiply matrix by the *inverse* of a matrix. If we have two matrices $A$ and $B$, we can do:


$$
A \div B = A \cdot B^{-1}
$$

Here, $B^{-1}$ is the inverse of matrix $B$. As taking inverse of a matrix requires computations and is not easy, let explain it below and then return here.


### Inverse of a Matrix

The matrix is said to be invertible if:

$$
A \cdot A^{-1} = I
$$

where $I$ is the identity matrix and $A^{-1}$ is the inverse of $A$. Generally, matrix inverse is only defined for 
square matrices, but there still exist ways to take the inverse of non-square matrices but this is out of the scope 
of this blog series and I will not consider.

For $2 \times 2$ matrix $A$

$$
A =
\begin{bmatrix}
    a & b \\
    c & d \\
\end{bmatrix}
$$

Inverse of $A$ is:

$$
A^{-1} =
\frac{1}{ad-bc}
\begin{bmatrix}
    d & -b \\
    -c & a \\
\end{bmatrix}
$$

What happened there?
- I swapped the positions of a and d
- I changed the signs of b and c
- I multiplied the resulting matrix by 1 over the **determinant** of the matrix $A$

For example,

$$
A =
\begin{bmatrix}
    6 & 2 \\
    1 & 2 \\
\end{bmatrix}
\quad
A^{-1} =
\frac{1}{(6 \times 2) - (2 \times 1)}
\begin{bmatrix}
    2 & -2 \\
    -1 & 6 \\
\end{bmatrix}
\ =
\begin{bmatrix}
    0.2 & -0.2 \\
    -0.1 & 0.6 \\
\end{bmatrix}
$$

to check if this is really the inverse of $A$, multiply $A$ by its inverse in order to get an identity matrix.

Now let take the inverse of $3 \times 3$ matrix. This process is long and involves taking minors, co-factors and determinant. 
After that, above-mentioned operations should be understandable. It has to be mentioned that there are several ways to take 
matrix inverse but as I started here explaining minors, co-factors and determinant I use this technique to find inverse.

We can calculate the inverse by:

* step 1: **Calculate the matrix of minors**
* step 2: **Turn the matrix of minors into the matrix of cofactors**
* step 3: **Transpose the matrix of cofactors**
* step 4: **Multiply transpose of cofactor by 1/determinant**

Let have matrix:

$$
A =
\begin{bmatrix}
    4 & 2 & 2 \\
    6 & 2 & 4 \\
    2 & 2 & 8
\end{bmatrix}
$$

Step 1: **Calculate the matrix of minors**

$$
M_{11} =
\begin{bmatrix}\color{blue}4 & \color{lightgray}2 & \color{lightgray}2\\\color{lightgray}6 & \color{red}2 & \color{red}4\\\color{lightgray}2 & \color{red}2 & \color{red}8\end{bmatrix}\;\;\;\;(2\times8) - (4\times2) = 8\;\;\;\;\begin{bmatrix}8 & \color{lightgray}? & \color{lightgray}?\\\color{lightgray}? & \color{lightgray}? & \color{lightgray}?\\\color{lightgray}? & \color{lightgray}? & \color{lightgray}?
\end{bmatrix}
$$

$$
M_{12} =
\begin{bmatrix}\color{lightgray}4 & \color{blue}2 & \color{lightgray}2\\\color{red}6 & \color{lightgray}2 & \color{red}4\\\color{red}2 & \color{lightgray}2 & \color{red}8\end{bmatrix}\;\;\;\;(6\times8) - (4\times2) = 40\;\;\;\;\begin{bmatrix}8 & 40 & \color{lightgray}?\\\color{lightgray}? & \color{lightgray}? & \color{lightgray}?\\\color{lightgray}? & \color{lightgray}? & \color{lightgray}?\end{bmatrix}
$$

$$
M_{13} =
\begin{bmatrix}\color{lightgray}4 & \color{lightgray}2 & \color{blue}2\\\color{red}6 & \color{red}2 & \color{lightgray}4\\\color{red}2 & \color{red}2 & \color{lightgray}8\end{bmatrix}\;\;\;\;(6\times2) - (2\times2) = 8\;\;\;\;\begin{bmatrix}8 & 40 & 8\\\color{lightgray}? & \color{lightgray}? & \color{lightgray}?\\\color{lightgray}? & \color{lightgray}? & \color{lightgray}?\end{bmatrix}
$$

$$
M_{21} =
\begin{bmatrix}\color{lightgray}4 & \color{red}2 & \color{red}2\\\color{blue}6 & \color{lightgray}2 & \color{lightgray}4\\\color{lightgray}2 & \color{red}2 & \color{red}8\end{bmatrix}\;\;\;\;(2\times8) - (2\times2) = 12\;\;\;\;\begin{bmatrix}8 & 40 & 8\\12 & \color{lightgray}? & \color{lightgray}?\\\color{lightgray}? & \color{lightgray}? & \color{lightgray}?\end{bmatrix}
$$

$$
M_{22} =
\begin{bmatrix}\color{red}4 & \color{lightgray}2 & \color{red}2\\\color{lightgray}6 & \color{blue}2 & \color{lightgray}4\\\color{red}2 & \color{lightgray}2 & \color{red}8\end{bmatrix}\;\;\;\;(4\times8) - (2\times2) = 28\;\;\;\;\begin{bmatrix}8 & 40 & 8\\12 & 28 & \color{lightgray}?\\\color{lightgray}? & \color{lightgray}? & \color{lightgray}?\end{bmatrix}
$$

$$
M_{23} =
\begin{bmatrix}\color{red}4 & \color{red}2 & \color{lightgray}2\\\color{lightgray}6 & \color{lightgray}2 & \color{blue}4\\\color{red}2 & \color{red}2 & \color{lightgray}8\end{bmatrix}\;\;\;\;(4\times2) - (2\times2) = 4\;\;\;\;\begin{bmatrix}8 & 40 & 8\\12 & 28 & 4\\\color{lightgray}? & \color{lightgray}? & \color{lightgray}?\end{bmatrix}
$$

$$
M_{31} =
\begin{bmatrix}\color{lightgray}4 & \color{red}2 & \color{red}2\\\color{lightgray}6 & \color{red}2 & \color{red}4\\\color{blue}2 & \color{lightgray}2 & \color{lightgray}8\end{bmatrix}\;\;\;\;(2\times4) - (2\times2) = 4\;\;\;\;\begin{bmatrix}8 & 40 & 8\\12 & 28 & 4\\4 & \color{lightgray}? & \color{lightgray}?\end{bmatrix}
$$

$$
M_{32} =
\begin{bmatrix}\color{red}4 & \color{lightgray}2 & \color{red}2\\\color{red}6 & \color{lightgray}2 & \color{red}4\\\color{lightgray}2 & \color{blue}2 & \color{lightgray}8\end{bmatrix}\;\;\;\;(4\times4) - (2\times6) = 4\;\;\;\;\begin{bmatrix}8 & 40 & 8\\12 & 28 & 4\\4 & 4 & \color{lightgray}?\end{bmatrix}
$$

$$
M_{33} =
\begin{bmatrix}\color{red}4 & \color{red}2 & \color{lightgray}2\\\color{red}6 & \color{red}2 & \color{lightgray}4\\\color{lightgray}2 & \color{lightgray}2 & \color{blue}8\end{bmatrix}\;\;\;\;(4\times2) - (2\times6) = -4\;\;\;\;\begin{bmatrix}8 & 40 & 8\\12 & 28 & 4\\4 & 4 & -4\end{bmatrix}
$$

Our matrix of minors is:

$$
M =
\begin{bmatrix}
    8 & 40 & 8 \\
    12 & 28 & 4 \\
    4 & 4 & -4
\end{bmatrix}
$$

Note that I used rows to find minors, in contrast to columns in the previous example.

Step 2: **Turn the matrix of minors into the matrix of cofactors**

To turn minors matrix into cofactor matrix, we just need to change the sign of elements in minors matrix according to 
the rule proposed above section.

Cofactor matrix is:

$$
C =
\begin{bmatrix}
    8 & -40 & 8 \\
    -12 & 28 & -4 \\
    4 & -4 & -4
\end{bmatrix}
$$

Step 3: **Transpose the matrix of cofactors**

We need to take the transpose of the cofactor matrix. In other words, swap their positions over the main diagonal (the main diagonal stays the same).

$$
C^{T}=
\begin{bmatrix}8 & \color{green}-\color{green}1\color{green}2 & \color{orange}4\\\color{green}-\color{green}4\color{green}0 & 28 & \color{purple}-\color{purple}4\\\color{orange}8 & \color{purple}-\color{purple}4 & -4\end{bmatrix}
$$

This matrix is called **Adjugate** or **Adjoint**, which is simple the transpose of the cofactor matrix.

Step 4: **Multiply transpose of cofactor by $\frac{1}{determinant}$**

As we did all the necessary operations to have determinant, let compute it firstly and then multiply the adjoint matrix by $\frac{1}{determinant}$.

Using formula:

$$
det(A) = \sum_{i\prime=1}^{3}a_{i\prime j}C_{i\prime j}
\ =
a_{11}C_{11} + a_{12}C_{12} + a_{13}C_{13}
$$

We have:

$$
det(A) = (4 \times 8) + (2 \times (-40)) + (2 \times 8) = -32
$$

Now the inverse is:

$$
A^{-1} =
\frac{1}{-32}
\cdot
\begin{bmatrix}
    8 & -40 & 8 \\
    -12 & 28 & -4 \\
    4 & -4 & -4
\end{bmatrix}
\ =
\begin{bmatrix}
    -0.25 & 0.375 & -0.125 \\
    1.25 & 0.875 & 0.125 \\
    -0.25 & 0.125 & 0.125
\end{bmatrix}
$$

Let's verify that the original matrix multiplied by the inverse results in an identity matrix:

$$
A \cdot A^{-1} =
\begin{bmatrix}4 & 2 & 2\\6 & 2 & 4\\2 & 2 & 8\end{bmatrix} \cdot \begin{bmatrix}-0.25 & 0.375 & -0.125\\1.25 & -0.875 & 0.125\\-0.25 & 0.125 & 0.125\end{bmatrix} =
\begin{bmatrix}1 & 0 & 0\\0 & 1 & 0\\0 & 0 & 1\end{bmatrix} = I
$$

Do you see how challenging can be finding the inverse of $4 \times 4$ matrix? That's why we use calculators or computer program to compute it.


### Matrix Division continuation

As we already know how to compute the inverse of a matrix, the division is easy now. If we have two matrices:

$$
A =
\begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix}
$$

and 

$$
B =
\begin{bmatrix}
    4 & 2 & 2 \\
    6 & 2 & 4 \\
    2 & 2 & 8
\end{bmatrix}
$$

then $A$ divided by $B$ is

$$
A \cdot B^{-1}
\ =
\begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix}
\cdot
\begin{bmatrix}
    -0.25 & 0.375 & -0.125 \\
    1.25 & 0.875 & 0.125 \\
    -0.25 & 0.125 & 0.125
\end{bmatrix}
\ =
\begin{bmatrix}
    1.5 & -1 & 0.5 \\
    3.75 & -2.125 & 0.875 \\
    6 & -3.25 & 1.25
\end{bmatrix}
\ \equiv
A \div B
$$


### Solving System of Equations with Matrices


In the previous blog, I talked about the system of linear equations and we solved this system graphically and algebraically. 
One of the great things about matrices, is that they can help us solve systems of equations. For example, consider the following system of equations:

$$
\begin{cases}
2x + 4y = 18 \\
6x + 2y = 34
\end{cases}
$$

We can write this in matrix form, like this:

$$
\begin{bmatrix}
    2 & 4 \\
    6 & 2
\end{bmatrix}
\cdot
\begin{bmatrix}
    x \\
    y
\end{bmatrix}
\ =
\begin{bmatrix}
    18 \\
    34
\end{bmatrix}
$$

If we calculate the dot product between the matrix and vector on the left side, we can see clearly that this represents the original equations.

Now let rename our matrices:

$$
A =
\begin{bmatrix}
    2 & 4 \\
    6 & 2
\end{bmatrix}
\quad
X =
\begin{bmatrix}
    x \\
    y
\end{bmatrix}
\quad
B = 
\begin{bmatrix}
    18 \\
    34
\end{bmatrix}
$$

This can be represented as $AX = B$ and we know that to find $X$ we have to solve this: $B \div A$. Since we cannot 
divide matrices in this way, we have to use the previous technique. Find the inverse of $A$ and multiply by $B$.

The inverse of $A$:

$$
A^{-1} =
\begin{bmatrix}
    -0.1 & 0.2 \\
    0.3 & -0.1
\end{bmatrix}
$$

$$
X =
\begin{bmatrix}
    -0.1 & 0.2 \\
    0.3 & -0.1
\end{bmatrix}
\cdot
\begin{bmatrix}
    18 \\
    34
\end{bmatrix}
\ =
\begin{bmatrix}
    5 \\
    2
\end{bmatrix}
$$

Now, instead of $x$ and $y$ in the original equation put $5$ and $2$ and this will make equality true.

$$
10 + 8 = 18
$$
$$
30 + 4 = 34
$$


### Elementary Row Operations

Elementary row operations (ERO) play an important role in many matrix algebra applications, 
such as finding the inverse of a matrix and solving simultaneous linear equations. 
These topics are covered in advance part of the series. An ERO transforms a given matrix $A$ into a 
new matrix $A^{'}$ via one of the following operations:

1. Interchange two rows (or columns)
2. Multiply each element in a row (or column) by a non-zero number
3. Multiply a row (or column) by a non-zero number and add the result to another row (or column)

To catch the idea behind this operations let do the example. We have a matrix $A$ such that

$$
A =
\begin{bmatrix}
    1 & 2 & 3 & 4 \\
    1 & 3 & 5 & 6 \\
    0 & 1 & 2 & 3
\end{bmatrix}
$$


Type 1 ERO that interchange rows 1 and 3 of $A$ would yield

$$
A^{'} =
\begin{bmatrix}
    0 & 1 & 2 & 3 \\
    1 & 3 & 5 & 6 \\
    1 & 2 & 3 & 4
\end{bmatrix}
$$


Type 2 ERO that multiplies row 2 of $A$ by 3 would yield

$$
A^{'} =
\begin{bmatrix}
    1 & 2 & 3 & 4 \\
    3 & 9 & 15 & 18 \\
    0 & 1 & 2 & 3
\end{bmatrix}
$$


Type 3 ERO that multiplies row 2 of $A$ by 4 and replace row 3 of $A$ by $4 \times (\text{row 2 of A}) + \text{row 3 of A}$ 
would yield row 3 of $A^{'}$ to be $ 4 \times [1 \ 3 \ 5 \ 6] + [0 \ 1 \ 2 \ 3] = [4 \ 13 \ 22 \ 27]$ and

$$
A^{'} =
\begin{bmatrix}
    1 & 2 & 3 & 4 \\
    1 & 3 & 5 & 6 \\
    4 & 13 & 22 & 27
\end{bmatrix}
$$

Except this, to perform an elementary row operation on the matrix $A$, first we can perform the operation on the 
corresponding **identity matrix** to obtain an elementary matrix, then multiply $A$ on the left by this elementary matrix. 
More precisely this means that we take one ERO, whichever we want and perform this operation on corresponding identity 
matrix of $A$. If $A$ has $m \times n$ dimension we have identity matrix $I_{m \times n}$. After that, we multiply $A$ 
by this identity matrix. If we denote the elementary matrix by $E$ then, we multiply $A$ by $E$ in the following way:

$$
\begin{equation}E_{1} \cdot A \end{equation}
$$

where $E_{1}$ is ERO one performed on identity matrix.


### Rank of a Matrix

The maximum number of linearly independent rows in a matrix $A$ is called the **row rank** of $A$ and the maximum 
number of linearly independent columns in $A$ is called the **column rank** of $A$. If $A$ is $n \times m$ matrix, 
that is if matrix $A$ has $m$ rows and $n$ columns then the following inequality holds:

$$
\text{row rank of} \ A \leq m \\
\text{column rank of} \ A \leq n
$$

Furthermore, for any matrix $A$

$$
\text{row rank of} \ A = \text{column rank of} \ A
$$

From the above inequality it follows that

$$
Rank(A) \leq min(m, n)
$$

This means that if a matrix has, for example, 3 rows and 5 columns, its rank cannot be more than 3. 
The rank of a matrix would be zero if and only if the matrix had no elements. If a matrix had even one element, 
its minimum rank would be one. When all of the vectors in a matrix are linearly independent, the matrix is said to be **full rank**.

To calculate the rank of a matrix, we have to compute the determinant. It turns out that the rank of a matrix $A$, 
denoted by $Rank(A)$ is the size of the largest non-zero $m \times m$ sub-matrix with non-zero determinant. 
To simplify further, if the determinant of $4 \times 4$ matrix $A$ is zero and any $3 \times3$ sub-matrix of original 
matrix $A$ has non-zero determinant then the rank of the original matrix $A$ is $3$. 
So we can say that rank shows the ["non-degenerateness"](https://en.wikipedia.org/wiki/Degenerate_bilinear_form) of the matrix $A$.

Actually, there is no only one way to compute the rank. I provided one more way in the advanced tutorial.


### Power of a Matrix

We can rise a square matrix $A$ in any nonnegative power just like any number. This is defined as the product of 
$A$ by itself $n$ times. If matrix $A$ has inverse, then $A^{-n} = (A^{-1})^{n}$ or take inverse of $A$ and multiply by itself $n$ times.

For example, if

$$
A =
\begin{bmatrix}
    1 & 2 \\
    3 & 4
\end{bmatrix}
$$

then

$$
A^{2} = 
A A
\ =
\begin{bmatrix}
    1 & 2 \\
    3 & 4
\end{bmatrix}
\cdot
\begin{bmatrix}
    1 & 2 \\
    3 & 4
\end{bmatrix}
\ =
\begin{bmatrix}
    7 & 10 \\
    15 & 22
\end{bmatrix}
$$


### Norm of a Matrix

In the previous post I talked about vector norms but did not mentioned matrix norm. There are three types of matrix norms:

* Matrix norms induced by vector norms
* Entrywise matrix norms
* Schatten norms

Here, I will introduce only the first two types of matrix norm and depict one example of each to give the general idea of matrix norms.

Induced norms usually are denoted by: $$\|A\|_p$$

In the special cases of $p = 1, 2, \infty$, the induced matrix norms can be computed by:

$$
\|A\|_1 = max_{1\leq j \leq m} \sum_{i = 1}^{m}|a_{ij}|
$$

Which is the maximum absolute column sum of the matrix $A$

$$
\|A\|_2 = \|A\|_F = \sigma_{max}(A)
$$

Where $\|A\|_F$ is Frobenius Norm, which will be discussed below and $\sigma_{max}(A)$ is the spectral norm. The later will be discussed in the next post.

$$
\|A\|_{\infty} = max_{1\leq i \leq m} \sum_{j = 1}^{n}|a_{ij}|
$$

which is the maximum absolute row sum of the matrix $A$.

To clarify this farther, let consider the following example:

$$
A =
\begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix}
$$

$$
\|A\|_1 = max(|1| + |4| + |7|; |2| + |5| + |8|; |3| + |6| + |9|) = max(12; 15; 18) = 18
$$
$$
\quad
$$
$$
\|A\|_{\infty} = max(|1| + |2| + |3|; |4| + |5| + |6|; |7| + |8| + |9|) = max(6; 15; 24) = 24
$$

Imagine, we have a vector whose elements are matrices instead of scalars. Then norm defined here is entrywise matrix norm.
The general formula for entrywise matrix norm is:

$$
\|A\|_{p,q} = \left(\sum_{j=1}^{n}\left(\sum_{i=1}^{m}|a_{ij}^{p}\right)^\frac{q}{p}\right)^\frac{1}{q}
$$

where $p,q \geq 1$

When $p=q=2$ we have Frobenius Norm or Frobenius Inner Product:

$$
\|A\|_{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^{n}|a_{ij}|^2}
$$

and when $p=q=\infty$, we have Max Norm:

$$
\|A\|_{max} = max_{ij}|a_{ij}|
$$

If we have matrix $A$ such that:

$$
A = 
\begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix}
$$

Then Frobenius Norm is:

$$
\|A\|_{F} = \sqrt{(|1|^2 + |2|^2 + |3|^2 + |4|^2 + |5|^2 + |6|^2 + |7|^2 + |8|^2 + |9|^2)} = \sqrt{1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81} = \sqrt{285} \approx 16.89
$$

For two matrices $A$ and $B$ we have Frobenius Inner Product:

$$
\langle A,B \rangle_{F} = \sum_{i,j}\overline{A_{i,j}}B_{i,j} = tr \left(\overline{A^T}B \right)
$$

Where overline denotes the complex conjugate of a matrix.

If

$$
A =
\begin{bmatrix}
    2 & 0 \\
    1 & 1 \\
\end{bmatrix}
$$

And

$$
B =
\begin{bmatrix}
    8 & -3 \\
    4 & 1 \\
\end{bmatrix}
$$

Then

$$
\langle A,B \rangle_{F} = 2 \cdot 8 + 0 \cdot (-3) + 1 \cdot 4 + 1 \cdot 1 = 21
$$


## Numerical Representation

### Types of Matrices

```python

import numpy as np


# Diagonal Matrix
diagonal = np.diag([1,2,3])

print('Diagonal Matrix')
print(diagonal)

print()

# Lower Triangular Matrix
low_triang = np.tril([[1,2,3],
                     [4,5,6],
                     [7,8,9]])

print('Lower Triangular Matrix')
print(low_triang)


print()

# Upper Triangular Matrix
upper_triang = np.triu([[1,2,3],
                     [4,5,6],
                     [7,8,9]])

print('Upper Triangular Matrix')
print(upper_triang)

print()


# Matrix of Zeros
zeros = np.zeros((3,3), dtype=int) # "dtype=int" is necessary not to get float values

print('Matrix of Zeros')
print(zeros)


print()


# Matrix of Ones
ones = np.ones((3,3), dtype=int)

print('Matrix of Ones')
print(ones)


print()



# Identity Matrix
identity = np.eye(3, dtype=int)

print('Identity Matrix')
print(identity)
```

```
Diagonal Matrix
[[1 0 0]
 [0 2 0]
 [0 0 3]]

Lower Triangular Matrix
[[1 0 0]
 [4 5 0]
 [7 8 9]]

Upper Triangular Matrix
[[1 2 3]
 [0 5 6]
 [0 0 9]]

Matrix of Zeros
[[0 0 0]
 [0 0 0]
 [0 0 0]]

Matrix of Ones
[[1 1 1]
 [1 1 1]
 [1 1 1]]

Identity Matrix
[[1 0 0]
 [0 1 0]
 [0 0 1]]
```



### Trace of a Matrix

```python

import numpy as np


A = np.array([[1,2,3],
             [4,5,6],
             [7,8,9]])

trace = np.trace(A)

print(trace)
```

```
15
```



### Determinant of a Matrix

```python

import numpy as np

A = np.array([[4,2,2],
             [6,2,4],
             [2,2,8]])

determinant = np.linalg.det(A)

print(determinant)
```

```
-32.000000000000014
```



### Inverse of a Matrix

```python

import numpy as np

A = np.array([[4,2,2],
             [6,2,4],
             [2,2,8]])


inverse = np.linalg.inv(A)

print(inverse)
```

```
[[-0.25   0.375 -0.125]
 [ 1.25  -0.875  0.125]
 [-0.25   0.125  0.125]]
```



### Matrix Division

```python

import numpy as np


A = np.array([[1,2,3],
             [4,5,6],
             [7,8,9]])


B = np.array([[4,2,2],
             [6,2,4],
             [2,2,8]])

# A divided by B is dot product btw A and inverse of B

inv_B = np.linalg.inv(B)

X = np.dot(A, inv_B)

print(X)
```

```
[[ 1.5   -1.     0.5  ]
 [ 3.75  -2.125  0.875]
 [ 6.    -3.25   1.25 ]]
```



### Solving System of Equations with Matrices

```python

import numpy as np

A = np.array([[2, 4],
              [6,2]])

B = np.array([[18],
              [34]])

A_inverse = np.linalg.inv(A)

print('The inverse of A is',A_inverse, sep='\n')

print()

X = np.dot(A_inverse,B)

print('X =', X, sep='\n')
```

```
The inverse of A is
[[-0.1  0.2]
 [ 0.3 -0.1]]

X =
[[5.]
 [2.]]
```



### Rank of a Matrix

```python

import numpy as np


A = np.array([[1,2,3],
             [2,4,6,],
             [1,-3,5]])

rank_A = np.linalg.matrix_rank(A)

print('Rank(A) = ', rank_A)

print()

# matrix B has full rank
B = np.array([[2,2,-1],
             [4,0,2],
             [0,6,-3]])

rank_B = np.linalg.matrix_rank(B)

print('Rank(B) = ', rank_B)
```

```
Rank(A) =  2

Rank(B) =  3
```



### Power of a Matrix

```python

import numpy as np


A = np.array([[1,2],
             [3,4]])


A_square = np.linalg.matrix_power(A,2)

print(A_square)
```

```
[[ 7 10]
 [15 22]]
```


### Norm of a Matrix

```python

import numpy as np

A = np.array([[1,2,3],[4,5,6],[7,8,9]])

frobenius_norm = np.linalg.norm(A, ord='fro')

column_max_norm = np.linalg.norm(A, ord=1)

row_max_norm = np.linalg.norm(A, ord=np.inf)   # same as infinity norm



# Frobenius Inner Product

A = np.array([[2, 0], [1, 1]])

B = np.array([[8, -3], [4,1]])

frobenius_inner_product = np.vdot(A,B)   # numpy.vdot function flattens high dimensional arrays and take dot product


print('Frobenius Norm is:', frobenius_norm)
print('Column Max Norm is:', column_max_norm)
print('Row Max Norm is:', row_max_norm)
print('Frobenius Inner Product is:', frobenius_inner_product)
```

```
Frobenius Norm is: 16.881943016134134
Column Max Norm is: 18.0
Row Max Norm is: 24.0
Frobenius Inner Product is: 21
```




## Conclusion

To sum up both part of intermediate linear algebra, we've reviewed a lot of materials. Some of them seemed easy, while some of them at the first glance seemed complex, but I do hope a little more practice and reading this tutorial 2 times will help you to master all of these intuitions further. 

For any questions, suggestions, corrections and/or short speeches comment below. 


## References

[Introduction To Linear Algebra](http://math.mit.edu/~gs/linearalgebra/)

[Linear Algebra Topics](https://en.wikipedia.org/wiki/List_of_linear_algebra_topics)

[Khan Academy - Linear Algebra](https://www.khanacademy.org/math/linear-algebra)

