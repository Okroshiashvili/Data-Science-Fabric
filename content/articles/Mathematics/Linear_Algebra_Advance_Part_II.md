Title: Advance Linear Algebra with Python - Part II
Date: 2019-03-05 12:14
Category: Mathematics
Tags: Linear Algebra, Advance Topics
Keywords: advance linear algebra, matrix decompositions in python, linear algebra advances in python, advance linear algebra for machine learning
Author: Nodar Okroshiashvili
Summary: This is the second part of advance linear algebra with Python


This is the fourth and last post in blog series about linear algebra.


1. Introduction
2. Basics of linear algebra
3. Intermediate linear algebra
4. **Advances in linear algebra**: Part I and **Part II**


This is the continuation of the part one and in this post I will introduce you the following topics:


- [Matrix Decompositions](#matrix-decompositions)
  - [Cholesky Decomposition](#cholesky-decomposition)
  - [QR Decomposition](#qr-decomposition)
  - [Eigendecomposition](#eigendecomposition)
  - [Singular Value Decomposition](#singular-value-decomposition)
  - [Inverse of a Square Full Rank Matrix](#inverse-of-a-square-full-rank-matrix)
- [Numerical Representation](#numerical-representation)
  - [Cholesky Decomposition](#cholesky-decomposition-1)
  - [QR Decomposition](#qr-decomposition-1)
  - [Eigendecomposition](#eigendecomposition-1)
  - [Singular Value Decomposition](#singular-value-decomposition-1)
  - [Inverse of a Square Full Rank Matrix](#inverse-of-a-square-full-rank-matrix-1)
- [Conclusion](#conclusion)
- [References](#references)


## Matrix Decompositions

In linear algebra, matrix decomposition or matrix factorization is a factorization of a matrix into a product of matrices. 
Factorizing a matrix means that we want to find a product of matrices that is equal to the initial matrix. 
These techniques have a wide variety of uses and consequently, there exist several types of decompositions. 
Below, I will consider some of them, mostly applicable to machine learning or deep learning.


### Cholesky Decomposition

The Cholesky Decomposition is the factorization of a given **symmetric** square matrix $A$ into the product of a 
lower triangular matrix, denoted by $L$ and its transpose $L^{T}$. This decomposition is named after French artillery 
officer [Andre-Louis Cholesky](https://en.wikipedia.org/wiki/Andr%C3%A9-Louis_Cholesky). The formula is:

$$
A =
LL^{T}
$$

For rough sense, let $A$ be

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$


Then we can represent $A$ as 

$$
A = LL^{T} =
\begin{bmatrix}
l_{11} & 0 & 0 \\
l_{21} & l_{22} & 0 \\
l_{31} & l_{32} & l_{33}
\end{bmatrix}
\cdot
\begin{bmatrix}
l_{11} & l_{12} & l_{13} \\
0 & l_{22} & l_{23} \\
0 & 0 & a_{33}
\end{bmatrix} =
\begin{bmatrix}
l_{11}^{2} & l_{21}l_{11} & l_{31}l_{11} \\
l_{21}l_{11} & l_{21}^{2} + l_{22}^{2} & l_{31}l_{21} + l_{32}l_{22} \\
l_{31}l_{11} & l_{31}l_{21} + l_{32}l_{22} & l_{31}^{2} + l_{32}^{2} + l_{33}^2
\end{bmatrix}
$$

The diagonal elements of matrix $L$ can be calculated by the following formulas:

$$
l_{11} = \sqrt{a_{11}}
\quad \quad
l_{22} = \sqrt{a_{22} - l_{21}^{2}}
\quad \quad
l_{33} = \sqrt{a_{33} - (l_{31}^{2} + l_{32}{2})}
$$

And in general, for diagonal elements of the matrix $L$ we have:

$$
l_{kk} =
\sqrt{a_{kk} - \sum_{j = 1}^{k - 1}l_{kj}^{2}}
$$

For the elements below the main diagonal, $l_{ik}$ where $i > k$, the formulas are

$$
l_{21} = \frac{1}{l_{11}}a_{21}
\quad \quad
l_{31} = \frac{1}{l_{11}}a_{31}
\quad \quad
l_{32} = \frac{1}{l_{22}}(a_{32} - l_{31}l_{21})
$$

And the general formula is

$$
l_{ik} =
\frac{1}{l_{kk}}\Big(a_{ik} - \sum_{j = 1}^{k - 1}l_{ij}l_{kj}\Big)
$$

Messy formulas! Consider a numerical example to see what happen under the hood. We have a matrix $A$

$$
A =
\begin{bmatrix}
25 & 15 & -5 \\
15 & 18 & 0 \\
-5 & 0 & 11
\end{bmatrix}
$$

According to the above formulas, let find a lower triangular matrix $L$. We have

$$
l_{11} = \sqrt{a_{11}} = \sqrt{25} = 5
\quad \quad
l_{22} = \sqrt{a_{22} - l_{21}^{2}} = \sqrt{18 - 3^{2}} = 3
\quad \quad
l_{33} = \sqrt{a_{33} - (l_{31}^{2} + l_{32}^{2})} = \sqrt{11 - ((-1)^{2} + 1^{2})} = 3
$$

Seems, we have missing non-diagonal elements, which are

$$
l_{21} = \frac{1}{l_{11}}a_{21} = \frac{1}{5}15 = 3
\quad \quad
l_{31} = \frac{1}{l_{11}}a_{31} = \frac{1}{5}(-5) = -1
\quad \quad
l_{32} = \frac{1}{l_{22}}(a_{32} - l_{31}l_{21}) = \frac{1}{3}(0 - (-1)\cdot 3) = 1
$$

So, our matrix $L$ is

$$
L =
\begin{bmatrix}
5 & 0 & 0 \\
3 & 3 & 0 \\
-1 & 1 & 3
\end{bmatrix}
\quad \quad
L^{T} =
\begin{bmatrix}
5 & 3 & -1 \\
0 & 3 & 1 \\
0 & 0 & 3
\end{bmatrix}
$$

Multiplication of this matrices is up to you.


### QR Decomposition

QR decomposition is another type of matrix factorization, where a given $m \times n$ matrix $A$ is decomposed into 
two matrices, $Q$ which is orthogonal matrix, which in turn means that $QQ^{T} = Q^{T}Q = I$ and the inverse of $Q$ 
equal to its transpose, $Q^{T} = Q^{-1}$, and $R$ which is upper triangular matrix. Hence, the formula is given by

$$
A = 
QR
$$

As $Q$ is an orthogonal matrix, there are three methods to find $Q$, one is [Gramm-Schmidt Process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process), 
second is [Householder Transformation](https://en.wikipedia.org/wiki/Householder_transformation), 
and third is [Givens Rotation](https://en.wikipedia.org/wiki/Givens_rotation). These methods are out of the scope of this blog 
post series and hence I'm going to explain all of them in separate blog posts. Consequently, there is no calculation besides python code in numerical representation section.


### Eigendecomposition

Here is the question. What's the usage of eigenvalues and eigenvectors? Besides other usages, they help us to perform 
matrix decomposition and this decomposition is called eigendecomposition or spectral decomposition. In the case of the 
eigendecomposition, we decompose the initial matrix into the product of its eigenvectors and eigenvalues by the following formula:

$$
A = Q \Lambda Q^{-1}
$$

$A$ is $n\times n$ square matrix, $Q$ is the matrix whose columns are the eigenvectors, which in turn are linearly 
independent and $\Lambda$ is diagonal matrix of eigenvalues of $A$ and these eigenvalues are not necessarily distinct.

To see the detailed steps of this decomposition, consider the aforementioned example of the matrix $A$ 
for which we already found eigenvalues and eigenvectors.

$$
A =
\begin{bmatrix}
2 & 0 & 0 \\
0 & 3 & 4 \\
0 & 4 & 9
\end{bmatrix}
\quad
Q =
\begin{bmatrix}
0 & 1 & 0 \\
-2 & 0 & 1 \\
1 & 0 & 2
\end{bmatrix}
\quad
\Lambda = 
\begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 11
\end{bmatrix}
\quad
Q^{-1} =
\begin{bmatrix}
0 & -0.4 & 0.2 \\
1 & 0 & 0 \\
0 & 0.2 & 0.4
\end{bmatrix}
$$

We have all the matrices and now take matrix multiplication according to the above formula. Particularly, 
multiply $Q$ by $\Lambda$ and by $Q^{-1}$. We have to get original matrix $A$

Furthermore, if matrix $A$ is a real symmetric matrix, then eigendecomposition can be performed by the following formula:

$$
A = Q \Lambda Q^{T}
$$

The only difference between this formula and above formula is that the matrix $A$ is $n\times n$ real symmetric square 
matrix and instead of taking the inverse of eigenvector matrix we take the transpose of it. Moreover, for a real 
symmetric matrix, eigenvectors corresponding to different eigenvalues are orthogonal. Consider the following example:

$$
A =
\begin{bmatrix}
6 & 2 \\
2 & 3
\end{bmatrix}
$$

The matrix is symmetric because of the original matrix equal to its transpose, $A = A^{T}$

Its eigenvalues are $\lambda_{1} = 7$ and $\lambda_{2} = 2$ and corresponding eigenvectors are 

$$
v_{\lambda_{1}} =
\begin{bmatrix}
0.89442719 \\
0.4472136
\end{bmatrix}
\quad
v_{\lambda_{2}} =
\begin{bmatrix}
-0.4472136 \\
0.89442719
\end{bmatrix}
$$

And in this set up, matrices $Q$, $\Lambda$ and $Q^{T}$ are the following:

$$
Q =
\begin{bmatrix}
0.89442719 & -0.4472136 \\
0.4472136 & 0.89442719 \\
\end{bmatrix}
\quad
\Lambda = 
\begin{bmatrix}
7 & 0 \\
0 & 2 \\
\end{bmatrix}
\quad
Q^{T} =
\begin{bmatrix}
0.89442719 & 0.4472136 \\
-0.4472136 & 0.89442719 \\
\end{bmatrix}
$$

Taking matrix product gives initial matrix $A$.

To verify all of this calculation see Python code below.

**Eigendecomposition cannot be used for non-square matrices. Below, we will see the Singular Value Decomposition (SVD) which is another way of decomposing matrices. The advantage of the SVD is that you can use it also with non-square matrices.**


### Singular Value Decomposition

Singular Value Decomposition (SVD) is another way of matrix factorization. It is the generalization of the eigendecomposition. 
In this context, generalization means that eigendecomposition is applicable only for square $n \times n$ matrices, 
while Singular Value Decomposition (SVD) is applicable for any $m \times n$ matrices.

SVD for a $m \times n$ matrix $A$ is computed by the following formula:

$$
A = U \ D \ V^{T}
$$

Where, $U$'s columns are *left singular vectors* of $A$, $V$'s columns are *right singular vectors* of $A$ and $D$  is a 
diagonal matrix, not necessarily square matrix, containing **singular values** of $A$ on main diagonal. 
Singular values of $m \times n$ matrix $A$ are the **square roots of the eigenvalues** of $A^{T}A$, which is a square matrix. 
If our initial matrix $A$ is square or $n \times n$ then singular values **coincide** eigenvalues. 
Moreover, all of these defines the path towards eigendecomposition. Let see how this path is defined.

Matrices, $U$, $D$, and $V$ can be found by transforming $A$ into a square matrix and computing eigenvalues and 
eigenvectors of this transformed matrix. This transformation is done by multiplying $A$ by its transpose $A^{T}$. 
After that, matrices $U$, $D$ and $V$ are the following:

* $U$ corresponds to the eigenvectors of $AA^{T}$

* $V$ corresponds to eigenvectors of $A^{T}A$

* $D$ corresponds to eigenvalues, either $AA^{T}$ or $A^{T}A$, which are the same

Theory almost always seems confusing. Consider a numerical example and Python code below for clarification.

Let our initial matrix $A$ be:

$$
A =
\begin{bmatrix}
0 & 1 & 0 \\
\sqrt{2} & 2 & 0 \\
0 & 1 & 1
\end{bmatrix}
$$

Here, to use SVD first we need to find $AA^{T}$ and $A^{T}A$.

$$
AA^{T} =
\begin{bmatrix}
2 & 2 & 2 \\
2 & 6 & 2 \\
2 & 2 & 2
\end{bmatrix}
\quad
A^{T}A =
\begin{bmatrix}
2 & 2\sqrt{2} & 0 \\
2\sqrt{2} & 6 & 2 \\
0 & 2 & 2
\end{bmatrix}
$$

In the next step, we have to find eigenvalues and eigenvectors for $AA^{T}$ and $A^{T}A$. The characteristic polynomial is

$$
-\lambda^{3} + 10\lambda^2 - 16\lambda
$$

with roots equal to $\lambda_{1} = 8$, $\lambda_{2} = 2$, and $\lambda_{3} = 0$. Note that these eigenvalues are 
the same for the $A^{T}A$. We need singular values which are square root from eigenvalues. 
Let denote them by $\sigma$ such as $\sigma_{1} = \sqrt{8} = 2\sqrt{2}$, $\sigma_{2} = \sqrt{2}$ and $\sigma_{3} = \sqrt{0} = 0$. 
We now can construct diagonal matrix of singular values:

$$
D =
\begin{bmatrix}
2\sqrt{2} & 0 & 0 \\
0 & \sqrt{2} & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

Now we have to find matrices $U$ and $V$. We have everything what we need. First find eigenvectors 
of $AA^{T}$ for $\lambda_{1} = 8$, $\lambda_{2} = 2$, and $\lambda_{3} = 0$, which are the following:

$$
U_{1} =
\begin{bmatrix}
\frac{1}{\sqrt{6}}\\
\frac{2}{\sqrt{6}} \\
\frac{1}{\sqrt{6}}
\end{bmatrix}
\quad
U_{2} =
\begin{bmatrix}
-\frac{1}{\sqrt{3}}\\
\frac{1}{\sqrt{3}} \\
-\frac{1}{\sqrt{3}}
\end{bmatrix}
\quad
U_{3} =
\begin{bmatrix}
\frac{1}{\sqrt{2}}\\
0 \\
-\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

Note that eigenvectors are normalized.

As we have eigenvectors, our $U$ matrix is:

$$
U =
\begin{bmatrix}
\frac{1}{\sqrt{6}} & -\frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}}\\
\frac{2}{\sqrt{6}} & \frac{1}{\sqrt{3}} & 0 \\
\frac{1}{\sqrt{6}} & -\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

In the same fashion, we can find matrix $V$, which is:

$$
V =
\begin{bmatrix}
\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}}\\
\frac{3}{\sqrt{12}} & 0 & -\frac{1}{2} \\
\frac{1}{\sqrt{12}} & -\frac{2}{\sqrt{6}} & \frac{1}{2}
\end{bmatrix}
$$

According to the formula we have

$$
A = U \ D \ V^{T} =
\begin{bmatrix}
\frac{1}{\sqrt{6}} & -\frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}}\\
\frac{2}{\sqrt{6}} & \frac{1}{\sqrt{3}} & 0 \\
\frac{1}{\sqrt{6}} & -\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
\cdot
\begin{bmatrix}
2\sqrt{2} & 0 & 0 \\
0 & \sqrt{2} & 0 \\
0 & 0 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}}\\
\frac{3}{\sqrt{12}} & 0 & -\frac{1}{2} \\
\frac{1}{\sqrt{12}} & -\frac{2}{\sqrt{6}} & \frac{1}{2}
\end{bmatrix}
^{T} = A
$$


### Inverse of a Square Full Rank Matrix

Here, I want to present one more way to find the inverse of a matrix and show you one more usage of eigendecomposition. 
Let's get started. If a matrix $A$ can be eigendecomposed and it has no any eigenvalue equal to zero, 
then this matrix has the inverse and this inverse is given by:

$$
A^{-1} =
Q \Lambda^{-1} Q^{-1}
$$

Matrices, $Q$, and $\Lambda$ are already known for us. Consider an example:

$$
A =
\begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}
$$

Its eigenvalues are $\lambda_{1} = -1$ and $\lambda_{2} = 5$ and eigenvectors are:

$$
v_{\lambda_{1}} =
\begin{bmatrix}
-0.70710678 \\
0.70710678
\end{bmatrix}
\quad
v_{\lambda_{2}} =
\begin{bmatrix}
0.4472136 \\
-0.89442719
\end{bmatrix}
$$

Let calculate the inverse of $A$

$$
A^{-1} = Q \Lambda^{-1} Q^{-1} =
\begin{bmatrix}
-0.70710678 & -0.4472136 \\
0.70710678 & -0.89442719
\end{bmatrix}
\cdot
\begin{bmatrix}
-1 & -0 \\
0 & 0.2
\end{bmatrix}
\cdot
\begin{bmatrix}
-0.94280904 & 0.47140452 \\
-0.74535599 & -0.74535599
\end{bmatrix} =
\begin{bmatrix}
-0.6 & 0.4 \\
0.8 & -0.2
\end{bmatrix}
$$


## Numerical Representation

### Cholesky Decomposition


```python

import numpy as np


A = np.array([[25,15,-5],
              [15,18,0],
              [-5,0,11]])


# Cholesky decomposition, find lower triangular matrix L
L = np.linalg.cholesky(A)

# Take transpose
L_T = np.transpose(L)


# Check if it's correct
A == np.dot(L, L_T)
```

```
array([[ True,  True,  True],
       [ True,  True,  True],
       [ True,  True,  True]])
```



### QR Decomposition


```python

import numpy as np


A = np.array([[12,-51,4],
              [6,167,-68],
              [-4,24,-41]])


# QR decomposition
Q, R = np.linalg.qr(A)


print("Q =", Q, sep='\n')
print()
print("R =", R, sep='\n')
print()

print("A = QR", np.dot(Q,R), sep='\n')
```

```
Q =
[[-0.85714286  0.39428571  0.33142857]
 [-0.42857143 -0.90285714 -0.03428571]
 [ 0.28571429 -0.17142857  0.94285714]]

R =
[[ -14.  -21.   14.]
 [   0. -175.   70.]
 [   0.    0.  -35.]]

A = QR
[[ 12. -51.   4.]
 [  6. 167. -68.]
 [ -4.  24. -41.]]
```



### Eigendecomposition


```python

import numpy as np


# Eigendecomposition for non-symmetric matrix

A = np.array([[2,0,0],
              [0,3,4],
              [0,4,9]])


eigenvalues1, eigenvectors1 = np.linalg.eig(A)


# Form diagonal matrix from eigenvalues
L1 = np.diag(eigenvalues1)


# Separate eigenvector matrix and take its inverse
Q1 = eigenvectors1
inv_Q = np.linalg.inv(Q1)


B = np.dot(np.dot(Q1,L1),inv_Q)


# Check if B equal to A
print("Decomposed matrix B:")
print(B)


# Numpy produces normalized eigenvectors and don't be confused with my calculations above




# Eigendecomposition for symmetric matrix

C = np.array([[6,2],[2,3]])

eigenvalues2, eigenvectors2 = np.linalg.eig(C)

# Eigenvalues
L2 = np.diag(eigenvalues2)

# Eigenvectors
Q2 = eigenvectors2
Q2_T = Q2.T


D = np.dot(np.dot(Q2,L2),Q2.T)


# Check if D equal to C
print()
print("Decomposed matrix D:")
print(D)
```

```
Decomposed matrix B:
[[2. 0. 0.]
 [0. 3. 4.]
 [0. 4. 9.]]

Decomposed matrix D:
[[6. 2.]
 [2. 3.]]
```



### Singular Value Decomposition


```python

import numpy as np

np.set_printoptions(suppress=True) # Suppress scientific notation

A = np.array([[0,1,0],
              [np.sqrt(2),2,0],
              [0,1,1]])


U, D, V = np.linalg.svd(A)
print("U =", U)
print()
print("D =", D)
print()
print("V =", V)

B = np.dot(U, np.dot(np.diag(D), V))
print()
print("B =", B)
```

```
U = [[-0.32099833  0.14524317 -0.93587632]
 [-0.87192053 -0.43111301  0.23215547]
 [-0.36974946  0.8905313   0.26502706]]

D = [2.75398408 1.09310654 0.46977627]

V = [[-0.44774472 -0.8840243  -0.13425984]
 [-0.55775521  0.15876626  0.81467932]
 [ 0.69888038 -0.43965249  0.56415592]]

B = [[ 0.          1.         -0.        ]
 [ 1.41421356  2.          0.        ]
 [ 0.          1.          1.        ]]
```



### Inverse of a Square Full Rank Matrix


```python

import numpy as np


A = np.array([[1,2],
              [4,3]])


# Eigenvalues and Eigenvectors
L, Q = np.linalg.eig(A)

# Diagonal eigenvalues
L = np.diag(L)
# Inverse
inv_L = np.linalg.inv(L)

# Inverse of igenvector matrix
inv_Q = np.linalg.inv(Q)


# Calculate the inverse of A
inv_A = np.dot(Q,np.dot(inv_L,inv_Q))

# Print the inverse
print("The inverse of A is")
print(inv_A)
```

```
The inverse of A is
[[-0.6  0.4]
 [ 0.8 -0.2]]
```




## Conclusion

In conclusion, my aim was to make linear algebra tutorials which are in absence, while learning machine learning or deep learning. 
Particularly, existing materials either are pure mathematics books which cover lots of unnecessary(actually they are necessary) 
things or machine learning books which assume that you already have some linear algebra knowledge. The series starts 
from very basic and at the end explains some advanced topics. I can say that I tried my best to filter the materials 
and only explained the most relevant linear algebra topics for machine learning and deep learning.

Based on my experience, these tutorials are not enough to master the concepts and all intuitions but the journey should 
be continuous. Meaning, that you have to practice more and more.


## References

[Cholesky Decomposition](https://rosettacode.org/wiki/Cholesky_decomposition)

[Matrix Decomposition](https://en.wikipedia.org/wiki/Matrix_decomposition)

[Introduction To Linear Algebra](http://math.mit.edu/~gs/linearalgebra/)
