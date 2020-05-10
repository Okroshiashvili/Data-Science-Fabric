Title: Integer Sequences in Python
Date: 2020-04-25 03:44
Category: Mathematics
Tags: Number Theory, Python, Sequences
Keywords: integer sequences with python, number theory, plot sequences in python, mathematical plotting in python, fibonacci sequence in python, factorial in python
Author: Nodar Okroshiashvili
Summary: Generate and plot integer sequences in python


In this article, I want to review some integer sequences from [number theory](https://en.wikipedia.org/wiki/Number_theory). 
Particularly, I will review their mathematical formulation, then will write Python functions for them and later on generate 
sequences to visualize them.

In short, **integer sequence** is an ordered list of integers. Generally, in mathematics, the sequence is a collection of objects, 
where repetition is allowed, and object order does not matter. The number of objects a.k.a elements is the length of the sequence. 
The sequence can be finite or infinite, increasing or decreasing, convergent or divergent, bounded or unbounded.

A sequence generally has a **rule**, which indicates how we can find the value of each element. An integer sequence 
can be declared either by the formula for its $n^{th}$ element, or by giving the relationship between its elements. 
Moreover, sequences have the initial condition, which gives us the value of the first few terms of the sequence.


> I'm intentionally not reviewing sequences such as Natural Numbers, Real Numbers, and Prime Numbers.


## Fibonacci Sequence


Let first review the well-known **[Fibonacci Sequence](https://en.wikipedia.org/wiki/Fibonacci_number)**. In the sequence 
each element is the sum of the two preceding elements starting from $0$ and $1$. The sequence is commonly denoted by $F_{n}$ and 
the initial condition is given as: $F_{0} = 0$ and $F_{1} = 1$. The general formula is: $F_{n} = F_{n-1} + F_{n - 2}$ for $n > 1$.



```python

import types

import matplotlib.pyplot as plt
import seaborn as sns
```




```python

# Fibonacci Sequence

def fibonacci(n): 
    if n == 0:
        return 0
    elif n == 1: 
        return 1
    else: 
        return fibonacci(n-1) + fibonacci(n-2) 


fibonacci_sequence = [fibonacci(n) for n in range(6)]

print(fibonacci_sequence)
```

```
[0, 1, 1, 2, 3, 5]
```




```python

sns.set(style="whitegrid")

ax = sns.barplot(x=list(range(6)), y=fibonacci_sequence)
ax.set_title("Fibonacci Sequence")
ax.set_ylabel("Sequence Elements")

plt.plot(fibonacci_sequence, linewidth=4, color='black')
plt.show()
```

![fibonacci_sequence]({static}../../images/Integer_Sequences_figure3_1.png)



## Factorial


The next sequence is **[Factorial](https://en.wikipedia.org/wiki/Factorial)**. Factorial of a positive integer is denoted by $n!$ and is the product 
to all positive integers less or equal to $n$. The formula is: $n! = n \times (n-1) \times (n-2) \times \cdots \times 2 \times 1$ 
For integers $n \geq 1$ we have the following formula: $n! = \prod_{i=1}^{n} i$, which leads us to the following reccurence 
relation: $n! = n \cdot (n-1)!$. The initial condition is that $n!=1$ for $n$ equal to 1 and 0.



```python
def factorial(n):
    if n < 2:
        return 1
    else:
        return n * factorial(n-1)


factorial_sequence = [factorial(n) for n in range(6)]

print(factorial_sequence)
```

```
[1, 1, 2, 6, 24, 120]
```




```python
sns.set(style="whitegrid")

ax = sns.barplot(x=list(range(6)), y=factorial_sequence)
ax.set_title("Factorial")
ax.set_ylabel("n!")
ax.set_xlabel("n")

plt.plot(factorial_sequence, linewidth=4, color='black')
plt.show()
```

![factorial_sequence]({static}../../images/Integer_Sequences_figure5_1.png)



## Padovan Sequence


**[Padovan Sequence](https://en.wikipedia.org/wiki/Padovan_sequence)** is the sequence of integers with initial value given by: 
$P(0) = P(1) = P(2) = 1$. The recurrence relation is defined by $P(n) = P(n - 2) + P(n - 3)$.



```python
def padovan(n):
    if n < 3:
        return 1
    else:
        return padovan(n - 2) + padovan(n - 3)


padovan_sequence = [padovan(i) for i in range(6)]

print(padovan_sequence)
```

```
[1, 1, 1, 2, 2, 3]
```




```python
sns.set(style="whitegrid")

ax = sns.barplot(x=list(range(6)), y=padovan_sequence)
ax.set_title("Padovan Sequence")
ax.set_ylabel("Sequence Elements")
ax.set_xlabel("n")

plt.plot(padovan_sequence, linewidth=4, color='black')
plt.show()
```

![padovan_sequence]({static}../../images/Integer_Sequences_figure7_1.png)



## Perrin Number


The reccurent formula for **[Perrin Sequence](https://en.wikipedia.org/wiki/Perrin_number)** with initial values $P(0) = 3$, $P(1) = 0$, and $P(2) = 2$ 
is defined by $P(n) = P(n - 2) + P(n - 3)$.



```python
def perrin(n):
    if n == 0:
        return 3
    elif n == 1:
        return 0
    elif n == 2:
        return 2
    else:
        return perrin(n - 2) + perrin(n - 3)


perrin_sequence = [perrin(i) for i in range(9)]

print(perrin_sequence)
```

```
[3, 0, 2, 3, 2, 5, 5, 7, 10]
```




```python
sns.set(style="whitegrid")

ax = sns.barplot(x=list(range(9)), y=perrin_sequence)
ax.set_title("Perrin Sequence")
ax.set_ylabel("Sequence Elements")
ax.set_xlabel("n")

plt.plot(perrin_sequence, linewidth=4, color='black')
plt.show()
```

![perrin_sequence]({static}../../images/Integer_Sequences_figure9_1.png)



## Sylvester's sequence


In the **[Sylvester's sequence](https://en.wikipedia.org/wiki/Sylvester%27s_sequence)**, each element is the product of previous terms, plus one. 
Due to this reason, Sylvester's sequence is considered to have doubly exponential growth, when sequence encreases a much higher rate than, say factorial sequence. 
The initial condition for the sequence is $S_{0} = 2$. This is because the product of an empty set is $1$. The sequence is given by: 
$S_{n} = 1 + \prod_{i=0}^{n-1} s_{i}$. Rewriting it into recurrence formula, gives: $S_{i} = S_{i-1}\cdot(S_{i-1} - 1) + 1$.



```python
def sylvester(n):
    if n == 0:
        return 2
    else:
        return sylvester(n - 1)* (sylvester(n - 1) - 1) + 1


sylvester_sequence = [sylvester(i) for i in range(6)]

print(sylvester_sequence)
```

```
[2, 3, 7, 43, 1807, 3263443]
```




```python
sns.set(style="whitegrid")

ax = sns.barplot(x=list(range(6)), y=sylvester_sequence)
ax.set_title("Sylvester Sequence")
ax.set_ylabel("Sequence Elements")
ax.set_xlabel("n")

plt.plot(sylvester_sequence, linewidth=4, color='black')
plt.show()
```

![sylvester_sequence]({static}../../images/Integer_Sequences_figure11_1.png)



## Fermat number


**[Fermat number or Fermat sequence](https://en.wikipedia.org/wiki/Fermat_number)**, named after well-known Pierre de Fermat, forms the sequence 
of the following form: $F_{n} = 2^{2^{n}} + 1$ with the condition that $n \geq 0$. The recurrence relation is given by: $F_{n} = (F_{n-1} - 1)^{2} + 1$. 
Note that, both formulae are valid. However, I use the first to implement it in Python.



```python
def fermat(n):
    return 2**(2**n) + 1


fermat_sequence = [fermat(i) for i in range(6)]

print(fermat_sequence)
```

```
[3, 5, 17, 257, 65537, 4294967297]
```




```python
sns.set(style="whitegrid")

ax = sns.barplot(x=list(range(6)), y=fermat_sequence)
ax.set_title("Fermat Sequence")
ax.set_ylabel("Sequence Elements")
ax.set_xlabel("n")

plt.plot(fermat_sequence, linewidth=4, color='black')
plt.show()
```

![fermat_sequence]({static}../../images/Integer_Sequences_figure13_1.png)



## Conclusion

In this post, I tried to review some integer sequences and implemented them in Python. The analysis of these sequences is a matter of another blog post. 
However, looking at the plots we can compare these sequences at least with their growth rate. The Fermat sequence has a pretty much higher growth rate 
than other sequences. Calculating $n^{th}$ term of these sequences can be quite challenging and causes [stack overflow](https://en.wikipedia.org/wiki/Stack_overflow). 
One solution can be [trampolining](https://en.wikipedia.org/wiki/Trampoline_(computing)). Try to implement these sequences with trampolining and share your 
insights below in the comments.



## References

- [Sequences](https://www.mathsisfun.com/algebra/sequences-series.html)

- [Sequence](https://en.wikipedia.org/wiki/Sequence)

- [List of OEIS sequences](https://en.wikipedia.org/wiki/List_of_OEIS_sequences)
