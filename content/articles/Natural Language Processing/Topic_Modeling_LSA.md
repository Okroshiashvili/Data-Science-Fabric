Title: Topic Modeling in Python: Latent Semantic Analysis
Date: 2020-06-22 03:44
Category: Natural Language Processing
Tags: Topic Modeling, NLP, LSA
Keywords: topic modeling, nlp, natural language modeling, python, latent semantic analysis, probabilistic latent semantic analysis, topic modeling in nltk, topic modeling in gensim
Author: Nodar Okroshiashvili
Summary: Topic modeling with NLTK and Gensim


Recently, I've started digging deeper into Natural Language Processing. During week or so, I took several 
MOOCs and read blogs. I tried some coding and now with this blog, I want to share my experience. With this blog series, 
I want to review [Topic Modeling](https://en.wikipedia.org/wiki/Topic_model). Particularly, [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis), 
[Non-Negative Matrix Factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization), and 
[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation). 
For the sake of brevity, these series will include three successive parts, reviewing each technique in each part.


This is the first part of this series, and here I want to discuss Latent Semantic Analysis, a.k.a LSA. 
I implement it using NLTK and Gensim. 
Before going into details of this algorithm, let say what topic modeling is and why we should care.


Every day, a massive amount of data is collected. Most of this data is a text, such as an email, blog post, books, 
articles. These all are unstructured data, and unstructured data means that there is no means by which the computer 
understands the semantic meaning of the words in the text. As the volume of the data increases, it's become difficult to 
search in these documents for a piece of particular information. Moreover, as time passes, humans are getting lazy 
to read and review tons of articles, books, and blogs to find the desired information. Here comes the topic modeling.

> Topic Modeling is an unsupervised machine learning technique that provides a simple way to analyze a large amount of 
> unstructured data and extract cluster of words that frequently occur together, or are connected to 
> each other in some statistically meaningful way ([Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics)). 
> Topic modeling strives to find hidden semantic structures in the text.


That's sort of "official" definition. *In my words*, topic modeling is a dimensionality reduction technique, where we express each 
document in terms of topics, that in turn, are in the lower dimension. That means that we have thousands of thousand documents, 
and we express them by using hundreds of words. Does it make sense? Imagine we perform PCA (Principal Component Analysis) 
for the text data. Here, the document is an independent text consisting of several sentences such as an email, a review, a tweet, and so on. 
The collection of documents is a corpus. Topic modeling is to find latent factors that exist in our text. 
To make it even concise, by doing topic modeling, we try to answer the following question: **What is this document about?** 
For example, we have millions of reviews of one particular hotel. Only two or three sentences telling about the topics of these 
reviews is much preferable than reading all the reviews. That's it!


Here comes "why we should care?" question. In natural language processing and general in information retrieval, 
topic modeling plays an essential role due to the reason mentioned above. Additionally, recent advancements in 
machine learning and increasing demand for analytical solutions give rise of the usage of topic modeling for 
dimensionality reduction in documents, recommendation systems, clustering documents, classification of documents, and many more.


Now, it's time for **Latent Semantic Analysis**. It is one of the foundational algorithms in topic modeling. 
It assumes that each document in our corpus consists of a mixture of topics, and each topic contains a collection of words. 
Based on this assumption, Latent Semantic Analysis takes **Document-Word Matrix** and employees [Singular Value Decomposion](https://dsfabric.org/advance-linear-algebra-with-python-part-ii) 
to decompose it into the product of two matrices. The first is **Document-Topic** matrix, and the second is **Topic-Word** matrix. 
The computers cannot understand the text, and it's even unimaginable for them how to decompose a text into a product of 
two matrices that intuitively has numbers. That's why we build Document-Word matrix, and then computer finds Document-Topic 
and Topic-Word matrices for us.

Let review each matrix in a detailed manner. First of all, we need to build a Document-Word matrix. To do so, we build vocabulary from our corpus. 
This vocabulary is a collection of all unique words coming from our corpus. If we have one document corpus and our document is 
"I like that movie" then our vocabulary will be ```{"I", "like", "that", "movie"}```. After we have the vocabulary, we apply 
[CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) or 
[TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer) 
to our corpus in order to have a Document-Word matrix. CountVectorizer captures how often a particular word occurs in a document, 
and TF-IDF captures how often a word occurs in a document as well as in the entire corpus.

For example, if we have two document corpus:

```
corpus = [
    "This is the first document.", 
    "This document is the second document."
]
```

Then, based on this corpus, our vocabulary will be:

```
vocab = {
    "document", "first", "is", 
    "second", "the", "this"
}
```

Applying CountVecorizer to our corpus, will give the following **Document-Word** matrix:

$$
Document-Word =
\begin{bmatrix}
    1 & 1 & 1 & 0 & 1 & 1 \\
    2 & 0 & 1 & 1 & 1 & 1
\end{bmatrix}
$$

Columns of this matrix represent words in our vocabulary, and rows represent our documents. Every individual element in 
this matrix indicates how often a specific word occurs within a particular document. For instance, the first row of 
this matrix shows how often each vocabulary word occurs in the first document. Hence, $D_{ij}$ element of 
Document-Word matrix will give the frequency of occurrence for word $j$ in a document $i$.


Having a Document-Word matrix, LSA tries to build [low-rank approximation](http://theory.stanford.edu/~tim/s15/l/l9.pdf) of that matrix. 
This will be achieved by using Singular Value Decomposition. Here, rank refers to the number of topics. For more about matrix decompositions, 
[read here](https://dsfabric.org/advance-linear-algebra-with-python-part-ii).


Decomposition process of Document-Word matrix follows the following formula: 

$$
DW_{n \times k} = DT_{n \times t} \cdot TW_{t \times k}
$$

Where $n$ is the number of documents, $k$ is the number of words in the vocabulary, and $t$ is the number of topics. 
$t$ is the hyperparameter, and we have to set it manually beforehand. Consequently, the number of topics is far fewer 
than the number of documents. That's why we can consider topic modeling as a dimensionality reduction.


Before diving into practical implementation, let review the other two matrices on the right-hand side of our main equation. 
The first is $DT$, Document-Topic matrix. It contains topics for every document. The second, $TW$, is the Topic-Word matrix. 
This matrix contains words for each topic. Seems clumsy? Let us consider an example, and things will become much succinct. 

For a large corpus, we will have a large vocabulary and, consequently, a large $DW$ matrix. For example, 

$$
DW =
\begin{bmatrix}
    D_{11} & D_{12} & \cdots & D_{1k} \\
    D_{21} & D_{22} & \cdots & D_{2k} \\
    \vdots & \vdots & \boldsymbol{D_{ij}} & \vdots \\
    D_{n1} & D_{n2} & \cdots & D_{nk}
\end{bmatrix}
$$

$$
DT =
\begin{bmatrix}
    U_{11} & U_{12} & \cdots & U_{1t} \\
    \vdots & \vdots & \vdots & \vdots \\
    \boldsymbol{U_{i1}} & \boldsymbol{U_{i2}} & \boldsymbol{\cdots} & \boldsymbol{U_{it}} \\
    \vdots & \vdots & \vdots & \vdots \\
    U_{n1} & U_{n2} & \cdots & U_{nt}
\end{bmatrix}
\quad
TW =
\begin{bmatrix}
    M_{11} & M_{21} & \cdots & \boldsymbol{M_{j1}} & \cdots & M_{t1} \\
    M_{12} & M_{22} & \cdots & \boldsymbol{M_{j2}} & \cdots & M_{t2} \\
    \cdots & \cdots & \cdots & \boldsymbol{\cdots} & \cdots & \cdots \\
    M_{1t} & M_{2t} & \cdots & \boldsymbol{M_{jt}} & \cdots & M_{tk}
\end{bmatrix}
$$

The columns of the $DT$ matrix represents topics and rows are documents. One row of this matrix $\begin{bmatrix} U_{i1} & U_{i2} & \cdots & U_{it} \end{bmatrix}$  
gives us the value for all of the topics associated with a particular document. 
The rows of the $TW$ matrix are topics, and columns are words. One column of this matrix $\begin{bmatrix}{M_{j1}} \\ M_{j2} \\ \vdots \\ M_{jt}\end{bmatrix}$ 
gives us a value for all of the words associated with each of these topics. So, when we multiply these row and column vector, 
we will get back our original word $D_{ij}$ from the $DW$ matrix.


## Practical Implementation


For the applied part, I use [AG's News Topic Classification Dataset](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv). 
The dataset contains 4 classes or 4 topics, such as **World**, **Sports**, **Business**, and **Sci/Tech**. I'd prefer this dataset due to its simplicity.

Let set all the necessary imports


```python

import re
from pprint import pprint

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.corpora import Dictionary
from gensim.models import LsiModel
```




```python
# # If you already have NLTK and WordNetLemmatizer DON'T run this line

# nltk.download('wordnet')
```



To download data, you can visit the link above and then load it by using.


```python
ag_news = pd.read_csv('data/train.csv', names=['Label', 'Title', 'Text'])

ag_news.head()
```

```
   Label                                              Title  \
0      3  Wall St. Bears Claw Back Into the Black (Reuters)
1      3  Carlyle Looks Toward Commercial Aerospace (Reu...
2      3    Oil and Economy Cloud Stocks' Outlook (Reuters)
3      3  Iraq Halts Oil Exports from Main Southern Pipe...
4      3  Oil prices soar to all-time record, posing new...

                                                Text
0  Reuters - Short-sellers, Wall Street's dwindli...
1  Reuters - Private investment firm Carlyle Grou...
2  Reuters - Soaring crude prices plus worries\ab...
3  Reuters - Authorities have halted oil export\f...
4  AFP - Tearaway world oil prices, toppling reco...
```



The dataset has a separate training and testing parts. I only use ```train.csv``` for the expositional purposes and my machine's limited capability.

In the dataset we have 3 columns and 120K training points.


```python
ag_news.shape
```

```
(120000, 3)
```



Now, let create a document list or document matrix. The first element in this list is the first document, the second element is the second document, 
and so on. While doing so, I will also do some pre-processing, such as removing stop words and lemmatize sentences.


```python

text = ag_news['Text']

text.head()
```

```
0    Reuters - Short-sellers, Wall Street's dwindli...
1    Reuters - Private investment firm Carlyle Grou...
2    Reuters - Soaring crude prices plus worries\ab...
3    Reuters - Authorities have halted oil export\f...
4    AFP - Tearaway world oil prices, toppling reco...
Name: Text, dtype: object
```




```python

documents_list = []

for line in text:
    sentence = line.strip()
    new_sentence = re.sub(r"\d","", sentence)
    
    documents_list.append(sentence)

print(documents_list[0])
```

```
Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics,
are seeing green again.
```



We need to remove stopwords and lemmatize each sentence


```python

# Standard stop words in NLTK
stop_words = set(stopwords.words('english'))

# Add some extra characters and words as stop words
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '#', '...', 
                   '--', "'s", 'also', '&', '-', '–', '=', 'known', 'mi', 'km', '$'])
```




```python

# Here is pre-processed documents
processed_list = []

# Lemmatizer
lemmatizer = WordNetLemmatizer()


for doc in documents_list:
    tokens = word_tokenize(doc.lower())

    stopped_tokens = [token for token in tokens if token not in stop_words]
    
    lemmatized_tokens = [lemmatizer.lemmatize(i, pos="n") for i in stopped_tokens]
    
    processed_list.append(lemmatized_tokens)
    
print(processed_list[0])
```

```
['reuters', 'short-sellers', 'wall', 'street', 'dwindling\\band',
'ultra-cynics', 'seeing', 'green']
```



In the loop above, we iterate over documents list, each element is one document, then remove stop words for each sentence and then apply lemmatization. 
After that, I append ```processed_list``` with pre-processed results, which again is the documents list.

As we already pre-processed our data, consequently we have data ready to build vocabulary. 
Vocabulary is a dictionary containing all unique words from ```processed_list``` with 
appropriate index. After doing this, we have to build **Bag-of-Words** in order to build 
**Document-Word** matrix.


```python

word_dictionary = Dictionary(processed_list)

print(word_dictionary)
```

```
Dictionary(85175 unique tokens: ['dwindling\\band', 'green',
'reuters', 'seeing', 'short-sellers']...)
```




```python

document_word_matrix = [word_dictionary.doc2bow(document) for document in processed_list]
```



The function ```doc2bow()``` simply counts the number of occurrences of each distinct word, converts the word to its integer word id 
and returns the result as a sparse vector.

The structure of ```document_word_matrix``` is a list of lists, where the first list corresponds to a list of documents, 
the inner list is a list of words in each document.

Create the model. I predefine the number of topics. We already know the number of topics. However, this can be considered as hyper-parameter, requiring fine-tuning.


```python

NUM_TOPICS = 4

lsi_model = LsiModel(corpus=document_word_matrix, num_topics=NUM_TOPICS, id2word=word_dictionary)
```



Let check the topics.


```python

lsi_topics = lsi_model.show_topics(num_topics=NUM_TOPICS, formatted=False)

pprint(lsi_topics)
```

```
[(0,
  [("''", 0.5525950728710217),
   ('gt', 0.5017034951239502),
   ('lt', 0.5012421420448145),
   ('http', 0.13235789377032975),
   ('reuters', 0.12738544638608676),
   ('href=', 0.1165203671943378),
   ('/a', 0.11609987400732807),
   ('new', 0.10805850440429328),
   ('said', 0.10550031843615736),
   ('//www.investor.reuters.com/fullquote.aspx',
0.09462537588321375)]),
 (1,
  [('39', 0.7641971094809981),
   ('said', 0.22375199975904078),
   ('new', 0.17201656666498566),
   ('quot', 0.1440324334139677),
   ("''", -0.12974757461304182),
   ('lt', -0.1258484664379667),
   ('gt', -0.12559624751229786),
   ('year', 0.102034017714784),
   ('company', 0.0967020195333899),
   ('u', 0.08643385295026677)]),
 (2,
  [('39', -0.5823634301756129),
   ('said', 0.37304053806244963),
   ('new', 0.31226846946427567),
   ('reuters', 0.28020531171237956),
   ('york', 0.15097794586351654),
   ('u.s.', 0.12230606105242761),
   ("''", -0.12071594616719289),
   ('gt', -0.11465765558602346),
   ('lt', -0.11451351758143592),
   ('ap', 0.10568473511696055)]),
 (3,
  [('new', 0.6280420959343559),
   ('said', -0.46484918292452715),
   ('quot', -0.3648027843052166),
   ('york', 0.2937236162058658),
   ("''", -0.09895153658039355),
   ('price', 0.09543427194214693),
   ('39', 0.087455449368572),
   ('oil', 0.08682102624838821),
   ('official', -0.08501911286445321),
   ('stock', 0.07554676848568334)])]
```



```.show_topics()``` method shows the most contributing words (both positively and negatively) for each of the first *n* the number of topics. 
Observing the output, we see that the model did not return well-defined topics. There are some junk words. This is due to the pre-processing and 
indicates the need for some extra pre-processing steps. As you already know the workflow, doing proper pre-processing is up to you.



## Conclusion

To conclude, LSA is much like principal component analysis for the unstructured data and is based on matrix decomposion technique. For creating 
**Document-Word Matrix** I used simple CountVectorizer. You can try **TF-IDF** and see the result. This is the first part of this series. In the 
following blogs, I will review **Latent Dirichlet Allocation** and **Non-Negative Matrix Factorization** and will show thier practical implementation.


## References

- [Topic_Model](https://en.wikipedia.org/wiki/Topic_model)

- [Document_Term_Matrix](https://en.wikipedia.org/wiki/Document-term_matrix)

- [Latent_Semantic_Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
