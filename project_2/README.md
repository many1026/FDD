# Project 2

### Overview

In this project, we'll leverage the techniques reviewed in class to manipulate and extract knowledge from a research-relevant dataset. Throughout the execution, the student will get hands-on experience with different concepts such as:

* Information extraction from NoSQL sources.
* Syntactical feature generation for text models.
* Model training and evaluation.
* Statistical Inference.

At this semester stage, the student should be able to make reasonable decisions under uncertainty or ambiguity---always document what you do. Therefore, the instructions will be open and merely descriptive. It is for the student to determine implementation details. Remember that in class, we've seen that some structures are more efficient than others. In case of doubt, remember:

* If you are thinking iterate, try first vectorize.
* Unique values can be retrieved instantly.

### Project Description

We want to study the relation between syntactical similarities between texts and the likelihood that such similarity might entail the link of question answers. Since this is not a prediction task, we are not very interested in measuring our model's performance but in its ability to consistently capture the strength of such a relation.

#### Data structure

Use the [SQUAD2 dataset](https://rajpurkar.github.io/SQuAD-explorer/)  to generate the following table:

| Jaccard(context_i_line_j, q_i_k)  | Ans |
| ------------- | ------------- |
| .234234  | 1  |
| .72343  | 0  |
| ...  | ...  |

**NOTE 1** the indexes imply that each sentence of every context should be paired against every question. If the given sentence from the context contains the answer to a question, such a pair should mark Ans = 1. In any other case, Ans = 0.

**NOTE 2** If |s_1| denotes the set of all words from string s_1 and |s_2| denotes the set of all words from string s_2. And  ||, &&  denote the mathematical set union and intersection. Then Jaccard(s_1, s_2) = (|s_1| && |s_2|)/(|s_1| || |s_2|).

*Preprocessing*

1. Set all words to lowercase.
2. Remove [stopwords](https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords).
3. Drop words with less than five occurrences.
4. We'll use '.' as a sentence separator.
5. Drop an empty string.

#### Model training

* Take a sample from  the previously generated population using the seed=123454321 (sample size=5%, 1%).

* Use your output variable with the Jaccard index to train the [logistic model seen in class](https://github.com/lromang/FDD_2/blob/main/classes/class_14/logit.py).

* Store the value for theta_0 and theta_1.

#### Model Stability

1. Sample with replacement from your sample population until you generate a new sample of the same size (bootstrap sample)
2. train a new model and register the values for theta_0 and theta_1 (the parameters from your regression model).
3. Repeat the above process 100 times and generate a bootstrap confidence interval of 95% confidence (provide the quantiles that exclude de .025 and .975 extremes of the bootstrap distribution.

#### Deliverables

* A plot of theta_0 and theta_1 bootstrap distribution together with the confidence intervals for sample size: 5% 1%.
* Explain what you observe.
* The code is expected to run in class, if your implementation is slow, try to find bottlenecks, if the model is the culprit, check what happens when you change the parameters of toleration or the learning rate. 
