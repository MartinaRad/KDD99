# KDD99

Comparing two machine learning algorithms for processing the KDD Cup 99 Dataset.

Dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html


**1. Preprocessing**

Data is downloaded and loaded. Columns are named and the different outcomes are labeled. 
**
2. Analyzing the dataset**

A pie chart is ploted (pie.jpg) to show the percentage of normal connections and attacks.
A matrix is ploted to show correlation between attributes (matrix.jpg).

**3. Random Forest**

Data is devided in train and test sets. A random forest algorithm runs first the train data. After that with a timer the algorithm is tested for accuracy and speed.
 - sklearn package is used
**
4. Neural Network**

A neural network is created. Data is devided again in train and test sets. Algorithm is trained and tested also for speed and accuracy. 
 - tensorflow is used
**
5. Comparison**

A bar chart is created to compare accuracy and speed of testing in both algorithms.
(comparison.jpg)
