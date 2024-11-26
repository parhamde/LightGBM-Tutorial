# LightGBM-Tutorial
In this project, I will discuss one of the most successful ML algorithm LightGBM Classifier. LightGBM is a fast, distributed, high performance gradient boosting framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks. It has helped Kagglers win data science competitions. 


So, let's get started.

## **Table of Contents** 

- 1.	Introduction to LightGBM
- 2.	LightGBM intuition
   - 2.1	Leaf-wise tree growth
   - 2.2	Level-wise tree growth
   - 2.3	Important points about tree-growth
- 3.	XGBoost Vs LightGBM
- 4.	LightGBM Parameters
   - 4.1	Control Parameters
   - 4.2	Core Parameters
   - 4.3	Metric Parameter
   - 4.4	IO Parameter
- 5.	LightGBM implementation in Python
   - 5.1	Load packages
   - 5.2	Read dataset
   - 5.3	Preview dataset
   - 5.4	Summary of dataset
   - 5.5	Check the distribution of target variable
   - 5.6	Declare feature vector and target variable
   - 5.7	Split data into training and test set
   - 5.8	LightGBM model development and training
   - 5.9	Model prediction
   - 5.10	Model accuracy
   - 5.11	Compare train and test set accuracy
   - 5.12	Check for overfitting
   - 5.13	Confusion-matrix
   - 5.14	Classification-metrices
- 6.	Results and conclusion
- 7.	LightGBM parameter tuning
   - 7.1	For faster speed
   - 7.2	For better accuracy
   - 7.3	To deal with over-fitting
- 8.	References

## **1. Introduction to LightGBM** 


- [LightGBM](https://github.com/Microsoft/LightGBM) is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

  - Faster training speed and higher efficiency.
  - Lower memory usage.
  - Better accuracy.
  - Support of parallel and GPU learning.
  - Capable of handling large-scale data.
  
  
- At present, decision tree based machine learning algorithms dominate Kaggle competitions. The winning solutions in these competitions have adopted an alogorithm called **XGBoost**. 

- A couple of years ago, Microsoft announced its gradient boosting framework LightGBM. Nowadays, it steals the spotlight in gradient boosting machines. Kagglers start to use LightGBM more than XGBoost. LightGBM is 6 times faster than XGBoost. 

- LightGBM is a relatively new algorithm and have long list of parameters given in the [LightGBM documentation](https://github.com/microsoft/LightGBM),

- The size of dataset is increasing rapidly. It is become very difficult for traditional data science algorithms to give accurate results. Light GBM is prefixed as **Light** because of its high speed. **LightGBM can handle the large size of data and takes lower memory to run**. 

- Another reason why Light GBM is so popular is because it focuses on accuracy of results. LGBM also supports GPU learning and thus data scientists are widely using LGBM for data science application development.

- It is not advisable to use LGBM on small datasets. LightGBM is sensitive to overfitting and can easily overfit small data.

## **2. LightGBM intuition** 


- LightGBM is a gradient boosting framework that uses tree based learning algorithm.


- LightGBM documentation states that -

 `LightGBM grows tree vertically while other tree based learning algorithms grow trees horizontally. 
 It means that LightGBM grows tree leaf-wise while other algorithms grow level-wise. It will choose 
 the leaf with max delta loss to grow. When growing the same leaf, leaf-wise algorithm can reduce more 
 loss than a level-wise algorithm.`
 
 
 - So, we need to understand the distinction between leaf-wise tree growth and level-wise tree growth.

### **2.1 Leaf-wise tree growth** 


- Leaf-wise tree growth can best be explained with the following visual -
(![Leaf-wise tree growth](https://i.sstatic.net/YOE9y.png))