**Credit-Card-Fraud-Detection-Capstone**

**Problem statement**
Predict fraudulent credit card transactions with the help of machine learning models.

The dataset is taken from the Kaggle website and it has a total of 2,84,807 transactions, out of which 1081 were duplicate and 492 are fraudulent. Since the dataset is highly imbalanced, it needs to be handled before model building.

**Business Problem Overview**
Banks goal is protect customer and banking fraud, poses a significant threat. Financial losses, trust and credibility are the concerning issue to both banks and customers.

Credit card fraud detection using machine is a necessity to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping financial institute reduce denials of legitimate transactions.

# Exploratory data analytics (EDA): 

**Data Dictionary**
The dataset was download from here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Data Understanding: **
The data set includes credit card transactions made by European cardholders. Out of a total of 2,84,807 transactions, 492 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for 0.16% of the total transactions. The data set has also been modified with Principal Component Analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value 1 in cases of fraud and 0 in others.

All the data columns are numerical columns with no missing values. Around 1081 duplicate rows were in the dataset. After reviewing the Class column (which is 0 for non-fraud and 1 for frad), I see that only 473 row of data were classified as fraud (0.16%). This should that we have a imbalanced dataset. Related to Time column, cannot find any specific pattern or relationshio to Class column. Due that reason, the Time column was dropped from the dataframe.

As part of the analysis process, idenfied skweness and addressed it using PowerTrasformer and displayed the data visually to show the skweness and after the skweness was addressed. Checkedfor skewness in the data and tried to mitigate it, as it might cause problems during the model-building phase. But at this point the data is still imbalanced.

### Train/Test Split: 
Now we are familiar with the train/test split, which we can perform in order to check the performance of our models with unseen data. Here, for validation, we can use the k-fold cross-validation method. We need to choose an appropriate k value so that the minority class is correctly represented in the test folds.

Model-Building/Hyperparameter Tuning: 
This is the final step at which I can try different models and fine-tune their hyperparameters until we get the desired level of performance on the given dataset. We should try and see if we get a better model by the various sampling techniques.

Model Evaluation: 
We need to evaluate the models using appropriate evaluation metrics. Note that since the data is imbalanced it is is more important to identify which are fraudulent transactions accurately than the non-fraudulent. We need to choose an appropriate evaluation metric which reflects this business goal.
