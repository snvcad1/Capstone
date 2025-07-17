## NOTE: The capstone project was changed from House price prediction to Creditcard fraud detection. This was also communicated to my coordinator due to some hardware constrain. Also, the number of records processed in this code are around 50,000 out of 2,84,807 transactions. Again this was done to due to hardware processing constrains. You can change the data size in the code from 50,000 to the level you like to process.


# Credit-Card-Fraud-Detection-Capstone

## Problem statement
Predict fraudulent credit card transactions with the help of machine learning models.

## Data Summary:
The dataset is taken from the Kaggle website and it has a total of 2,84,807 transactions, out of which 1081 were duplicate and 492 are fraudulent. Since the dataset is highly imbalanced, it needs to be handled before model building.

## Business Problem Overview
Banks goal is protect customer and banking fraud, poses a significant threat. Financial losses, trust and credibility are the concerning issue to both banks and customers.

Credit card fraud detection using machine is a necessity to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping financial institute reduce denials of legitimate transactions.

## Exploratory data analytics (EDA): 

### Data Dictionary
The dataset was download from here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud .

## Configuration 
Please create a data folder within the source code parent folder and download the data file from kaggle link provided above.

### Data Understanding:
The data set includes credit card transactions made by European cardholders. Out of a total of 50,000 transactions, 83 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for 0.16% of the total transactions. The data set has also been modified with Principal Component Analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value 1 in cases of fraud and 0 in others.

All the data columns are numerical columns with no missing values. Around 46 duplicate rows were in the dataset. After reviewing the Class column (which is 0 for non-fraud and 1 for frad), I see that only .16% were classified as fraud. This shows that we have a imbalanced dataset. Related to Time column, cannot find any specific pattern or relationshio to Class column. Due that reason, the Time column was dropped from the dataframe.

to address the imbalanced dataset, I used SMOTE and Random UnderSampling to balance the data. 

# Train/Test Split: 
#### Training on the SMOTE dataset
The data was split for traning and testing. The first steps was to get the baseline models best parameter and score for LogisticRegression, RandomForestClassifier and DecisionTreeClassifier model. Used the best parameter for all the models to compare the model performance in terms of Accuracy, Precision, Recall, F1 Score, and ROC_AUC. For each model the information was captured in a dataframe and show as output as shown below.

Model Comparison (SMOTE):

                 Model  Accuracy  Precision    Recall  F1 Score   ROC AUC
                 
                 ---------------------------------------------------------
0  Logistic Regression  0.967617   0.976502  0.958292  0.967311  0.997549
1        Random Forest  0.999699   0.999399  1.000000  0.999699  0.999999
2         DecisionTree  0.998997   0.998298  0.999699  0.998998  0.998997


Based on the above informatiob, RandomForestClassifier performed the best. I fined tuned the RandomeForestClassifier with different parameters overfall score remained the same.

#### Training on the RandomUnderSAmpler dataset
Repeated the same training on this dataset for learning purpose.

Model Comparison (RandomUnderSampler):

                 Model  Accuracy  Precision    Recall  F1 Score   ROC AUC
0  Logistic Regression  0.882353   0.933333  0.823529  0.875000  0.930796
1        Random Forest  0.911765   1.000000  0.823529  0.903226  0.934256
2         DecisionTree  0.823529   0.789474  0.882353  0.833333  0.823529

Based on the above traning SMOTE was a better fit based on the data size.

### Model-Building/Hyperparameter Tuning: 
We can try different models and fine-tune their hyperparameters until we get the desired level of performance on the given dataset. 

### Final model results:
Will provide additional details once all the model evaluation steps are complete.
##Model Comparison (SMOTE):

 # Model:Random Forest
 # Accuracy: 0.999699   
 # Precision: 0.999399
 # Recall: 1.000000  
 # F1 Score: 0.999699  
 # ROC AUC: 0.999999

 
