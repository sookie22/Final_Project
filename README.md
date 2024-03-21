# Final_Project
## Presentation deck: https://docs.google.com/presentation/d/1HkYyFTOZlRZ4I7RjwapBqX095envNZ_g/edit#slide=id.p1

## Dataset

Source Kaggle - Wine quality: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
This dataset contains information about chemical composition of red variants of the Portuguese "Vinho Verde" wine.   

Input variables (based on physicochemical tests):  
1 - fixed acidity  
2 - volatile acidity  
3 - citric acid  
4 - residual sugar  
5 - chlorides  
6 - free sulfur dioxide  
7 - total sulfur dioxide  
8 - density  
9 - pH  
10 - sulphates  
11 - alcohol  
Output variable (based on sensory data):  
12 - quality (score between 0 and 10)  

## Objective  
Build machine learning model to predict wine quality based on its chemical composition.  

## Process Overview  
1. Import packages and import input data into spark db  and convert into Panda dataFrame
2. Data Clean-up  
  2.1 Check for Null and data type inconsistence
  2.2 Check bias in the dataset and classifications.    
  2.3 Run correlation matrix of variables
   ![image](https://github.com/sookie22/Final_Project/assets/143486132/c867b94a-1724-40dc-9bd1-bb14abc087c5)
3. Normalize and Standardization - StandardScaler ().  
4. Machine Learning Models Attempts.  
  4.1 Random forest model.  
     The Random Forest model is a highly versatile machine learning algorithm known for its adaptability to both regression and classification tasks. One of its key strengths lies in its ability to provide insights into the relative importance assigned to different input features. With straightforward hyperparameters, this model is user-friendly and generally resistant to overfitting when a sufficient number of trees are used.
   
In the given dataset, assessing alcohol quality is based on 11 input features. This model enables us to understand which features play the most significant role in determining wine quality.
 
![image](https://github.com/sookie22/Final_Project/assets/143486132/e6409c01-f0e8-4b48-883b-a55cda72ed59)

When run without optimization, the model achieves a predictive accuracy of 72% on the test set.
    
![image](https://github.com/sookie22/Final_Project/assets/144679119/ac026457-6d59-463e-b615-101fbe2a908b)

After optimization through Hyperparameter Tuning using GridSearchCV, the model attains its highest score of 80%, with an accuracy reaching 81%.
   
![image](https://github.com/sookie22/Final_Project/assets/144679119/4e17ac0e-23e2-4584-bfda-d3e3ba0863a3)

![image](https://github.com/sookie22/Final_Project/assets/144679119/4e749a28-0a68-4b11-a36e-02eeb703f55a)

4.2 Linear regression vs. logistic regression

Linear regression is a statistical modeling process that compares the relationship between two variables, which are usually independent or explanatory variables and dependent variables. Logistic regression is a statistical method used for binary classification tasks, where the outcome variable (dependent variable) is categorical with two possible outcomes. As we decide continue models with an evaluation for prediction of all ratings or a categorization in good (6 and above quality rating) and Bad (Below 6 quality rating)

4.2.1 Linear Regression            
R-squared: 0.4032

4.2.2 Optimization: Utilize categorization Good (6 and above) and Bad (below 6)- Logistic Regression

The accuracy of this model is 74%

![image](https://github.com/sookie22/Final_Project/assets/143486132/f072b20d-5ec4-4763-9d48-6116b5e89087)
  
ROC Curve

![image](https://github.com/sookie22/Final_Project/assets/143486132/f75d1129-7de5-46a6-aaa6-ac01ddb04ff4)

Precision-Recall Curve

![image](https://github.com/sookie22/Final_Project/assets/143486132/e4d79cc9-a9c1-4513-9976-9527e5de84b3)


  4.3 Decision Tree  
    4.3.1 Optimization: with categorization (Remove the least important feature)
    For our first optimization, we categorized wine quality into 'good' (quality ≥ 6) and 'bad' (quality < 6) categories. We removed the least important feature, free sulfur dioxide, to simplify the model's decision-making process and focus on attributes with higher significance. 
    This enhanced interpretability and prediction accuracy by reducing noise. The model achieved an initial accuracy of 75.31% on the test set.
    
![image](https://github.com/sookie22/Final_Project/assets/145446182/b7790dda-96bb-49da-b29e-540fa132e521)      

  ROC Curve
  
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/8a8df245-4d76-4867-9daf-bcc7847aa21c)   

  Precesion-Recall Curve
  
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/a0c104c4-8eea-4789-8975-19dd36b7cbf4)    

 4.3.1 Optimization 1  
    Our second optimization involved hyperparameter tuning using GridSearchCV to optimize model performance. 
    By fine-tuning parameters like 'max_depth', 'min_samples_leaf', and 'min_samples_split', we aimed to improve the model's ability to generalize to unseen data. 
    However, despite our efforts, the accuracy slightly reduced from 75.31% to 73% on the test set.
    
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/448b4bd8-d25f-40f5-b2a8-7d5be848887f)    

 4.3.2 Optimization 2   
    For the third optimization, we refined the classification to distinguish 'good' wines (quality ≥ 7) and 'bad' wines (quality < 6), focusing on wines with quality ratings above and below 7. 
    This refinement led to an accuracy increase of 87.19% on the test set.
    While the 87.19% accuracy is promising, it's important to recognize that oversimplified classifications may hinder informed decision-making, potentially affecting industry practices. 
    Thus, we should consider the initial accuracy of 75.31% as a valuable reference point for Decision Tree optimization at this stage.

  ROC Curve
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/738ed564-79ab-4948-a588-5e2b94d958e5)    

  Precision-Recall Curve
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/af558790-a313-4c4c-a1b8-68e2dfbde3a0)

4.4 Tensoflow neural network   

Tensoflow is an open-source machine learning library used for deep learning neural network models. 
The target variable y has integer values that represent different categories.

Neural network model is defined as follows.  

![image](https://github.com/sookie22/Final_Project/assets/10916160/ece58467-e11e-4483-848d-2c729f15dfd4)  

The accuracy is 0. This model fails 100% to predict wine quality. The loss function binary cross-entropy is used for binary classification. wine quality is range of values from 1 to 10.

![image](https://github.com/sookie22/Final_Project/assets/10916160/3d2ba24c-80b5-4970-9e22-7bf91947f1bc)

Optimization 1  
The loss function and metric are changed to mean absolute error, which is more suitable for regression type models.

![image](https://github.com/sookie22/Final_Project/assets/10916160/b9472974-280d-4abc-a096-9d7ddb664e4e)

The MAE for training data set is 0.5. This means that predicted wine quality can be +/- 0.5 range of actual wine quality.
![image](https://github.com/sookie22/Final_Project/assets/10916160/b0a2695a-1679-4111-abb6-a9c62aaff2af)



## Summary  

![image](https://github.com/sookie22/Final_Project/assets/144679119/f81c25e0-01ba-4417-b816-9ee88314d783)


## Conclusion   
Random forest would be the most suitable model to predict wine quality. However, it's crucial to validate the classification process utilised with industry bodies for applicability. 	If applicable, this model offers strategic advantages for wine producers by providing early insights into wine quality. It empowers producers to plan and implement quality control measures effectively, ultimately enhancing production efficiency.




