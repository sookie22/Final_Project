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
      Random forest is very versatilite machine learning model. It can be used for both regression and classification tasks, and it's also easy to view the relative importance it assigns to the input features.
      It has easy-to-understand hyperparameters and classifier doesn't overfit with enough trees.
      Alcohol quality depends on 11 input features. But which features are dominant in deciding wine quality is as follows.  
![image](https://github.com/sookie22/Final_Project/assets/143486132/e6409c01-f0e8-4b48-883b-a55cda72ed59)

      This model uses test_size 0.25. The accuracy of this model is 72%  
      ![image](https://github.com/sookie22/Final_Project/assets/10916160/10031233-48d6-41b0-847d-6e99702107e3)  


   4.1.1 Optimization 1  
      The test size has been reduced to 0.2. The accuracy has increased to 74%.  

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/59c9b8e0-0cc1-42df-9b94-f99b95708f5f)  

   4.1.2 Optimization 2  
      Gridsearch classifier is used to improve the accuracy. GridSearchCV is the process of performing hyperparameter tuning in order to determine the optimal values for a given model.
      Parameter grid for the classifier is defined as follows.   

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/48990eb4-af86-404f-98fa-92436fee3acd)

      Accuracy is significantly improved to 82%.    
   
      ![image](https://github.com/sookie22/Final_Project/assets/10916160/eafae585-176d-4db5-b8ca-218c01a53f6c)  

   
  4.2 Linear regression vs. logistic regression 
    4.2.1 Linear Regression
      Linear regression is a statistical modeling process that compares the relationship between two variables, which are usually independent or explanatory variables and dependent variables.
      It is simple to implement and easier to interpret the output coefficients.
      For the first model, quality has been converted to category type assuming wine quality id good when quality rating is between 6-8 and bad when quality rating is below 6.    
      R-squared: 0.4032
      ![image](https://github.com/sookie22/Final_Project/assets/145446182/d28cb3bb-3cd1-475d-9d6c-27bc580822fc)  
 
    4.2.2 Optimization: Utilize categorization Good (6 and above) and Bad (below 6)
      The accuracy of this model is 74%
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/15ea3b12-1d2b-4b14-ae62-37cc388a7ef5)  
  

      ROC Curve
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/c876a80d-51d2-4434-8b1f-1cfd8d81164e)   

      Precision-Recall Curve  
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/f9e10cf9-083a-4f0c-b2e8-a2ec7a90ef71)   

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
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/303031ad-01e7-4d9f-9db6-6fc676709125)     

    ROC Curve
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/738ed564-79ab-4948-a588-5e2b94d958e5)    

    Precision-Recall Curve
  ![image](https://github.com/sookie22/Final_Project/assets/145446182/af558790-a313-4c4c-a1b8-68e2dfbde3a0)

 4.4 Tensoflow neural network   
     Tensoflow is an open-source machine learning library used for deep learning neural network models. 
     The target variable y has integer values that represent different categories.

     ![image](https://github.com/sookie22/Final_Project/assets/10916160/a3addbce-d66a-4eee-b74b-a6bbcde3610c)   

     This should be converted to a matrix.
     ![image](https://github.com/sookie22/Final_Project/assets/10916160/15862f4b-b20b-4eff-9b5f-5c6a64988120)  

     Tensorflow layers are defined as follows.

     ![image](https://github.com/sookie22/Final_Project/assets/10916160/961f02d6-a449-4242-ac05-7ad6c672909d)  

     Accuracy is 60%
     ![image](https://github.com/sookie22/Final_Project/assets/10916160/c1d0c013-16b9-4431-9648-607a6f67c83e)  

4.4.1 Optimization 1   
     The loss function is changed to mean squared error.

     ![image](https://github.com/sookie22/Final_Project/assets/10916160/79415653-acdc-4f88-9f69-6510c3a07e78)  

     Accuracy is reduced to 57.9%

     ![image](https://github.com/sookie22/Final_Project/assets/10916160/fc597fae-aaba-4b30-8c7d-d90cd5fc3914)  
     


## Summary  

![image](https://github.com/sookie22/Final_Project/assets/10916160/a0e2ae05-0d64-4761-87f4-0dd752728680)  

## Conclusion   
Decision tree with narrow classification has best accuracy followed by Random forest model with gridsearch classifier. 



