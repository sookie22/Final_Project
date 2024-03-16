# Final_Project

## Dataset

We have selected red wine quality csv dataset from Kaggle.   
https://www.kaggle.com/datasets/yasserh/wine-quality-dataset    

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
1. Import packages and import input data into spark db  
2. Convert spark table to pandas dataframe and data clean-up. Remove any rows with missing value.  
  2.1 Check bias in the dataset and classifications.    
  2.2 Check correlations of all features.  

   ![image](https://github.com/sookie22/Final_Project/assets/10916160/1d9baa70-d2b1-49e6-85c9-c6400222dcad)
   

3. Normalize and standardise dataframe.  
4. Apply machine learning models.  
  4.1 Random forest model.
      Random forest is very versatilite machine learning model. It can be used for both regression and classification tasks, and it's also easy to view the relative importance it assigns to the input features.
      It has easy-to-understand hyperparameters and classifier doesn't overfit with enough trees.
      Alcohol quality depends on 11 input features. But which features are dominant in deciding wine quality is as follows.

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/82c7c6da-83da-4089-930c-a551c6b33373)

      This model uses test_size 0.25. The accuracy of this model is 82%
      ![image](https://github.com/sookie22/Final_Project/assets/10916160/10031233-48d6-41b0-847d-6e99702107e3)  


   4.1.1 Optimization
      The test size has been reduced to 0.2. The accuracy has increased to 74%.

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/59c9b8e0-0cc1-42df-9b94-f99b95708f5f)  

   
   4.2 Linear regression
      Linear regression is a statistical modeling process that compares the relationship between two variables, which are usually independent or explanatory variables and dependent variables.
      It is simple to implement and easier to interpret the output coefficients.
      For the first model, quality has been converted to category type assuming wine quality id good when quality rating is between 6-8 and bad when quality rating is below 6.

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/6ab3de2f-2847-4279-a57c-288ba5687c33)

      The accuracy of this model is 74%
   
      ![image](https://github.com/sookie22/Final_Project/assets/10916160/e58f6dda-05b7-41f6-bd26-42614a24a27a)

      ????

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/f29496d2-0395-4907-a27a-4db6f1a96c04)

      ?????

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/aaa6f65d-425a-4588-bb31-be182384fa91)

   4.3 Decision Tree
   




      

## Different models used  
1. Random Forest  
2. Dimension reduction neural network  
3. Linear regression  
4. Decision tree  

## Conclusion  




