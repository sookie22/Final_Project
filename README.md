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
      From this matrix we can see that pH, fixed acidity, citric acid and density are highly correlated.
      Total sulfur dioxide and free sulfur dioxide and very highly correlated.
      These highly correlated columns can be used for dimension reduction to fine tune accuracy.
   
   ![image](https://github.com/sookie22/Final_Project/assets/10916160/1d9baa70-d2b1-49e6-85c9-c6400222dcad)
   

4. Normalize and standardise dataframe.  
5. Apply machine learning models.  
  4.1 Random forest model.
      Random forest is very versatilite machine learning model. It can be used for both regression and classification tasks, and it's also easy to view the relative importance it assigns to the input features.
      It has easy-to-understand hyperparameters and classifier doesn't overfit with enough trees.
      Alcohol quality depends on 11 input features. But which features are dominant in deciding wine quality is as follows.

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/82c7c6da-83da-4089-930c-a551c6b33373)

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

   
  4.2 Linear regression
      Linear regression is a statistical modeling process that compares the relationship between two variables, which are usually independent or explanatory variables and dependent variables.
      It is simple to implement and easier to interpret the output coefficients.
      For the first model, quality has been converted to category type assuming wine quality id good when quality rating is between 6-8 and bad when quality rating is below 6.    

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/d46f81c1-ae50-488d-94f8-5f231a49b06d)  
 

      The accuracy of this model is 74%   
   
      ![image](https://github.com/sookie22/Final_Project/assets/10916160/a41d6001-b826-4ba9-8967-0154df7a3a99)  
  

      ????   

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/f29496d2-0395-4907-a27a-4db6f1a96c04)   

      ?????   

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/aaa6f65d-425a-4588-bb31-be182384fa91)   

  4.3 Decision Tree
      Decision tree is a non-parametric supervised learning algorithm and is hierarchical in structure. Like a tree, it has root nodes, branches, internal nodes, and leaf nodes.
      It divides the data space into sections, and producing decision rules that help in coming up with a prediction or a label. Decision trees are good for non linear predictions.

      For this model, two least important features in the input dataset have been removed.   
      ![image](https://github.com/sookie22/Final_Project/assets/10916160/7531d212-0376-4a6d-8c2d-c15fd2f05ca8)   

      Accuracy of prediction has improved to 75.3%   
      ![image](https://github.com/sookie22/Final_Project/assets/10916160/56285a15-0c63-4536-b372-3d22ac787bd0)   

      ?????   
      ![image](https://github.com/sookie22/Final_Project/assets/10916160/5b75be2e-be63-40db-8dd5-52e3ca5b49cb)   

      ??????   
      ![image](https://github.com/sookie22/Final_Project/assets/10916160/be04e066-b181-4b94-8fc4-2adab8c1717f)    

 4.3.1 Optimization 1
      Hyperparameter tuning is applied to the model. Following param grid is used.   

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/d7b48305-9da4-4efd-a201-98570b8556bb)    

      Accuracy is reduced to 73%.   

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/5ac9c2a7-1d2f-4d54-b752-8fae827d6bf7)    

 4.3.2 Optimization 2
      Wine quality has been converted to category type assuming wine quality id good when quality rating is over 7 and bad when quality rating is below 7. This is very narrow classification.

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/abcc5b79-4d34-44fc-8025-97baae005388)     

      Accuracy has improved drastically to 87.2%   

      ![image](https://github.com/sookie22/Final_Project/assets/10916160/2093e4fb-af5c-4dac-8246-a9aa068a3966)    

 4.4 Tensoflow neural network
     Tensoflow is an open-source machine learning library used for deep learning neural network models. 
    


   



   




      


## Summary  

![image](https://github.com/sookie22/Final_Project/assets/10916160/a0e2ae05-0d64-4761-87f4-0dd752728680)  




## Conclusion   
Decision tree with narrow classification has best accuracy followed by Random forest model with gridsearch classifier. 



