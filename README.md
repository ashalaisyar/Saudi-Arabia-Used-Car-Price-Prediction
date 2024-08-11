# **Regression Model Machine Learning for Predicting Used Cars Prices in Saudi Arabia**

**Tools:** Python<br>
**Dataset:** Saudi Arabia Used Cars

**Outline:**

1. Business Problem Understanding
2. Data Understanding
3. Data Cleaning
4. Feature Selection
5. Feature Engineering
6. Analytics
    - Model Benchmarking and Cross Validation
    - Hyperparameter Tunning
    - Model Evaluation
    - Compare Actual and Predicted Value (Model Limitations)
    - Feature Importance
7. Conclusion & Recommendation
8. Save Model to Pickle

---

# **Business Problem Understanding**

---

## Context

Syarah Company is an online platform that facilitates the sale and purchase of guaranteed used cars. On Syarah.com, there are three parties involved in the used car trading process: the seller, the buyer, and Syarah Company as an intermediary. The online platform helps used car sellers reach consumers more easily and quickly with competitive selling prices. Additionally, buyers can more easily search for used cars through ads that can be filtered according to preferences displayed on Syarah.com. Sellers can advertise cars by listing their specifications and free to set the selling price. 

## Problem Statement

The Saudi Arabian used car market has experienced notable growth, with a strong CAGR projected for the coming years. The Saudi Arabian used car market, valued at USD 4.91 billion in 2021, is projected to reach USD 8.69 billion by 2027, with a 7.36% CAGR growth expected.

One of the biggest challenges for Syarah Company is `preventing overpricing and underpricing of used cars` that price being determined by sellers. A price prediction model based on car specifications is needed, which Syarah.com can easily use to `provide price recommendations` to sellers. This means that after the seller inputs the specifications, a price recommendation will automatically appear, which the seller can use as a reference in setting the price.

## Goals

- Having a price recommendation feature on Syarah.com can make it easier for sellers, potentially attracting more sellers to list their used cars on Syarah.com. This, in turn, increases the company's revenue due to the increase in charges applied, such as shipping charges and so on.
- More people will buy used cars and transact on Syarah.com because of the well-fitted price offers according to the car specifications.

From the two goals above, if more transactions occur, it will achieve the main goal, which is increasing the company's profit.

## Analytic Approach

Therefore, what we need to do is analyze the data to find patterns from the existing features that distinguish one car from another.

Next, we will `build a regression model` that will help the company provide a price prediction tool for newly listed used cars, which will be useful for sellers in determining the selling price of used cars.

## Metrics Evaluation

In the data cleaning process, not all outliers are removed (only extreme outliers). Therefore, the data will still have some outliers. The evaluation metrics used should not be sensitive to outliers. Evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Root Mean Squared Percentage Error (RMSPE) are not suitable for this dataset because they are sensitive to outliers.

The evaluation metrics used to measure the model's performance are:

- Mean Absolute Percentage Error (MAPE): the average percentage error produced by the regression model
- Mean Absolute Error (MAE): the average absolute value of the error
- R-squared (R2): R-squared if the final selected model is a linear model. The R-squared value is used to determine how well the model can represent the overall variance of the data. The closer it is to 1, the better the model fits the observational data. However, this metric is not valid for non-linear models.

`MAPE will be the primary reference for selecting the best model`; the smaller the MAPE, the better the model.

---

# **Data Understanding**

---

## Context

Raw data imported for cleaning:

https://github.com/ashalaisyar/Saudi-Arabia-Used-Car-Price-Prediction/blob/main/data_saudi_used_cars.csv

The dataset contains 5624 records of used cars collected from syarah.com. Each row represents a used car. Other information regarding each car is the brand name, model, manufacturing year, origin, options, engine capacity, transmission type, mileage that the car covered, region price, and negotiable.

## Columns

-	Type: Type of used car.
-	Region: The region in which the used car was offered for sale.
-	Make: The company name.
-	Gear_Type: Gear type size of used car.
-	Origin: Origin of used car.
-	Options: Options of used car.
-	Year: Manufacturing year.
-	Engine_Size: The engine size of used car.
-	Mileage: Mileage of used car	
-	Negotiable: True if the price is 0, that means it is negotiable.
-	Price: Used car price.

---

# **Data Cleaning**

---

## Check Missing Value
No Missing Value on Dataset.

## Check and Handling Ouliers
Only Extreme Outliers will be removed.
- Extreme Lower Limit : the lower bound for extreme outliers, calculated by subtracting three times the Interquartile Range (IQR) from the first quartile (Q1).
- Extreme Upper Limit : the upper bound for extreme outliers, calculated by adding three times the Interquartile Range (IQR) to the third quartile (Q3).

Extreme outliers are only found in the 'Year', 'Mileage', and 'Price' columns. Outliers that **will be removed** from the 'Year', 'Mileage', and 'Price' columns are those that are **outside the Extreme Lower Bound and Extreme Upper Bound**.

## Check and Handling Faulty Data
- Handling Faulty Data: 'Price' = 0
'Price' = 0 indicates the car can be obtained for free. This does not make sense because the context is the sale of used cars which should have a price that needs to be paid.

Percentage of data with 'Price' = 0 (31.7%) is more than 10%, but the data **still needs to be dropped**, because incorrect value data can cause incorrect prediction results.

- Handling Faulty Data: 'Price' = 1
'Price' = 1 which indicates the unreasonable price of used car. The cheapest used car for sale in Riyadh is priced at 8000 based on this article : https://ksa.carswitch.com/en/riyadh/used-cars/under-35000-for-sale-in-riyadh 

Percentage of data with 'Price' < 8000 (1.46%) is lower than 10% --> can be dropped because the percentage is lower than 10% and data that used in the model or algorithm is data that is expected to provide correct and reliable predictions.

- Handling Faulty Data: 'Mileage'
Based on statistic summary, there is an extreme maximum value of 'Mileage' = 20,000,000 km, which that value is not reliable.

Based on ksa.motory.com, the average mileage per year is assumed to be 16,000 miles for used cars in Saudi Arabia, which is equivalent to 25,750 km.

The oldest car manufacturing year in the dataset after cleaning extreme outliers is 1994 (30 years ago). Therefore, the maximum possible mileage of a car as of this year (2024) is 772,500 km (result of average mileage multiple with the difference between the current year and the car manufacturing year).

The above statement is made assuming the dataset was taken in 2024.

Therefore, data with 'Mileage' > 772,500 will be dropped because it is not reliable as a reference for prediction.

## Check Duplicate
Remove 3 duplicate data found.


## Check and Save Clean Data

The data has been cleaned, ready for analysis process. Clean data is in the repository: 

https://github.com/ashalaisyar/Saudi-Arabia-Used-Car-Price-Prediction/blob/main/clean_data_saudi_used_cars.csv

---

# **Feature Selection**

---

## Numerical Features
Determining numeric features will be assisted by inferential statistical analysis (correlation analysis).

P-Value of 'Price' < 0.05 indicates that 'Price' is not normally distributed so the non-parametric correlation test (Spearman) will be used. The result are:
- 'Year' has `positive medium correlation` with price.
- 'Engine_Size' `positive low correlation` with price.
- 'Mileage' has `negative low correlation` with price.
All Numerical features **will be used**

## Categorical Features
'Negotiable' feature is not related to the actual price of the used car, but rather describes negotiation process. Therefore, 'Negotiable' feature can be dropped. 

Determining other categorical features will be assisted by ANOVA (for 'Type', 'Region', 'Make', 'Gear_Type', 'Origin', and Options').

All P-Value of other categorical features < 0.05 --> there is a price difference between various categories for each feature, then 'Type', 'Region', 'Make', 'Gear_Type', 'Origin', and 'Options' are features that **will be used**.

---

# **Feature Engineering**

---

- Doesn't need to use an imputer because there are no missing values ​​in the dataset.
- Scaling is applied to equalize the scale of all numerical features using the Robust Scaler method (a method that is not sensitive to outliers). Robust Scaler is applied because there are still outliers in the dataset, when data cleaning extreme outliers are removed.
- Encoding is applied to transform categorical data into numerical format. One Hot Encoder is used for categorical features with <= 5 categories, while Binary Encoder is used for categorical features with > 5 categories.

---

# **Analytics**

---

## Model Benchmarking
Several regression models will be tried to predict prices. The models that will be used are:
- Base Model (KNN Regressor, Decision Tree Regressor, and Linear Regression)
- Voting & Stacking (Soft Voting, Stacking - KNN, Stacking - DT, Stacking - Linear Regression)
- Bagging (Linear Regression and Random Forest Regressor)
- Boosting (AdaBoost Regressor, Gradient Boosting Regressor, and XGBoost Regressor)

## Hyperparameter Tunning
Hyperparameter Tunning will be applied on the 3 best models to determine the best parameters of each model.
Best 3 Models: Random Forest Regressor, XGBoost, and Stacking - KNN.

The final model selected based on the smallest MAPE value after Hyperparameter Tunning is --> XGBoost Regressor with parameters 'subsample': 0.6; 'n_estimators': 500; 'max_depth': 13; and 'learning_rate': 0.1.

## Compare Actual and Predicted Value
To determine model limitations, it is necessary to check the absolute value of error and MAPE on each row of data. Data that has a MAPE > 50% is considered to have a high error, meaning that the model is not accurate when applied to that data, only **data with a MAPE < 50% can be predicted accurately by the final model**.

There are model limitations, meaning that used car data that the price can be accurately predicted by the model has the following numerical characteristics:
    - Used cars with actual price range of 11.000 - 498,000 SR
    - Used cars with a manufacturing year range of 1995 - 2021; an engine size range of 1 - 8.8 L; and a Mileage range of 100 - 570,000 KM

## Check Residual
- Most points are close to the zero line, showing that the model's predictions are generally accurate.
- Points far from the zero line are outliers or inaccurate predictions, especially when the target value (Price) is higher or more variable (over 300,000 SR).
- The even spread of residuals around the zero line suggests that the model's predictions are not biased (residuals are not only above or below the line but on both sides).

## Feature Importance
- The features that most influence 'Price' are 'Options', 'Make','Engine-Size', 'Origin', and 'Type'.

---

# **Conclusion & Recommendation**

---

## Conclusion
1. XGBoost Regressor model is the best model for predicting used car sales prices with parameters:
    - 'model__subsample': 0.6
    - 'model__n_estimators': 500
    - 'model__max_depth': 13
    - 'model__learning_rate': 0.1

2. The MAPE value of the XGBoost Regressor model on train dataset is 26%, meaning if the model predicts the selling price of a newly listed used car, the estimate will deviate by 26% from the actual price.

3. Prediction errors can be caused by other factors or other car specifications beyond the features trained on, because several processes (Feature Engineering, Model Benchmarking with ensemble models, boosting, bagging, and Hyperparameter Tuning) have been performed to reduce errors.

4. The features that most influence 'Price' are 'Options', 'Make','Engine-Size', 'Origin', and 'Type'.

5. The model has limitations, meaning it will perform well if applied within the range of data it has been trained on, which includes:
    - Used cars with a price range of 9000 - 850,000 SR
    - Used cars with a manufacturing year range of 1994 - 2021; an engine size range of 1 - 9 L; and a Mileage range of 100 - 749,000 KM

6. Although the model can be applied to datasets with the above characteristics, it only provides accurate predictions for data:
    - Used cars with actual price range of 11.000 - 498,000 SR
    - Used cars with a manufacturing year range of 1995 - 2021; an engine size range of 1 - 8.8 L; and a Mileage range of 100 - 570,000 KM

8. The impact of using this regression model if implemented by Syarah.com could save time and cost in conducting market research on competitive used car selling prices. Furthermore, Price prediction using regression model can increase profit up to 1.4% for each car.

## Recommendation
1. Add the amount of data (especially for prices above 300,000 SR) so that the model can train on more data and produce more accurate predictions with lower MAPE. The model for this prediction case was built using only 3688 clean rows of data (initially 5624 rows), so the data amount could be increased to around > 5000 clean rows.

2. Add features that are likely to correlate with the target 'Price', such as the condition of the car's interior, exterior, and fuel type.

3. Update the data with more recent manufacturing years (> 2021), because the newest car manufacturing year used in this dataset is 2021.

---

# **Save Model Using Pickle**

Saved Model:
https://github.com/ashalaisyar/Saudi-Arabia-Used-Car-Price-Prediction/blob/main/XGBoost_usedcars.sav


Summary of the process and results of project work can be seen in the following presentation file:

https://github.com/ashalaisyar/Saudi-Arabia-Used-Car-Price-Prediction/blob/main/PPT%20Result.pdf

---


