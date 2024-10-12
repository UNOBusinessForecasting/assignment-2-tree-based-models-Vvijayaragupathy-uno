[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SSrAZsYo)
# Assignment 2
## Econ 8310 - Business Forecasting

This assignment will make use of the models covered in Lessons 4 to 6. Models include:

- Decision Trees
- Random Forests
- Boosted Trees

Your job will be to forecast whether or not an individual purchased a “meal” at Stedman’s Café. You may need to restructure the data in order to make appropriate forecasts, but I have done significant preparation of the data to streamline the cleaning process. You will be graded based on the following:

- Your code executing without errors
- Storing your models to make predictions on new data
- Making reasonable predictions based on the data provided
- Forecasting a score for each team during each week

To complete this assignment, your code will need to contain the following:
1. A forecasting algorithm named `model` using the sklearn or xgboost implementation of one of the models covered in lessons 4 to 6. Your dependent variable is the `meal` variable (indicating whether or not a purchase was classified as a meal or not), and may or may not use exogenous variables from the remainder of the dataset. The training data is stored at the following URL: https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv
2. A fitted model named `modelFit`. This should be a model instance capable of generating forecasts using new data in the same shape as the data used in part (1).
3. A list or series of predictions using the data from the test period named `pred`. Your data should simply contain binary values (`1` or `0`, not boolean values) for whether or not you expect that an individual purchased a meal.
Make sure that your forecast includes predictions for EACH observation! There are 1000 observations in the prediction timeframe. The data from which to generate predictions is stored at the following URL: https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv

Note: While all models from lessons 4 to 6 are available to you, they may not all be good fits for the data. I recommend considering the data carefully, then choosing models to try. See which models seem to perform best on this data, and implement the best performer for the final submission of the project.

All code for this assignment should be written in the `assignment2.py` file found in the filetree.