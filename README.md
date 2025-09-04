# 📊 Customer Spending Regression Project in R 
# This is as directed from an online tutor -> Alejandro AO (on Youtube)

This project uses **linear and multiple regression** techniques in R to predict how much a customer spends yearly based on their behavior and engagement metrics. It's part of my learning journey to strengthen my understanding of regression and revisit key concepts from my statistics classes.

---

## 🔍 Objective

To predict **Yearly Amount Spent** by users based on:
- Average Session Length
- Time on Website
- Time on App
- Length of Membership

The project also explores how relevant each predictor is in explaining customer spending.

---

## 🧪 Methodology

- **Data Splitting**  
  Used `sample()` to randomly split the dataset into 80% training and 20% testing.  
  Ensured reproducibility with `set.seed(1)`.

- **Model Training**  
  Built a multiple linear regression model using the training set:
  ```r
  lm(Yearly.Amount.Spent ~ Avg..Session.Length + Time.on.App + Time.on.Website + Length.of.Membership)



## 🎯 Why I Built This

I created this project to:
- Practice regression modeling in R using real-world business data
- Recap key statistical concepts like error metrics, model fit, and residual analysis
- Explore how different customer behavior metrics data
- Strengthen my data storytelling skills through visualizations



## 📊 Visual Explorations Included

- Scatter plots 
- Pair plot of all continuous variables
- Histograms and boxplots for distribution analysis
- Residual diagnostics:
  - Histogram of residuals
  - Q-Q plot
  - Shapiro-Wilk test for normality

These visualizations helped me interpret relationships, check assumptions, and validate model performance.

---

## 🚀 Next Steps (Probably)

- Add more advanced models (e.g., decision trees, random forests)
- Explore feature importance and multicollinearity
- Try regularization techniques like ridge or lasso regression
- Package the workflow into a reusable R script or notebook
---

## 🛠️ Tools Used

- **R**: For data manipulation, modeling, and visualization  
- **RStudio**: As the development environment  
- **ggplot2**: For elegant and customizable plots  
- **Base R functions**: `lm()`, `predict()`, `summary()`, `sample()`, `shapiro.test()`  
- **Business customer dataset**: Provided in `.txt` format

---

Thanks for checking out my project! Feel free to explore the code, suggest improvements, or reach out if you'd like to collaborate.
