# ----------
# IMPORT DATA AND SET UP
# ----------
data <- read.csv("./data/customer-data.txt")

getwd() # This is to get the current file's directory

# Then to view the data
View(data)

# To check the data structure
str(data)

# Then to check a quick summary of the data
summary(data) # This will analyze every column that we have

# ------------------------
# CREATE PLOTS AND SEARCH FOR INSIGHTS
# ------------------------

# We are going to be using the common ggplot package
# To install it!
# install.packages("ggplot2")
library(ggplot2)


# Correlation between time on website and yearly amount spent
# geompoint() is the dot color
# After running this, we can notice that there is no serious correlation between the x and y axis
ggplot(data, aes(x = Time.on.Website, y = Yearly.Amount.Spent)) +
  geom_point(color = "orange") + 
  ggtitle("Tinme on website against Yearly amount spent") +
  xlab("Time on website") +
  ylab("Yearly amount spent")


# Avg session length vs yearly amount spent
# We noticed a decent positive correlation -> The higher the session length the higher the yearly amount spent and vicer versa
ggplot(data, aes(x = Avg..Session.Length, y = Yearly.Amount.Spent)) +
  geom_point(color = "orange") + 
  ggtitle("Session Length against Yearly amount spent") +
  xlab("Session Length") +
  ylab("Yearly amount spent")

# This is the pair plot of all the columns (continuous)
# Note that pch is the size of each dot 
pairs(data[c("Avg..Session.Length",
             "Time.on.App",
             "Time.on.Website",
             "Length.of.Membership",
             "Yearly.Amount.Spent"
            )],
      col = "orange",
      pch = 16, 
      main = "Pairplot of all continuoys variables"
          )

# ----------------
# EXPLORE THE SELECTED VARIABLE
# ----------------

# is the variable normally distributed?
hist(data$Length.of.Membership)

# Then to use ggplot with a histogram
ggplot(data, aes(x= Length.of.Membership)) +
  geom_histogram(
    color = "white",
    fill = "orange",
    binwidth = 0.5
  )


# To create a box plot
boxplot(data$Length.of.Membership)


# Then to create one using ggplot
ggplot(data, aes(x=Length.of.Membership)) +
  geom_boxplot(fill = "orange")

# ---------------
# Fitting a linear model
# ---------------

attach(data) # This will allow us to treat "data" as a variable then, we can treat it as a normal variable in functions

# lm() means linear model
# "~" means "is predicted by"
# Here, we are modelling Yearly.Amount.Spent as a function of Length.of.Membership

# What can we derive from this linear fitting?

# lm.fit holds the fitted model object:
# Estimated coefficients (intercept and slope)
# Residuals (errors between predicted and actual values)
# Fitted values (predicted spending)
# Diagnostic info (R-squared, p-values, etc.)

lm.fit1 <- lm(Yearly.Amount.Spent ~ Length.of.Membership)

# This will help us view the stats
summary(lm.fit1)

# Then to view the linear regression plot
plot(Yearly.Amount.Spent ~ Length.of.Membership)

# Then to display the abline - This will help us analyze the regression better
abline(lm.fit1, col = "red")


# --------------------------
# RESIDUALS ANALYSIS
# --------------------------
# This is a way of checking if our data is normally distributed. 
# We can use a histogram to check -> If we see equal distribution on both sides, then we are sure that it is normally distributed
hist(residuals(lm.fit1))

# We can also use the qqnorm to check it, if we have the normal distribution as well
# Here, we get the "Sample Quantiles" on the y axis and then we get the "Theoretical Quantiles" on the x axis

qqnorm(residuals(lm.fit1)) # This will actually be the line showcasing the residual distributions
qqline(residuals(lm.fit1), col = "red")

# There is a way to test our distribution
# This test here tests the distribution -> It kind of tells us if it is normal
shapiro.test(residuals(lm.fit1)) # The test assumes that the distribution is normal (Null Hypothesis)
# Then based on the p-value, we can conclude on a decision for this


# -----------------------
# EVALUATION OF THE MODEL
# -----------------------
# sets the random number generator seed in R. This means that any random operations (like sampling, shuffling, or generating random numbers) will produce the same results every time you run the code — as long as the seed is the same.
set.seed(1)


# nrow(data) gives you the total number of rows in your dataset.
# 1:nrow(data) creates a sequence of row indices (e.g., 1 to 1000).
# 0.8 * nrow(data) calculates how many rows you want — 80% of the total.
# sample(...) randomly picks that many row indices without replacement.
# The result is stored in row.number, which is a vector of randomly selected row indices.

row.number <- sample(1:nrow(data), 0.8*nrow(data))

# Then to train
# This will be the train data
# For the train, it will create a subset of the original data -> In this instance, it is 80%
train <- data[row.number,] # This is 80 percent of it

test <- data[-row.number,] # This is the other way around. That is, (100 - 80) % of it. This will be about 20 % then

# Then to train a new model based on our train data
# estimate the linear fitting with the training set
lm.fit0.8 <- lm(Yearly.Amount.Spent~Length.of.Membership, data = train) # Note that the predictor here is "Length.of.Membership" and then the "Yearly.Amount.Spent" is the response variable
summary(lm.fit0.8) # This will show the stats based on the training data

# Without looking at the actual data here, we are going to try using the test data on the train data linear model
# lm.fit0.8 is your trained linear regression model, likely built using 80% of your data.
# test is the remaining 20% of your data that the model hasn’t seen — used to evaluate performance.
# predict() uses the model to estimate the target variable (e.g., Yearly.Amount.Spent) for each row in test.
# The result — stored in prediction0.8 — is a vector of predicted values.
prediction0.8 <- predict(lm.fit0.8, newdata = test)

# Note that the predictor is the one to be compared
# That is, we can check the error by comparing it to the actual column (That is, the predictor column)

# This is to check the difference between what was predicted and the actual value
err0.8 <- prediction0.8 - test$Yearly.Amount.Spent # This will then compare the predicted values

# This is to check the root mean square error
# RMSE stands for Root Mean Squared Error.
# It calculates the square root of the average squared differences between predicted and actual values.
# Use: Tells you how far off your predictions are, on average. Lower RMSE = better fit.
# Interpretation: If RMSE is 5, your predictions are off by about $5 on average.
rmse <- sqrt(mean(err0.8^2))

# mean absolute percentage error
# MAPE stands for Mean Absolute Percentage Error.
# It measures the average percentage error between predicted and actual values.
# Use: Great for understanding error in relative terms — especially when your target variable spans a wide range.
# Interpretation: If MAPE is 0.10, your predictions are off by 10% on average.
mape <- mean(abs(err0.8/test$Yearly.Amount.Spent))

# This creates a table instead of a list since we have a heading for each
c(rmse = rmse, mape = mape, R2 = summary(lm.fit0.8)$r.squared)

# -------------
# MULTIPLE REGRESSION  (This creates multiple regression)
# -------------

attach(data)

# This is for multiple regression
# This is training a multidimensional linear model
lm.fit <- lm(Yearly.Amount.Spent ~ Avg..Session.Length + Time.on.App + Time.on.Website + Length.of.Membership)

# Then to check the summary
summary(lm.fit) # We get the coefficients, the "Estimate" tells us how important the variable is to the model
# From the analysis we can notice that the "Length.of.Membership" is the most important one!
# The P value tells us if the variable is significant (This lets us decide if we need to keep the variable or not)
# We can also check the "stars" beside the p value -> The ones with no stars are simply "insignificant"
# Here the time on Website is insignificant


# -----------------------
# EVALUATION OF THE MULTIPLE REGRESSION
# -----------------------
# sets the random number generator seed in R. This means that any random operations (like sampling, shuffling, or generating random numbers) will produce the same results every time you run the code — as long as the seed is the same.
set.seed(1)


# nrow(data) gives you the total number of rows in your dataset.
# 1:nrow(data) creates a sequence of row indices (e.g., 1 to 1000).
# 0.8 * nrow(data) calculates how many rows you want — 80% of the total.
# sample(...) randomly picks that many row indices without replacement.
# The result is stored in row.number, which is a vector of randomly selected row indices.

row.number <- sample(1:nrow(data), 0.8*nrow(data))

# Then to train
# This will be the train data
# For the train, it will create a subset of the original data -> In this instance, it is 80%
train <- data[row.number,] # This is 80 percent of it

test <- data[-row.number,] # This is the other way around. That is, (100 - 80) % of it. This will be about 20 % then

# Then to train a new model based on our train data
# estimate the linear fitting with the training set
# This then trains the data on multiple variables instead
multi.lm.fit0.8 <- lm(Yearly.Amount.Spent ~ Avg..Session.Length + Time.on.App + Time.on.Website + Length.of.Membership,
                data = train)
# Note that the predictor here is "Length.of.Membership" and the rest & then the "Yearly.Amount.Spent" is the response variable
summary(multi.lm.fit0.8) # This will show the stats based on the training data

# Without looking at the actual data here, we are going to try using the test data on the train data linear model
# lm.fit0.8 is your trained linear regression model, likely built using 80% of your data.
# test is the remaining 20% of your data that the model hasn’t seen — used to evaluate performance.
# predict() uses the model to estimate the target variable (e.g., Yearly.Amount.Spent) for each row in test.
# The result — stored in prediction0.8 — is a vector of predicted values.
prediction0.8 <- predict(multi.lm.fit0.8, newdata = test)

# Note that the predictor is the one to be compared
# That is, we can check the error by comparing it to the actual column (That is, the predictor column)

# This is to check the difference between what was predicted and the actual value
err0.8 <- prediction0.8 - test$Yearly.Amount.Spent # This will then compare the predicted values

# This is to check the root mean square error
# RMSE stands for Root Mean Squared Error.
# It calculates the square root of the average squared differences between predicted and actual values.
# Use: Tells you how far off your predictions are, on average. Lower RMSE = better fit.
# Interpretation: If RMSE is 5, your predictions are off by about $5 on average.
rmse <- sqrt(mean(err0.8^2))

# mean absolute percentage error
# MAPE stands for Mean Absolute Percentage Error.
# It measures the average percentage error between predicted and actual values.
# Use: Great for understanding error in relative terms — especially when your target variable spans a wide range.
# Interpretation: If MAPE is 0.10, your predictions are off by 10% on average.
mape <- mean(abs(err0.8/test$Yearly.Amount.Spent))

# This creates a table instead of a list since we have a heading for each
c(rmse = rmse, mape = mape, R2 = summary(multi.lm.fit0.8)$r.squared)













