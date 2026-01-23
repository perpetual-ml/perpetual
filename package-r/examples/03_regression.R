# Regression Example with Perpetual
# Using the built-in 'mtcars' dataset to predict fuel efficiency (mpg)

library(perpetual)

# Prepare data
data(mtcars)
X <- as.matrix(mtcars[, c("cyl", "disp", "hp", "wt", "qsec", "gear", "carb")])
y <- mtcars$mpg

# Split into train/test
set.seed(42)
train_idx <- sample(nrow(X), size = 24)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Train a regression model with SquaredLoss
cat("Training regression model with SquaredLoss...\n")
model <- perpetual(
  x = X_train,
  y = y_train,
  objective = "SquaredLoss",
  budget = 1.0
)

# Print model summary
print(model)

# Predict
predictions <- predict(model, X_test, type = "raw")
cat("\nPredictions vs Actual (MPG):\n")
results <- data.frame(
  Actual = y_test,
  Predicted = round(predictions, 2),
  Error = round(y_test - predictions, 2)
)
print(results)

# Calculate RMSE
rmse <- sqrt(mean((predictions - y_test)^2))
cat("\nRMSE:", round(rmse, 3), "mpg\n")

# Calculate R-squared
ss_res <- sum((y_test - predictions)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
r_squared <- 1 - (ss_res / ss_tot)
cat("R-squared:", round(r_squared, 3), "\n")
