# Prediction Intervals Example with Perpetual
# Using the built-in 'airquality' dataset for regression with uncertainty

library(perpetual)

# Prepare data - predict Ozone from other variables
data(airquality)
# Remove rows with missing values for simplicity
aq <- na.omit(airquality)
X <- as.matrix(aq[, c("Solar.R", "Wind", "Temp", "Month", "Day")])
y <- aq$Ozone

# Split into train/calibration/test
set.seed(42)
n <- nrow(X)
train_idx <- sample(n, size = floor(0.6 * n))
remaining <- setdiff(1:n, train_idx)
cal_idx <- remaining[1:floor(length(remaining)/2)]
test_idx <- remaining[(floor(length(remaining)/2)+1):length(remaining)]

X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_cal <- X[cal_idx, ]
y_cal <- y[cal_idx]
X_test <- X[test_idx, ]
y_test <- y[test_idx]

cat("Train:", length(y_train), "samples\n")
cat("Calibration:", length(y_cal), "samples\n")
cat("Test:", length(y_test), "samples\n")

# Train model
cat("\nTraining regression model...\n")
model <- perpetual(
  x = X_train,
  y = y_train,
  objective = "SquaredLoss",
  budget = 1.0
)

print(model)

# Make point predictions
point_preds <- predict(model, X_test, type = "raw")

# Display predictions vs actual
cat("\n--- Predictions vs Actual (first 10 samples) ---\n")
results <- data.frame(
  Actual = y_test[1:10],
  Predicted = round(point_preds[1:10], 1),
  Error = round(y_test[1:10] - point_preds[1:10], 1)
)
print(results)

# Calculate RMSE
rmse <- sqrt(mean((point_preds - y_test)^2))
cat("\nRMSE:", round(rmse, 2), "\n")

# Calculate R-squared
ss_res <- sum((y_test - point_preds)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
r_squared <- 1 - (ss_res / ss_tot)
cat("R-squared:", round(r_squared, 3), "\n")

# Note: For conformal prediction intervals, use the perpetual_calibrate() function
# after training. This example shows basic prediction - see documentation for
# full calibration workflow.
