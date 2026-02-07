# Binary Classification Example with Perpetual
# Using the built-in 'mtcars' dataset to predict automatic vs manual transmission

library(perpetual)

# Prepare data: mtcars has 'am' column (0 = automatic, 1 = manual)
data(mtcars)
X <- as.matrix(mtcars[, c("mpg", "cyl", "disp", "hp", "wt", "qsec")])
y <- mtcars$am

# Split into train/test
set.seed(42)
train_idx <- sample(nrow(X), size = 24)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Train a binary classification model
cat("Training binary classification model...\n")
model <- perpetual(
  x = X_train,
  y = y_train,
  objective = "LogLoss",
  budget = 0.5
)

# Print model summary
print(model)

# Predict class labels
pred_class <- predict(model, X_test, type = "class")
cat("\nPredicted classes:", pred_class, "\n")
cat("Actual classes:   ", y_test, "\n")

# Calculate accuracy
accuracy <- mean(pred_class == y_test)
cat("\nAccuracy:", round(accuracy * 100, 1), "%\n")

# Predict probabilities
pred_prob <- predict(model, X_test, type = "prob")
cat("\nPredicted probabilities (P(manual)):\n")
print(round(pred_prob, 3))

# Get raw log-odds scores
pred_raw <- predict(model, X_test, type = "raw")
cat("\nRaw log-odds scores:\n")
print(round(pred_raw, 3))
