# Advanced Regression Example with Perpetual
# Comparing model performance with different training budgets

library(perpetual)

# Prepare data using airquality dataset (predicting Ozone levels)
data(airquality)
aq <- na.omit(airquality)
X <- as.matrix(aq[, c("Solar.R", "Wind", "Temp", "Month", "Day")])
y <- aq$Ozone

# Split into train/test
set.seed(42)
n <- nrow(X)
train_idx <- sample(n, size = floor(0.8 * n))
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Train models with different budgets to see effect on complexity vs performance
budgets <- c(0.1, 0.5, 1.0, 2.0)
results <- data.frame(
  Budget = numeric(),
  Trees = numeric(),
  RMSE = numeric(),
  R2 = numeric()
)

cat("Training models with different budgets...\n\n")

for (b in budgets) {
  model <- perpetual(
    x = X_train,
    y = y_train,
    objective = "SquaredLoss",
    budget = b
  )
  
  preds <- predict(model, X_test, type = "raw")
  rmse <- sqrt(mean((preds - y_test)^2))
  
  ss_res <- sum((y_test - preds)^2)
  ss_tot <- sum((y_test - mean(y_test))^2)
  r2 <- 1 - (ss_res / ss_tot)
  
  results <- rbind(results, data.frame(
    Budget = b,
    Trees = model$number_of_trees(),
    RMSE = round(rmse, 2),
    R2 = round(r2, 3)
  ))
}

cat("--- Results by Training Budget ---\n")
print(results)

cat("\nNote: Higher budget allows longer training and more trees,\n")
cat("which can improve performance but may risk overfitting.\n")
cat("The budget parameter controls the stopping criterion.\n")
