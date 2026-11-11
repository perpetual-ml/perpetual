# SHAP-style Feature Contributions Example with Perpetual
# Using the built-in 'iris' dataset

library(perpetual)

# Prepare data - binary classification (setosa vs non-setosa)
data(iris)
X <- as.matrix(iris[, 1:4])
y <- as.numeric(iris$Species == "setosa")  # 1 if setosa, 0 otherwise

# Train model
cat("Training model for SHAP contributions...\n")
model <- perpetual(
  x = X,
  y = y,
  objective = "LogLoss",
  budget = 0.5
)

print(model)

# Get SHAP-style contributions for predictions
# type = "contribution" returns a matrix (n_samples x (n_features + 1))
# The last column is the bias term
contributions <- predict(model, X[1:5, ], type = "contribution")

# Add column names
feature_names <- colnames(iris)[1:4]
colnames(contributions) <- c(feature_names, "bias")

cat("\n--- SHAP-style Contributions (first 5 samples) ---\n")
print(round(contributions, 4))

# Show how contributions sum to predictions
cat("\n--- Verification: Contributions sum to predictions ---\n")
predictions <- predict(model, X[1:5, ], type = "raw")
contribution_sums <- rowSums(contributions)

comparison <- data.frame(
  Sample = 1:5,
  Prediction = round(predictions, 4),
  ContributionSum = round(contribution_sums, 4),
  Difference = round(predictions - contribution_sums, 6)
)
print(comparison)

# Analyze feature impact for a single prediction
cat("\n--- Feature Impact for Sample 1 ---\n")
sample_1_contrib <- contributions[1, ]
cat("Sample 1 (", ifelse(y[1] == 1, "setosa", "not setosa"), "):\n", sep = "")
for (i in seq_along(sample_1_contrib)) {
  cat(sprintf("  %-15s: %+.4f\n", names(sample_1_contrib)[i], sample_1_contrib[i]))
}
