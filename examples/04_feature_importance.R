# Feature Importance Example with Perpetual
# Using the built-in 'mtcars' dataset

library(perpetual)

# Prepare data
data(mtcars)
feature_names <- c("cyl", "disp", "hp", "wt", "qsec", "gear", "carb")
X <- as.matrix(mtcars[, feature_names])
y <- mtcars$mpg

# Train model
cat("Training model for feature importance analysis...\n")
model <- perpetual(
  x = X,
  y = y,
  objective = "SquaredLoss",
  budget = 1.0
)

print(model)

# Get feature importance using different methods
cat("\n--- Feature Importance (Gain) ---\n")
imp_gain <- perpetual_importance(model, method = "gain", normalize = TRUE)
# Map feature indices to names
names(imp_gain) <- feature_names[as.numeric(names(imp_gain)) + 1]
print(sort(imp_gain, decreasing = TRUE))

cat("\n--- Feature Importance (Weight) ---\n")
imp_weight <- perpetual_importance(model, method = "weight", normalize = TRUE)
names(imp_weight) <- feature_names[as.numeric(names(imp_weight)) + 1]
print(sort(imp_weight, decreasing = TRUE))

cat("\n--- Feature Importance (Cover) ---\n")
imp_cover <- perpetual_importance(model, method = "cover", normalize = TRUE)
names(imp_cover) <- feature_names[as.numeric(names(imp_cover)) + 1]
print(sort(imp_cover, decreasing = TRUE))

# Simple bar plot if available
if (interactive()) {
  barplot(
    sort(imp_gain, decreasing = TRUE),
    main = "Feature Importance (Gain)",
    xlab = "Features",
    ylab = "Importance",
    las = 2,
    col = "steelblue"
  )
}
