# Multiclass Classification Example with Perpetual
# Using the built-in 'iris' dataset to predict flower species

library(perpetual)

# Prepare data
data(iris)
X <- as.matrix(iris[, 1:4])  # Sepal.Length, Sepal.Width, Petal.Length, Petal.Width
y <- as.numeric(iris$Species)  # 1 = setosa, 2 = versicolor, 3 = virginica

# Split into train/test
set.seed(42)
train_idx <- sample(nrow(X), size = 120)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Train a multiclass classification model
cat("Training multiclass classification model...\n")
model <- perpetual(
  x = X_train,
  y = y_train,
  objective = "LogLoss",
  budget = 1.0
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

# Predict probabilities (returns matrix: rows = samples, cols = classes)
pred_prob <- predict(model, X_test, type = "prob")
cat("\nPredicted probabilities (first 5 samples):\n")
colnames(pred_prob) <- levels(iris$Species)
print(round(pred_prob[1:5, ], 3))

# Map numeric predictions back to species names for display
species_names <- levels(iris$Species)
pred_species <- species_names[pred_class]
actual_species <- species_names[y_test]
cat("\nPrediction comparison:\n")
print(data.frame(Actual = actual_species, Predicted = pred_species))
