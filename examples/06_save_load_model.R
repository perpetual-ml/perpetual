# Save and Load Model Example with Perpetual
# Demonstrates model persistence

library(perpetual)

# Prepare data
data(mtcars)
X <- as.matrix(mtcars[, c("cyl", "disp", "hp", "wt", "qsec")])
y <- mtcars$mpg

# Train model
cat("Training model...\n")
model <- perpetual(
  x = X,
  y = y,
  objective = "SquaredLoss",
  budget = 0.5
)

print(model)

# Make predictions before saving
pred_before <- predict(model, X[1:3, ], type = "raw")
cat("\nPredictions before saving:\n")
print(round(pred_before, 4))

# Save model to a temporary file
model_path <- tempfile(fileext = ".json")
cat("\nSaving model to:", model_path, "\n")
perpetual_save(model, model_path)

# Check file was created
cat("Model file size:", file.size(model_path), "bytes\n")

# Load model back
cat("\nLoading model from file...\n")
loaded_model <- perpetual_load(model_path)
print(loaded_model)

# Make predictions with loaded model
pred_after <- predict(loaded_model, X[1:3, ], type = "raw")
cat("\nPredictions after loading:\n")
print(round(pred_after, 4))

# Verify predictions match
if (all(abs(pred_before - pred_after) < 1e-6)) {
  cat("\n✓ Predictions match! Model saved and loaded successfully.\n")
} else {
  cat("\n✗ Predictions differ. Something went wrong.\n")
}

# Clean up
unlink(model_path)
cat("\nTemporary model file deleted.\n")
