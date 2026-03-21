# ---- Iris Classification ----
perpsnip(mode = "classification", budget = 0.5) |>
    parsnip::fit(Species ~ ., data = iris) |>
    parsnip::augment(new_data = iris) |>
    yardstick::metrics(truth = Species, estimate = .pred_class)

# ---- mtcars 'mpg' regression ----
perpetual::perpsnip(mode = "regression", budget = 1) |>
    parsnip::fit(mpg ~ cyl + . - vs - am, data = mtcars) |>
    parsnip::augment(new_data = mtcars) |>
    yardstick::metrics(truth = mpg, estimate = .pred)
