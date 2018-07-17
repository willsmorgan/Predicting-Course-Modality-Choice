### Parameter Search - Fine
### Author: William Morgan

## Purpose:
# Use results from coarse hyperparameter search to inform more refined search
# of final hyperparameters for each tested model

## Outline:
# 0. Setup (libraries, paths, functions)
# 1. Preprocessing (df --> matrix)
# 2. Penalized Logistic Regression
# 3. Random Forest
# 4 Boosted Classifier

#------------------------------------------------------------------------------#

## 0. Setup
rm(list = ls(all = TRUE))

libs <- c("tidyverse", "magrittr", "doParallel", "foreach", 'caret')
lapply(libs, library, character.only = TRUE)
set.seed(18)

# Load models
source("Code/CV_utils.R")

# Start Log
log <- file("Logs/Fine Search.txt", open = "wt")
cat("Fine Parameter Search Result Log", file = log, sep = '\n')
#------------------------------------------------------------------------------#

## 1. Import and Preprocessing
training <- readRDS("Data/course choice training.Rds")
validation <- readRDS("Data/course choice validation.Rds")

# ## for testing purposes only:
# training %<>% sample_frac(0.005)
# validation %<>% sample_frac(0.005)

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ ., training)
Y_train <- training[, "icourse"] %>% unlist()

X_val <- model.matrix(icourse ~ ., validation)
Y_val <- validation[, "icourse"] %>% unlist()

# Define folds for cross-validation here
folds <-  createFolds(Y_train, k = 10, list = FALSE)

#------------------------------------------------------------------------------#

## 2. Penalized Logit

# Define list of mixing parameter values to test
alpha_seq <- seq(0, 1, length.out = 50)

# Run CV
logit_results <- cvLogit(X_train, Y_train, alpha_seq, ncores = 8)

# Print results for log
cat("\nLogit Results:", file = log, sep = '\n')

logit_results %>%
  arrange(misclassification) %>%
  select(misclassification, alpha, lambda) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(logit_results, "Results/Fine Search/logit.csv")

#------------------------------------------------------------------------------#

## 3. Random Forests

# Define parameter search
tree_sizes <- seq(100, 1500, by = 100)

# Run CV
rf_results <- foreach(i = 1:length(tree_sizes), .combine = bind_rows, .inorder = FALSE) %do% {
  
  # Train
  cvRF(X_train, Y_train, ntrees = tree_sizes[i], folds, parallel = TRUE, ncores = 8)
  
}

# Print results
cat("\nRandom Forest Results:", file = log, sep = '\n')

rf_results %>%
  arrange(misclassification) %>%
  select(misclassification, everything()) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(rf_results, "Results/Fine Search/rf.csv")
#------------------------------------------------------------------------------#

## 4. Boosted Classifier

# Define param. grid
boost_grid <- expand.grid(
  depth = seq(1, 10, by = 1),
  gamma = 10 ** runif(5, -2, 2)
)

# Run CV
boost_results <- cvXGB(X_train, Y_train, boost_grid, folds, ncores = 8)

# Print results
cat("Boosting Results:", file = log, sep = '\n')

boost_results %>%
  arrange(misclassification) %>%
  select(misclassification, everything()) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(boost_results, "Results/Fine Search/boost.csv")

#------------------------------------------------------------------------------#

## 5. Close Log and run initial Parameter Selection review

close(log)

source("Code/Course Choice/(3A) Parameter Selection - Fine.R")
