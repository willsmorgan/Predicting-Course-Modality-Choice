### Parameter Search - Coarse
### Author: William Morgan

## Purpose:
# Run a series of classification models to create the most accurate classifier
# for determining whether a student will enroll in an iCourse

## Outline:
# 0. Setup (libraries, paths, functions)
# 1. Preprocessing (df --> matrix)
# 2. Penalized Logistic Regression
# 3. Support Vector Machine (RBF, Linear)
# 4. Random Forest
# 5. Boosted Classifier


###NOTES:

# make it so that you are able to define a vector of variable names and then
# recreate the entire sample in case some are shitty and you want to get rid
# of them (looking at you, trnsfr_credit_accpt)

# Decide if you want to put in class weights for imbalanced classes

#------------------------------------------------------------------------------#
## 0. Setup
rm(list = ls(all = TRUE))

libs <- c("tidyverse", "magrittr", "doParallel", "foreach", 'caret')
lapply(libs, library, character.only = TRUE)
set.seed(18)

# Load models
source("Code/CV_utils.R")

# Start Log
log <- file("Logs/Course Choice modeling.txt", open = "wt")
cat("Course Choice Cross-Validation Result Log", file = log, sep = '\n')
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

#-----------------------------------------------------------------------------#  

## 2. Penalized Logit

# Define list of mixing parameter values to test
alpha_seq <-seq(0, 1, length.out = 50)

# Run CV
logit_results <- cvLogit(X_train, Y_train, alpha_seq)

# Print results for log
cat("Logit Results:", file = log, sep = '\n')

logit_results %>%
  arrange(misclassification) %>%
  select(misclassification, alpha, lambda) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(logit_results, "Results/Coarse Search/logit.csv")
#------------------------------------------------------------------------------#

## 3.1: Support Vector Machine with RBF kernel

# Define param. grid
svm_grid <- expand.grid(
  cost = 10 ** runif(5, -3, 3),
  gamma = 10 ** runif(5, -4, 1)
)

# Run CV
rbf_svm_results <- cvSVM(X_train, Y_train, kernel = 'rbf', svm_grid, folds)

# Print results for log
cat("RBF SVM Results:", file = log, sep = '\n')

rbf_svm_results %>%
  arrange(misclassification) %>%
  select(misclassification, gamma, cost) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(rbf_svm_results, "Results/Coarse Search/rbf_svm.csv")
#------------------------------------------------------------------------------#

## 3.2: SVM with Linear kernel

# Parameters
svm_grid <- expand.grid(cost = 10 ** runif(25, -3, 3))

# Run CV
lin_svm_results <- cvSVM(X_train, Y_train, kernel = 'linear', svm_grid, folds)

# Print results
cat("Linear SVM Results:", file = log, sep = '\n')

lin_svm_results %>%
  arrange(misclassification) %>%
  select(misclassification, everything()) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(lin_svm_results, "Results/Coarse Search/lin_svm.csv")
#------------------------------------------------------------------------------#

## 4. Random Forests

# Define parameter search
tree_sizes <- seq(100, 1500, by = 100)

# Run CV
rf_results <- foreach(i = 1:length(tree_sizes), .combine = bind_rows, .inorder = FALSE) %do% {
  
  # Train
  cvRF(X_train, Y_train, ntrees = tree_sizes[i], folds, parallel = TRUE)
  
}

# Print results
cat("Random Forest Results:", file = log, sep = '\n')

rf_results %>%
  arrange(misclassification) %>%
  select(misclassification, everything()) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(rf_results, "Results/Coarse Search/rf.csv")
#------------------------------------------------------------------------------#

## 5. Boosted Classifier

# Define param. grid
boost_grid <- expand.grid(
  depth = seq(1, 10, by = 1),
  gamma = 10 ** runif(5, -2, 2)
)

# Run CV
boost_results <- cvXGB(X_train, Y_train, boost_grid, folds)

# Print results
cat("Boosting Results:", file = log, sep = '\n')

boost_results %>%
  arrange(misclassification) %>%
  select(misclassification, everything()) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(boost_results, "Results/Coarse Search/boost.csv")

#------------------------------------------------------------------------------#

## 6. Close Log and run initial Parameter Selection review

close(log)

source("Code/Course Choice/(2A) Parameter Selection - Coarse.R")
