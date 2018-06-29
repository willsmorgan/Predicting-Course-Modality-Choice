### Course Choice Modeling
### Author: William Morgan

## Purpose:
# Run a series of classification models to create the most accurate classifier
# for determining whether a student will enroll in an iCourse

## Outline:
# 0. Setup (libraries, paths, functions)
# 1. Preprocessing (df --> matrix)
# 2. Penalized Logistic Regression
# 3. Support Vector Machine
# 4. Random Forest
# 5. NN Classifier


###NOTES:

# make it so that you are able to define a vector of variable names and then
# recreate the entire sample in case some are shitty and you want to get rid
# of them (looking at you, trnsfr_credit_accpt)

# Decide if you want to put in class weights for imbalanced classes

#------------------------------------------------------------------------------#
## 0. Setup
rm(list = ls(all = TRUE))

libs <- c("tidyverse", "glmnet", "e1071", "magrittr", "doParallel", "foreach",
          'caret', 'randomForest')
lapply(libs, library, character.only = TRUE)
set.seed(18)

# Load models
source("Code/CV_utils.R")
#------------------------------------------------------------------------------#

## 1. Import and Preprocessing
training <- readRDS("Data/course choice training.Rds")
testing <- readRDS("Data/course choice testing.Rds")

## for testing purposes only:
training %<>% sample_frac(0.01)
testing %<>% sample_frac(0.01)

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ ., training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ ., testing)
Y_test <- testing[, "icourse"] %>% unlist()

# Define folds for cross-validation here
folds <-  createFolds(Y_train, k = 10, list = FALSE)

#-----------------------------------------------------------------------------#  

## 2. Penalized Logit

# Define list of mixing parameter values to test
alpha_seq <-seq(0, 1, length.out = 11)

# Run CV
logit_results <- cvLogit(X_train, Y_train, alpha_seq)

#------------------------------------------------------------------------------#

## 3. Support Vector Machine with RBF kernel

# Define param. grid
svm_grid <- expand.grid(
  cost = 10 ** runif(1, -3, 3),
  gamma = 10 ** runif(1, -4, 1)
)

# Run CV
svm_results <- cvSVM(X_train, Y_train, svm_grid, folds)

#------------------------------------------------------------------------------#

## 4. Random Forests

# Define parameter search
tree_sizes <- c(100, 200, 500, 1000)

# Run CV
rf_results <- foreach(i = 1:length(tree_sizes), .combine = bind_rows, .inorder = FALSE) %do% {
  
  # Train
  cvRF(X_train, Y_train, ntrees = tree_sizes[i], folds, parallel = TRUE)
  
}
