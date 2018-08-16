### Parameter Search - Coarse
### Author: William Morgan

## Purpose:
# For each model test a wide array of hyperparameter values as a first pass for 
# parameter selection. Process results in secondary plotting script for easy
# evaluation

## Outline:
# 0. Setup (libraries, paths, functions)
# 1. Preprocessing (df --> matrix)
# 2. Penalized Logistic Regression
# 3. Random Forest
# 4. Boosted Classifier


###NOTES:

# make it so that you are able to define a vector of variable names and then
# recreate the entire sample in case some are shitty and you want to get rid
# of them (looking at you, trnsfr_credit_accpt)

# Decide if you want to put in class weights for imbalanced classes

#------------------------------------------------------------------------------#
## 0. Setup
rm(list = ls(all = TRUE))

libs <- c("tidyverse", "magrittr", 'caret')
lapply(libs, library, character.only = TRUE)
set.seed(18)

# Load models
source("Code/CV_utils.R")

# Start Log
log <- file("Logs/Coarse Search.txt", open = "wt")
cat("Coarse Parameter Search Result Log", file = log, sep = '\n')
#------------------------------------------------------------------------------#

## 1. Import and Preprocessing
training <- readRDS("Data/course choice training.Rds")

# ## for testing purposes only:
training %<>% sample_frac(0.02)

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ .-1, training)
Y_train <- training[, "icourse"] %>% unlist()

# Define folds for cross-validation here
folds <-  createFolds(Y_train, k = 10, list = FALSE)

#------------------------------------------------------------------------------#  

## 2. Penalized Logit

# Define list of mixing parameter values to test
alpha_seq <- seq(0, 1, length.out = 25)

# Run CV
logit_results <- cvLogit(X_train, Y_train, alpha_seq, ncores = 8)

# Print results for log
cat("\nLogit Results:", file = log, sep = '\n')

logit_results %>%
  arrange(train_misclass) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(logit_results, "Results/Coarse Search/logit.csv")

#------------------------------------------------------------------------------#

## 3. Random Forests

# Define parameter search
tree_sizes <- seq(200, 3000, by = 200)

# Run CV
rf_results <- cvRF(X_train, Y_train, folds, 
                   tree_sizes, sample_frac = 1,
                   proximity = FALSE, nodesize = 50,
                   ncores = 8)

# Print results
cat("\nRandom Forest Results:", file = log, sep = '\n')

rf_results %>%
  arrange(cv_misclass) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(rf_results, "Results/Coarse Search/rf.csv")
#------------------------------------------------------------------------------#

## 4. Boosted Classifier

# Define param. grid
boost_grid <- expand.grid(
  depth = seq(1, 15, by = 1),
  gamma = 10 ** runif(10, -2, 2)
)

# Run CV
boost_results <- cvXGB(X_train, Y_train, boost_grid, folds, ncores = 8)

# Print results
cat("\nBoosting Results:", file = log, sep = '\n')

boost_results %>%
  arrange(train_misclass) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(boost_results, "Results/Coarse Search/boost.csv")

#------------------------------------------------------------------------------#

## 5. Close Log and run initial Parameter Selection review

close(log)

source("Code/Course Choice/(2A) Parameter Selection - Coarse.R")
