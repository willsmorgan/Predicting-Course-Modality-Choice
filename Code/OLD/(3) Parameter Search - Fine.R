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
testing <- readRDS("Data/course choice testing.Rds")

## for testing purposes only:
training %<>% sample_frac(0.005)
testing %<>% sample_frac(0.005)

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ . -1, training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ . -1 , testing)
Y_test <- testing[, "icourse"] %>% unlist()


#------------------------------------------------------------------------------#

## 2. Penalized Logit

# Define list of mixing parameter values to test
alpha_seq <- seq(0, 1, length.out = 5)

# Run CV
logit_results <- cvLogit(X_train, Y_train,
                         alpha_seq,
                         X_test,
                         Y_test)

# Print results for log
cat("\nLogit Results:", file = log, sep = '\n')

logit_results %>%
  arrange(train_misclass) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(logit_results, "Results/Fine Search/logit.csv")

#------------------------------------------------------------------------------#

## 3. Random Forests

# Define parameter search
tree_sizes <- seq(100, 1500, by = 500)

rf_results <- cvRF(X_train, Y_train,
                   tree_sizes,
                   folds,
                   X_test,
                   Y_test)
# Print results
cat("\nRandom Forest Results:", file = log, sep = '\n')

rf_results %>%
  arrange(train_misclass) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(rf_results, "Results/Fine Search/rf.csv")
#------------------------------------------------------------------------------#

## 4. Boosted Classifier

# Define param. grid
boost_grid <- expand.grid(
  depth = seq(1, 10, by = 1),
  gamma = 10 ** runif(2, -2, 2)
)

# Run CV
boost_results <- cvXGB(X_train, Y_train,
                       boost_grid,
                       folds,
                       X_test,
                       Y_test)

# Print results
cat("Boosting Results:", file = log, sep = '\n')

boost_results %>%
  arrange(train_misclass) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

write_csv(boost_results, "Results/Fine Search/boost.csv")

#------------------------------------------------------------------------------#

## 5. Close Log and run initial Parameter Selection review

close(log)

source("Code/Course Choice/(3A) Parameter Selection - Fine.R")
