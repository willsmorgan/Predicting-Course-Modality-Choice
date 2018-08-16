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
libs <- c("tidyverse", "magrittr", "doParallel", "foreach", 'caret')
lapply(libs, library, character.only = TRUE)
set.seed(18)

# Load models
source("Code/model_utils.R")

# Start Log
log <- file("Logs/Parameter Search.txt", open = "wt")
cat("Parameter Search Result Log", file = log, sep = '\n')

#------------------------------------------------------------------------------#
## 1. Import and Preprocessing

training <- readRDS("Data/course choice training.Rds")
testing <- readRDS("Data/course choice testing.Rds")

## for testing purposes only:
# training %<>% sample_frac(0.005)
# testing %<>% sample_frac(0.005)

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ . -1, training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ . -1 , testing)
Y_test <- testing[, "icourse"] %>% unlist()

#------------------------------------------------------------------------------#
## 2. Penalized Logit

# Set up Parameters
alpha_seq <- seq(0, 1, length.out = 25)

# Train models
logit_results <- PLogit(X_train, Y_train,
                        X_test, Y_test,
                        alpha_seq,
                        ncores = 6)


# Send results to log
cat("\nLogit Results:", file = log, sep = '\n')

logit_results %>%
  arrange(train_error) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

# Save results
write_csv(logit_results, "Results/Parameter Search/logit.csv")
#------------------------------------------------------------------------------#
## 3. Random Forest

# Define parameter search
forest_sizes <- seq(100, 3000, by = 100)

# Train the models
rf_results <- RF(X_train, Y_train,
                 X_test, Y_test,
                 forest_sizes = forest_sizes,
                 proximity = FALSE,
                 nodesize = 10,
                 ncores = 6)

# Print results
cat("\nRandom Forest Results:", file = log, sep = '\n')

# Send results to log
rf_results %>%
  arrange(train_error) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

# Save results
write_csv(rf_results, "Results/Parameter Search/rf.csv")
#------------------------------------------------------------------------------#
## 4. Boosted Classifier

# Set up parameter grid
boost_grid <- expand.grid(
  depth = seq(2, 15, by = 2),
  gamma = 10 ** runif(3, -1, 1),
  learning_rate = runif(3, 0, 1)
)

# Run training
boost_results <- XGB(X_train, Y_train,
                     X_test, Y_test,
                     boost_grid,
                     ncores = 6)
                     
# Print results
cat("Boosting Results:", file = log, sep = '\n')

# Send results to log
boost_results %>%
  arrange(train_error) %>%
  mutate_all(function(x) round(x, 5)) %>%
  head() %>%
  write.table(., file = log, row.names = FALSE)

# Save results
write_csv(boost_results, "Results/Parameter Search/boost.csv")

#------------------------------------------------------------------------------#
## 5. Close Log and run initial Parameter Selection review

close(log)

source("Code/Course Choice/(2A) Parameter Search - Plotting.R")