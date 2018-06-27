### Course Choice Modeling
### Author: William Morgan

## Purpose:
# Run a series of classification models to create the most accurate classifier
# for determining whether a student will enroll in an iCourse

## Outline:
# 0. Setup (libraries, paths, data import)
# 1. Preprocessing (df --> matrix)
# 2. Penalized Logistic Regression
# 3. Support Vector Machine
# 4. Random Forest
# 5. NN Classifier

#------------------------------------------------------------------------------#
## 0. Setup

rm(list = ls(all = TRUE))

libs <- c("tidyverse", "glmnet", "e1071", "magrittr", "doParallel")
lapply(libs, library, character.only = TRUE)
set.seed(18)

training <- readRDS("Data/course choice training.Rds")
testing <- readRDS("Data/course choice testing.Rds")
#------------------------------------------------------------------------------#

## 1. Preprocessing

X_train <- model.matrix(icourse ~ ., training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ ., testing)
Y_test <- testing[, "icourse"] %>% unlist()

###NOTES:
# make it so that you are able to define a vector of variable names and then
# recreate the entire sample in case some are shitty and you want to get rid
# of them (looking at you, trnsfr_credit_accpt)


#-----------------------------------------------------------------------------#  

## 2. Penalized Logit
cvLogit <- function(X, Y, alpha = 1, parallel = TRUE, cores = 10) {
  '
  Train a cross-validated penalized logit model option for Ridge/Enet/Lasso
  and parallelized performance
  
  Returns:
  - best lambda
  - misclassification performance
  '
  
  if (parallel) {
    cl <- makeCluster(cores)
    registerDoParallel(cl)
  }
  
  logit <- cv.glmnet(X, Y,
                     family = 'binomial',
                     alpha = alpha,
                     standardize = FALSE,
                     nfolds = 10,
                     type.measure = 'class',
                     parallel = parallel,
                     intercept = FALSE)
  
  # Extract results
  ind <- which.min(logit$cvm)
  
  result <- list(
    alpha = alpha,
    lambda = logit$lambda.min,
    error = logit$cvm[ind],
    up = logit$cvup[ind],
    down = logit$cvlo[ind]
  )
  
  
  # End cluster if necessary
  if (parallel) {stopCluster(cl)}
  
  return(result)
}

# Define mixing parameter search and empty DF for CV results
alpha_seq <- seq(0, 1, length.out = 10)
logit_results <- data.frame(alpha = double(),
                            lambda = double(),
                            error = double(),
                            up = double(),
                            down = double())

# Run through list of mixing parameters and store 
for (i in seq_along(alpha_seq)) {
  cat("Training for model:", i, "of ", length(alpha_seq), "\n")
  
  # Train
  result <- cvLogit(X_train, Y_train,
                    alpha = alpha_seq[i],
                    parallel = TRUE)
  
  # Store outcomes
  logit_results <- bind_rows(logit_results,
                             result)
}

## ADD TEST SET RESULTS??
## Class weights???
#------------------------------------------------------------------------------#

## Support Vector Machine

svm_tune <- tune(method = 'svm',
                 train.x = X_train,
                 train.y = Y_train,
                 ranges = list(
                   cost = c(0.1, 1, 10, 100, 1000),
                   gamma = c(0.01, 0.1, 1, 5)),
                 tunecontrol = tune.control(sampling = 'cross',cross = 10),
                 scale = FALSE)

