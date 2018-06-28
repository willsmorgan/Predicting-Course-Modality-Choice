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

libs <- c("tidyverse", "glmnet", "e1071", "magrittr", "doParallel", "foreach",
          'caret', 'randomForest')
lapply(libs, library, character.only = TRUE)
set.seed(18)

training <- readRDS("Data/course choice training.Rds")
testing <- readRDS("Data/course choice testing.Rds")

# ## for testing purposes only:
# training %<>% sample_frac(0.01)
# testing %<>% sample_frac(0.01)
#------------------------------------------------------------------------------#

## 1. Preprocessing

X_train <- model.matrix(icourse ~ ., training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ ., testing)
Y_test <- testing[, "icourse"] %>% unlist()

# Define folds for cross-validation here
folds <-  createFolds(Y_train, k = 10, list = FALSE)

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
                            misclassification = double(),
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


#------------------------------------------------------------------------------#

## 3. Support Vector Machine with RBF kernel

# Define param. grid
svm_grid <- expand.grid(
  cost = 10 ** runif(1, -3, 3),
  gamma = 10 ** runif(1, -4, 1)
)

# Initiate cluster
cl <- makeCluster(10)
registerDoParallel(cl)

svm_results <- foreach(i = 1:nrow(svm_grid), .combine = bind_rows) %do% {
  
  cat("Beginning iteration", i, "of", nrow(svm_grid), '\n')
  
  # Define params. for current iteration
  c <- svm_grid[i, "cost"]
  g <- svm_grid[i, "gamma"]
  
  # Begin CV
  out <- foreach(j = 1:max(folds), .combine = bind_rows, .inorder = FALSE, .packages = 'e1071') %dopar% {
    
    # Train
    model <- svm(x = X_train[folds != j, ],
                 y = Y_train[folds != j],
                 scale = FALSE,
                 gamma = g,
                 cost = c)
    
    # Predict and collect results
    pred <- predict(model, X_train[folds == j, ])
    data.frame(gamma = g,
               cost = c,
               misclassification = 1 - mean(pred == Y_train[folds == j]),
               fold = j)
  }
}

stopCluster(cl)

# Clean results
svm_results %<>% 
  group_by(gamma, cost) %>%
  summarise(misclassification = mean(misclassification))


### Class weights???

#------------------------------------------------------------------------------#

## 4. Random Forests

cvParRF <- function(X, Y, folds, ntrees) {
  '
  Use foreach to parallelize RF tree growth for a specified number of trees and
  estimate OOB error using cross-validation
  
  Returns:
  - df with two columns:
  - num_trees
  - misclassification rate
  '
  
  # Begin CV
  result <- foreach(j = 1:max(folds), .combine = bind_rows) %do% {
    
    cat("Fold", j, "of", max(folds), "\n")
    
    # Grow trees in parallel and combine into one
    rf <- foreach(ntree = rep(ntrees/10, 10), .combine = combine, .packages = 'randomForest') %dopar% {
      
      # Grow indvl tree on the k-1 folds
      randomForest(
        x = X[folds != j, ],
        y = Y[folds != j],
        ntree = ntree,
        mtry = sqrt(dim(X)[2])
      )
    }
    
    # Predict on out-of-sample fold
    pred <- predict(rf, X[folds == j, ], type = 'response')
    
    # Return df of results
    data.frame(
      num_trees = ntrees,
      misclassification = 1 - mean(pred == Y[folds == j]),
      fold = j
    )
  }
  
  # Average error rate across folds
  result %<>% group_by(num_trees) %>% summarise_at(vars(misclassification), mean)
  
  return(result)
}

cl <- makeCluster(10)
registerDoParallel(cl)

# Create empty df to store results
rf_results <- data.frame(
  num_trees = double(),
  misclassification = double()
)

# Begin training
for (i in c(100, 200)) {
  
  cat("Beginning CV for RF of size:", i, "\n")
  
  # Train
  result <- cvParRF(X_train, Y_train, folds, i)
  
  # Store outcome
  rf_results %<>% bind_rows(., result)
}
stopCluster(cl)
