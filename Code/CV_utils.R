### Model Definitions for Cross-Validation

## Penalized Logit
## Support Vector Machines
## Random Forests
## XGBoost
#------------------------------------------------------------------------------#
libs <- c("glmnet", "e1071", "magrittr", "doParallel", "foreach",
          "randomForest", "xgboost")

lapply(libs, library, character.only = TRUE)

cvLogit <- function(X, Y, alpha, parallel = TRUE, ncores = 10) {
  '
  Train a cross-validated penalized logit model; option for Ridge/Enet/Lasso
  and parallelized performance
  
  Args:
  - X --> matrix of predictors
  - Y --> vector/factor of outcomes
  - alpha --> vector of mixing parameter values to test
  
  Returns:
  - best penalty value (lambda)
  - tested mixing parameter value (alpha)
  - cv`d misclassification rate
  '
  
  if (parallel) {
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
  }
  
  result <- foreach(i = 1:length(alpha), .combine = bind_rows, .inorder = FALSE) %do% {
    
    cat("Testing mixing parameter", i, "of ", length(alpha), "\n")
    
    # Begin CV
    logit <- cv.glmnet(X, Y,
                       family = 'binomial',
                       alpha = alpha[i],
                       standardize = FALSE,
                       nfolds = 10,
                       type.measure = 'class',
                       parallel = parallel,
                       intercept = FALSE)
    
    # Extract results
    ind <- which.min(logit$cvm)
    
    list(
      alpha = alpha[i],
      lambda = logit$lambda.min,
      misclassification = logit$cvm[ind],
      up = logit$cvup[ind],
      down = logit$cvlo[ind]
    )
  }

  # End cluster if necessary
  if (parallel) {on.exit(stopCluster(cl))}
  
  return(result)
}

cvSVM <- function(X, Y, kernel = 'linear', grid, folds, parallel = TRUE, ncores = 10) {
  '
  Use foreach to parallelize SVM cross-validation for parameter tuning
  
  Args:
  - X --> matrix of predictors
  - Y --> vector/factor of outcomes
  - kernel --> SVM Kernel, either "rbf" or "linear"
  - grid --> df of parameter combinations to test
  - folds --> user-supplied vector of folds


  Returns:
    - df with three columns:
      - gamma (if kernel rbf)
      - cost
      - misclassification rate
  '
  
  if (parallel) {
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
  }
  
  svm_results <- foreach(i = 1:nrow(grid), .combine = bind_rows) %do% {
    
    cat("Testing parameter combination", i, "of", nrow(grid), '\n')
    
    # Define params. for current iteration
    c <- grid[i, "cost"]
    
    if (kernel == "rbf") {
      g <- grid[i, "gamma"]
    }
      
    # Begin CV
    foreach(j = 1:max(folds), .combine = bind_rows, .inorder = FALSE, .packages = 'e1071') %dopar% {
      
      # Train
      if (kernel == 'rbf') {
        model <- svm(x = X[folds != j, ],
                     y = Y[folds != j],
                     scale = FALSE,
                     gamma = g,
                     cost = c,
                     kernel = 'radial')
        
        # Predict and collect results
        pred <- predict(model, X[folds == j, ])
        data.frame(misclassification = 1 - mean(pred == Y[folds == j]),
                   gamma = g,
                   cost = c,
                   fold = j)
      } else {
        model <- svm(x = X[folds != j, ],
                     y = Y[folds != j],
                     scale = FALSE,
                     cost = c,
                     kernel = 'linear')
        
        # Predict and collect results
        pred <- predict(model, X[folds == j, ])
        data.frame(misclassification = 1 - mean(pred == Y[folds == j]),
                   cost = c,
                   fold = j)
      }
    }
  }
  
  if (parallel) {on.exit(stopCluster(cl))}
  
  # Clean results
  if (kernel == 'rbf') {
    svm_results %<>%
      group_by(gamma, cost) %>%
      summarise_at(vars(misclassification), mean)
  } else {
    svm_results %<>%
      group_by(cost) %>%
      summarise_at(vars(misclassification), mean)
  }
  
  return(svm_results)
  
}

cvRF <- function(X, Y, ntrees, folds, parallel = TRUE, ncores = 10) {
  '
  Use foreach to parallelize RF tree growth for a specified number of trees and
  estimate OOB error using cross-validation
  
  Args:
  - X --> matrix of predictors
  - Y --> vector/factor of outcomes
  - ntrees --> number of trees to grow in forest
  - folds --> user-supplied vector of folds

  Returns:
  - df with two columns:
  - num_trees
  - misclassification rate
  '
  
  if (parallel) {
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
  }
  
  cat("Testing RF of size:", ntrees, "\n")
  
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
  
  if (parallel) {on.exit(stopCluster(cl))}
  
  # Average error rate across folds
  result %<>% group_by(num_trees) %>% summarise_at(vars(misclassification), mean)
  
  return(result)
}

cvXGB <- function(X, Y, grid, folds, parallel = TRUE, ncores = 10) {
  '
  Use foreach to parallelize xgboost for hyperparameter tuning with CV

  Args:
  - X --> matrix of predictors
  - Y --> vector/factor of outcomes
  - grid --> df of parameter combinations to test
  - folds --> user-supplied vector of folds

  Returns:
  - df specifying:
    - tested parameters
    - misclassification rate
  '
  # Force Y to numeric if inputted as factor
  if (is.factor(Y)) {
    Y <- as.numeric(levels(Y))[Y]
  }
  
  # Initialize cluster
  if (parallel) {
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
  }
  
  # Iterate through parameter grid
  results <- foreach(i = 1:nrow(grid), .combine = bind_rows) %do% {
    
    cat("Testing parameter combination", i, "of", nrow(grid), '\n')
    
    # Define parameters for current iteration    
    depth <- grid[i, "depth"]
    gamma <- grid[i, "gamma"]
    
    # Begin CV
    foreach(j = 1:max(folds), .combine = bind_rows, .packages = 'xgboost') %dopar% {
      
      # Train model
      model <- xgboost(
        data = X[folds != j, ],
        label = Y[folds != j],
        params = list(
          nthread = 1,
          objective = 'binary:logistic',
          max_depth = depth,
          gamma = gamma
        ),
        nrounds = 250,
        early_stopping_rounds = 10,
        verbose = 0
      )
      
      # Predict on Kth fold
      preds <- as.numeric(predict(model, X[folds == j, ]) >= 0.5)
      
      # Spit out results
      data.frame(
        misclassification = 1 - mean(preds == Y[folds == j]),
        max_depth = depth,
        gamma = gamma,
        fold = j
      )
    }
  }
  
  if (parallel) {on.exit(stopCluster(cl))}
  
  # Average error across folds
  results %<>%
    group_by(gamma, max_depth) %>%
    summarise_at(vars(misclassification), mean)
  
  return(results)
  
}

