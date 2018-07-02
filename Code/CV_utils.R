### Model Definitions for Cross-Validation

## Penalized Logit
## Support Vector Machines
## Random Forests
## XGBoost
#------------------------------------------------------------------------------#
libs <- c("glmnet", "e1071", "magrittr", "doParallel", "foreach",
          "randomForest", "xgboost")

lapply(libs, library, character.only = TRUE)

cvLogit <- function(X, Y, alpha, parallel = TRUE) {
  '
  Train a cross-validated penalized logit model option for Ridge/Enet/Lasso
  and parallelized performance
  
  
  Returns:
  - best penalty value
  - tested mixing parameter value
  - misclassification performance
  '
  
  if (parallel) {
    cl <- makeCluster(detectCores() - 2)
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

cvSVM <- function(X, Y, grid, folds, parallel = TRUE) {
  '
  Use foreach to parallelize SVM cross-validation for parameter tuning
  
  Returns:
    - df with three columns:
      - gamma
      - cost
      - misclassification rate
  '
  
  if (parallel) {
    cl <- makeCluster(detectCores() - 2)
    registerDoParallel(cl)
  }
  
  svm_results <- foreach(i = 1:nrow(svm_grid), .combine = bind_rows) %do% {
    
    cat("Testing parameter combination", i, "of", nrow(svm_grid), '\n')
    
    # Define params. for current iteration
    c <- svm_grid[i, "cost"]
    g <- svm_grid[i, "gamma"]
    
    # Begin CV
    foreach(j = 1:max(folds), .combine = bind_rows, .inorder = FALSE, .packages = 'e1071') %dopar% {
      
      # Train
      model <- svm(x = X[folds != j, ],
                   y = Y[folds != j],
                   scale = FALSE,
                   gamma = g,
                   cost = c)
      
      # Predict and collect results
      pred <- predict(model, X[folds == j, ])
      data.frame(gamma = g,
                 cost = c,
                 misclassification = 1 - mean(pred == Y[folds == j]),
                 fold = j)
    }
  }
  
  if (parallel) {on.exit(stopCluster(cl))}
  
  # Clean results
  svm_results %<>%
    group_by(gamma, cost) %>%
    summarise(misclassification = mean(misclassification))
  
  return(svm_results)
  
}

cvRF <- function(X, Y, ntrees, folds, parallel = TRUE) {
  '
  Use foreach to parallelize RF tree growth for a specified number of trees and
  estimate OOB error using cross-validation
  
  Returns:
  - df with two columns:
  - num_trees
  - misclassification rate
  '
  if (parallel) {
    cl <- makeCluster(detectCores() - 2)
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


cvXGB <- function(X, Y, folds, nrounds) {
  
  # Force Y to numeric if inputted as factor
  if (is.factor(Y)) {
    Y <- as.numeric(levels(Y))[Y]
  }
  
  result <- foreach(i = 1:max(folds), .combine = bind_rows) %do% {
    
    cat("Fold", i, "of", max(folds), "\n")
    
    # Train model
    model <- xgboost(
      data = X[folds != i, ],
      label = Y[folds != i],
      params = list(
        nthread = detectCores() - 2,
        objective = 'binary:logistic'
      ),
      nrounds = nrounds,
      verbose = 0
    )
    
    # Predict on Kth fold
    preds <- as.numeric(predict(model, X[folds == i, ]) >= 0.5)
    
    # Spit out results
    data.frame(
      misclassification = 1 - mean(preds == Y[folds == i]),
      fold = i
    )
  }
  
  # Average error across folds
  result %<>% summarise_at(vars(misclassification), mean)
  
  return(result)
  
}
