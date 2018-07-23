### Model Definitions for Cross-Validation and Test set error estimation

## Penalized Logit
## Support Vector Machines
## Random Forests
## XGBoost
#------------------------------------------------------------------------------#
libs <- c("glmnet", "e1071", "magrittr", "doParallel", "foreach",
          "randomForest", "xgboost")

lapply(libs, library, character.only = TRUE)

cvLogit <- function(X_train, Y_train, alpha, X_test = NULL, Y_test = NULL, parallel = TRUE, ncores = 10) {
  '
  Train a cross-validated penalized logit model; option for Ridge/Enet/Lasso
  and parallelized performance
  
  Args:
  - X_train --> matrix of training predictors
  - Y_train --> vector/factor of training outcomes
  - alpha --> vector of mixing parameter values to test
  - X_test, Y_test ---> test set used for prediction (optional)
  
  Returns:
  - best penalty value (lambda)
  - tested mixing parameter value (alpha)
  - training misclassification rate
  - cv misclassification rate
  - test misclassification rate (if test set is supplied)
  '

  if (parallel) {
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
  }
  
  result <- foreach(i = 1:length(alpha), .combine = bind_rows, .inorder = FALSE) %do% {
    
    cat("Testing mixing parameter", i, "of ", length(alpha), "\n")
    
    # Begin CV
    logit <- cv.glmnet(X_train, Y_train,
                       family = 'binomial',
                       alpha = alpha[i],
                       standardize = FALSE,
                       type.measure = 'class',
                       parallel = parallel,
                       intercept = FALSE)
    
    # Extract results
    ind <- which.min(logit$cvm)
    
    # Get train error of full model
    train_preds <- predict(logit, X_train, type = 'class', s = 'lambda.min')
    train_misclass <- 1 - mean(train_preds == Y_train)

    # Evaluate on test set if supplied
    if (!is.null(X_test) & !is.null(Y_test)) {
      
      # Predict and calculate misclassification
      test_preds <- predict(logit, newx = X_test, type = 'class', s = 'lambda.min')
      test_misclass <-  1 - mean(test_preds == Y_test)
      
      # Assemble results
      list(
        alpha = alpha[i],
        lambda = logit$lambda.min,
        cv_misclass = logit$cvm[ind],
        train_misclass = train_misclass,
        test_misclass = test_misclass
      )
    } else {
      # Assemble results 
      list(
        alpha = alpha[i],
        lambda = logit$lambda.min,
        cv_misclass = logit$cvm[ind],
        train_misclass = train_misclass
      )
    }
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

cvRF <- function(X_train, Y_train, tree_sizes, folds, X_test = NULL, Y_test = NULL, parallel = TRUE, ncores = 10) {
  '
  Use foreach to parallelize RF tree growth for a specified number of trees and
  estimate OOB error using cross-validation
  
  Args:
  - X_train --> matrix of training predictors
  - Y_train --> vector/factor of training outcomes
  - ntrees --> number of trees to grow in forest
  - folds --> user-supplied vector of folds
  - X_test, Y_test ---> test set used for prediction
  
  Returns:
  - df with two columns:
  - num_trees
  - training misclassification rate
  - cv misclassification rate
  - test misclassiciation rate (if test set supplied)
  '
  
  if (parallel) {
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
  }
  
  # Start loop through ntrees sequence
  result <- foreach(i = 1:length(tree_sizes), .combine = bind_rows) %do% {
    
    cat("Testing RF of size:", tree_sizes[i], "\n") 
    
    # Begin CV
    cv_result <- foreach(j = 1:max(folds), .combine = bind_rows) %do% {
      
      cat("Fold", j, "of", max(folds), "\n")
      
      # Grow trees in parallel and combine into one
      rf <- foreach(ntree = rep(tree_sizes[i]/10, 10), .combine = combine, .packages = 'randomForest') %dopar% {
        
        # Grow indvl tree on the k-1 folds
        randomForest(
          x = X_train[folds != j, ],
          y = Y_train[folds != j],
          ntree = ntree,
          mtry = sqrt(dim(X_train)[2])
        )
      }
      
      # Predict on out-of-sample fold
      pred <- predict(rf, X_train[folds == j, ], type = 'response')
      
      # Return df of results
      data.frame(
        num_trees = tree_sizes[i],
        cv_misclass = 1 - mean(pred == Y_train[folds == j]),
        fold = j
      )
    }
    
    # Average CV results for given tree size
    cv_result %<>%
      group_by(num_trees) %>%
      summarise_at(vars(cv_misclass), mean)
    
    # Regrow forest on full train set for training error
    rf <- foreach(ntree = rep(tree_sizes[i]/10, 10), .combine = combine, .packages = 'randomForest') %dopar% {
      
      # Grow indvl trees
      randomForest(
        x = X_train,
        y = Y_train,
        ntree = ntree,
        mtry = sqrt(dim(X_train)[2])
      )
    }
    
    # Training error
    train_preds <- predict(rf, X_train, type = 'response')
    train_misclass <- 1 - mean(train_preds == Y_train)
    
    # Bind training error to result DF
    cv_result$train_misclass <- train_misclass
    
    # Evaluate on test set if available
    if (!is.null(X_test) & !is.null(Y_test)) {
      
      # Evaluate on test set
      test_pred <- predict(rf, X_test, type = 'response')
      test_misclass <- 1 - mean(test_pred == Y_test)
      
      # Bind training and testing result
      cv_result$test_misclass <- test_misclass
    }
    
    # Print result for row binding
    cv_result
    
  }
  
  if (parallel) {on.exit(stopCluster(cl))}
  
  return(result)
}

cvXGB <- function(X_train, Y_train, grid, folds, X_test = NULL, Y_test = NULL, parallel = TRUE, ncores = 10) {
  '
  Use foreach to parallelize xgboost for hyperparameter tuning with CV
  
  Args:
  - X_train --> matrix of training predictors
  - Y_train --> vector/factor of training outcomes
  - grid --> df of parameter combinations to test
  - folds --> user-supplied vector of folds
  - X_test, Y_test --> set used for out-of-sample error estimation  
  
  Returns:
  - df specifying:
  - tested parameters
  - train misclassification rate
  - cv misclassification rate
  - test misclassification rate (if test set is supplied)
  '
  # Force Y to numeric if inputted as factor
  if (is.factor(Y_train)) {
    Y_train <- as.numeric(levels(Y_train))[Y_train]
  }
  
  # Repeat for test set Y if available
  if (!is.null(Y_test) & is.factor(Y_test)) {
    Y_test <- as.numeric(levels(Y_test))[Y_test]
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
    cv_results <- foreach(j = 1:max(folds), .combine = bind_rows, .packages = 'xgboost') %dopar% {
      
      # Train model
      model <- xgboost(
        data = X_train[folds != j, ],
        label = Y_train[folds != j],
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
      preds <- as.numeric(predict(model, X_train[folds == j, ]) >= 0.5)
      
      # Spit out results
      data.frame(
        cv_misclass = 1 - mean(preds == Y_train[folds == j]),
        max_depth = depth,
        gamma = gamma,
        fold = j
      )
    }
    
    # Average across folds
    cv_results %<>% group_by(gamma, max_depth) %>% summarise_at(vars(cv_misclass), mean)
    
    # Retrain model on entire training set
    model <- xgboost(
      data = X_train,
      label = Y_train,
      params = list(
        nthread = ncores,
        objective = 'binary:logistic',
        max_depth = depth,
        gamma = gamma
      ),
      nrounds = 250,
      early_stopping_rounds = 10,
      verbose = 0
    )
    
    # Get training error
    train_preds <- as.numeric(predict(model, X_train) >= 0.5)
    train_misclass <- 1 - mean(train_preds == Y_train)
    
    # Bind training error to result DF
    cv_results$train_misclass <- train_misclass
    
    # Evaluate on test set if possible
    if (!is.null(X_test) & !is.null(Y_test)) {
      
      # Evaluate on test set
      preds <- as.numeric(predict(model, X_test) >= 0.5)
      test_misclass <- 1 - mean(preds == Y_test)
      
      # Add test misclassification rate to working result DF
      cv_results$test_misclass <- test_misclass
    }
    
    # Print results for row binding
    cv_results
  }
  
  if (parallel) {on.exit(stopCluster(cl))}
  
  return(results)
  
}

