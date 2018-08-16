### Model Definitions for Test set error estimation

## Penalized Logit
## Support Vector Machines
## Random Forests
## XGBoost
#------------------------------------------------------------------------------#
libs <- c("glmnet", "e1071", "magrittr", "doParallel", "foreach",
          "randomForest", "xgboost")

lapply(libs, library, character.only = TRUE)

#------------------------------------------------------------------------------#
## PENALIZED LOGIT
PLogit <- function(X_train, Y_train,
                   X_test = NULL, Y_test = NULL,
                   alpha,
                   ncores = 10) {
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
  
  if (ncores > 1) {
    # Initialize cluster
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
    on.exit(stopCluster(cl))
  }
  
  result <- foreach(i = 1:length(alpha), .combine = bind_rows, .inorder = FALSE) %do% {
    
    cat("Testing mixing parameter", i, "of ", length(alpha), "\n")
    
    # Begin CV
    logit <- cv.glmnet(X_train, Y_train,
                       family = 'binomial',
                       alpha = alpha[i],
                       standardize = FALSE,
                       type.measure = 'class',
                       parallel = TRUE,
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
      data.frame(
        train_error = train_misclass,
        test_error = test_misclass,
        alpha = alpha[i],
        lambda = logit$lambda.min
      )
      
    } else {
      
      # Assemble results
      data.frame(
        train_error = train_misclass,
        alpha = alpha[i],
        lambda = logit$lambda.min
      )
    }
  }

  return(result)
}

#------------------------------------------------------------------------------#
## RANDOM FORESTS
RF <- function(X_train, Y_train,
               X_test = NULL, Y_test = NULL,
               forest_sizes,
               proximity = FALSE,
               nodesize = 5,
               ncores = 10) {
  '
  Use foreach to parallelize RF tree growth for a specified number of trees and
  estimate OOB error using cross-validation
  
  Args:
  - X_train --> matrix of training predictors
  - Y_train --> vector/factor of training outcomes
  - forest_sizes --> forest sizes to test
  - X_test, Y_test ---> test set used for prediction
  
  Returns:
  - df with two columns:
  - forest_size
  - test misclassiciation rate (if test set supplied)
  '
  # Define functino to parallelize forest growth
  parGrow <- function(X, Y, forest_size, proximity, nodesize) {
    
    # Search parent environment for ncores and initalize cluster
    if (ncores > 1) {
      cl <- makeCluster(ncores)
      registerDoParallel(cl)
      on.exit(stopCluster(cl))
    }
    
    # Grow trees in parallel and combine into one
    rf <- foreach(ntree = rep(forest_size/10, 10),
                  .combine = randomForest::combine,
                  .multicombine = TRUE,
                  .inorder = FALSE,
                  .packages = 'randomForest') %dopar% {
                    
                    # Grow indvl tree on the k-1 folds
                    randomForest(
                      x = X,
                      y = Y,
                      ntree = ntree,
                      mtry = floor(sqrt(ncol(X))),
                      proximity = proximity,
                      nodesize = nodesize
                    )
                  }
    
    # Return an RF object
    rf
  }

  # Run through vector of tree sizes
  result <- foreach(i = seq_along(forest_sizes),
                    .combine = bind_rows,
                    .multicombine = TRUE,
                    .inorder = FALSE) %do% {
                      
                      cat("Testing RF of size:", forest_sizes[i], "\n")
                      
                      # Establish forest size to test
                      forest_size <- forest_sizes[i]
                      
                      # Create forest
                      forest <- parGrow(X = X_train,
                                        Y = Y_train,
                                        forest_size = forest_size,
                                        proximity = proximity,
                                        nodesize = nodesize)
                      
                      # Evaluate on training set (to check for overfit)
                      train_pred <- predict(forest,
                                            X_train,
                                            type = 'response')
                      
                      train_misclass <- 1 - mean(train_pred == Y_train)

                      
                      # Find test error if supplied
                      if (!is.null(X_test) & !is.null(Y_test)) {
                        
                        # Predict
                        test_pred <- predict(forest, X_test, type = 'response')
                        test_misclass <- 1 - mean(test_pred == Y_test)
                        
                        # Print result for row binding
                        data.frame(
                          train_error = train_misclass,
                          test_error = test_misclass,
                          forest_size = forest_size
                        )
                      } else {
                        
                        # Print result
                        data.frame(
                          train_error = train_misclass,
                          forest_size = forest_size
                        )
                      }
                    }
}

#------------------------------------------------------------------------------#
## GRADIENT BOOSTING
XGB <- function(X_train, Y_train,
                X_test = NULL, Y_test = NULL,
                grid,
                ncores = 10) {
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
  
  # Iterate through parameter grid
  results <- foreach(i = 1:nrow(grid), .combine = bind_rows) %do% {
    
    cat("Testing parameter combination", i, "of", nrow(grid), '\n')
    
    # Define parameters for current iteration    
    xgb_params <- list(
      nthread = 10,
      eta = grid[i, "learning_rate"],
      gamma = grid[i, "gamma"],
      max_depth = grid[i, "depth"],
      objective = 'binary:logistic',
      eval_metric = 'error'
    )
    
    # Begin training
    model <- xgboost(
      data = X_train,
      label = Y_train,
      params = xgb_params,
      nrounds = 250,
      early_stopping_rounds = 10,
      verbose = 0
    )
    
    # Evaluate on train set (keep this for identifying overfit)
    train_pred <- as.numeric(predict(model, X_train) >= 0.5)
    train_misclass <- 1 - mean(train_pred == Y_train)
    
    # Evaluate on test set if possible
    if (!is.null(X_test) & !is.null(Y_test)) {
      
      # Evaluate on test set
      preds <- as.numeric(predict(model, X_test) >= 0.5)
      test_misclass <- 1 - mean(preds == Y_test)
      
      # Print results for binding
      data.frame(
        train_error =  train_misclass,
        test_error = test_misclass,
        learning_rate = xgb_params$eta,
        gamma = xgb_params$gamma,
        depth = xgb_params$max_depth
      )
    } else {
      
      # Print results for binding
      data.frame(
        train_error = train_misclass,
        learning_rate = xgb_params$eta,
        gamma = xgb_params$gamma,
        depth = xgb_params$max_depth
      )
    }
  }
  
  return(results)
  
}

#------------------------------------------------------------------------------#
