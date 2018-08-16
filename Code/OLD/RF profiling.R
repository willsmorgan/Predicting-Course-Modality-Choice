libs <- c("tidyverse", "magrittr", "doParallel", "foreach", 'profvis', 'randomForest',
          'data.table', 'caret', 'ranger')
lapply(libs, library, character.only = TRUE)
set.seed(18)

#------------------------------------------------------------------------------#

## 1. Import and Preprocessing
training <- readRDS("Data/course choice training.Rds")
testing <- readRDS("Data/course choice testing.Rds")

## for testing purposes only:
# training %<>% sample_frac(0.005)
# testing %<>% sample_frac(0.005)

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ ., training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ ., testing)
Y_test <- testing[, "icourse"] %>% unlist()

### Set parameters for the test
ncores <- 2
tree_sizes <- seq(100, 3000, by = 500)
folds <-  createFolds(Y_train, k = 10, list = FALSE)

#------------------------------------------------------------------------------#

# RF long vector issue test
rf <- randomForest(x = X_train,
                   y = Y_train,
                   ntree = 3000,
                   mtry = sqrt(dim(X_train)[2]),
                   proximity = FALSE,
                   keep.forest = FALSE)

rf <- ranger(formula = icourse ~  .,
             data = training,
             num.trees = 3000,
             mtry = sqrt(dim(training)[2]),
             num.threads = 8,
             verbose = TRUE)


ranger(dependent.variable.name = "Species", data = iris)


