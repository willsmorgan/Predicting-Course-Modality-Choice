## 1. Import and Preprocessing
libs <- c("magrittr", "xgboost", "randomForest", "data.table", "tidyverse")
lapply(libs, library, character.only = TRUE)
#------------------------------------------------------------------------------#

# Import
logit <- read_csv("Results/Parameter Search/logit.csv")
rf <- read_csv("Results/Parameter Search/rf.csv")
boost <- read_csv("Results/Parameter Search/boost.csv")


## XGB
training <- readRDS("Data/course choice training.Rds") 

testing <- readRDS("Data/course choice testing.Rds")

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ .-1, training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ .-1, testing)
Y_test <- testing[, "icourse"] %>% unlist()


# Force Y to numeric if inputted as factor
if (is.factor(Y_train)) {
  Y_train <- as.numeric(levels(Y_train))[Y_train]
}

# Force Y to numeric if inputted as factor
if (is.factor(Y_test)) {
  Y_test <- as.numeric(levels(Y_test))[Y_test]
}

model <- xgboost(
  data = X_train,
  label = Y_train,
  params = list(
    nthread = 10,
    eta = 0.5,
    gamma = 0.25,
    max_depth = 10,
    objective = 'binary:logistic',
    eval_metric = 'error'
  ),
  nrounds = 250,
  verbose = 1
)

pr <- predict(model, X_test)
plot(density(pr))

xgb.importance(model$feature_names, model)

table(pr > 0.5, Y_test)

mean(as.numeric(pr > 0.5) == Y_test)

plot <- xgb.plot.tree(model = model, trees = 249, render = FALSE)
DiagrammeR::export_graph(plot, 'Graphics/boost_example.pdf', width = 1080, height = 1920)
#------------------------------------------------------------------------------#

## RF
training <- readRDS("Data/course choice training.Rds") 

testing <- readRDS("Data/course choice testing.Rds")

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ .-1, training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ .-1, testing)
Y_test <- testing[, "icourse"] %>% unlist()


rf <- randomForest(
  x = X_train,
  y = Y_train,
  ntree = 3000,
  mtry = sqrt(dim(X_train)[2]),
  nodesize = 50,
  keep.forest = FALSE,
  proximity = FALSE,
  do.trace = TRUE
)


rf_pr <- predict(rf, X_test, type = 'response')
mean(rf_pr == Y_test)

table(as.numeric(rf_pr > 0.5), Y_test)

