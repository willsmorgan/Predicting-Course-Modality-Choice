## 1. Setup
libs <- c("magrittr", "xgboost", "randomForest",
          "data.table", "tidyverse", 'glmnet', 'parallel', "doParallel")
lapply(libs, library, character.only = TRUE)
#------------------------------------------------------------------------------#

## XGB
boost <- read_csv("Results/Parameter Search/boost.csv")

training <- readRDS("Data/course choice training.Rds")
testing <- readRDS("Data/course choice testing.Rds")
validation <- readRDS("Data/course choice holdout.Rds")

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ .-1, training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ .-1, testing)
Y_test <- testing[, "icourse"] %>% unlist()

X_val <- model.matrix(icourse ~ .-1, validation)
Y_val <- validation[, "icourse"] %>% unlist()


# Force Y to numeric if inputted as factor
if (is.factor(Y_train)) {
  Y_train <- as.numeric(levels(Y_train))[Y_train]
}

# Force Y to numeric if inputted as factor
if (is.factor(Y_test)) {
  Y_test <- as.numeric(levels(Y_test))[Y_test]
}

# Force Y to numeric if inputted as factor
if (is.factor(Y_val)) {
  Y_val <- as.numeric(levels(Y_val))[Y_val]
}

# Run Model on best parameter set
model <- xgboost(
  data = X_train,
  label = Y_train,
  params = list(
    nthread = 10,
    eta = 0.5,
    gamma = 0.25,
    max_depth = 1,
    objective = 'binary:logistic',
    eval_metric = 'logloss'
  ),
  nrounds = 500,
  verbose = 1
)

pr_test <- predict(model, X_test)
pr_val <- predict(model, X_val)

# Validation error
1 - mean(as.numeric(pr_val >= 0.5) == Y_val)

# Test error
1 - mean(as.numeric(pr_test >= 0.5) == Y_test)


# # Variable distributions b/w predicted classes
# yes <- validation[pr_val >= 0.5, ]
# no <- validation[pr_val < 0.5, ]
# 
# log <- file("Logs/Predicted Label Summary Stats.txt", open = "wt")
# write_lines('Variable distributions of predicted classes', path = log)
# 
# for (i in seq_along(no)){
#   if (is.numeric(unlist(no[, i]))) {
#     cat("\n", file = log)
#     cat("\n", "F2F", file = log)
#     capture.output(summary(no[, i]), file = log)
#     cat("\n", "ICO", file = log)
#     capture.output(summary(yes[, i]), file = log)
#   } else {
#     cat("\n", names(no)[i], file = log)
#     cat("\n", "F2F", file = log)
#     capture.output(table(no[, i]), file = log)
#     cat("\n", "ICO", file = log)
#     capture.output(table(yes[, i]), file = log)
#   }
# }
# 
# close(log)

# Model characteristics
xgb.importance(model$feature_names, model)
plot <- xgb.plot.tree(model = model, trees = 199, render = FALSE)
DiagrammeR::export_graph(plot, 'Graphics/boost_example.pdf', width = 1080, height = 1920)
#------------------------------------------------------------------------------#

## RF
rf <- read_csv("Results/Parameter Search/rf.csv")

training <- readRDS("Data/course choice training.Rds") 
testing <- readRDS("Data/course choice testing.Rds")

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ .-1, training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ .-1, testing)
Y_test <- testing[, "icourse"] %>% unlist()

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

ncores <- 8

rf <- parGrow(
  X_train,
  Y_train,
  100,
  proximity = FALSE,
  nodesize = 50
)




rf_pr <- predict(rf, X_test, type = 'response')
1 - mean(rf_pr == Y_test)

table(as.numeric(rf_pr > 0.5), Y_test)

#------------------------------------------------------------------------------#

## Logit
logit <- read_csv("Results/Parameter Search/logit.csv")

training <- readRDS("Data/course choice training.Rds") %>%
  select(-enrl_cap)

testing <- readRDS("Data/course choice testing.Rds")%>%
  select(-enrl_cap)

# Create matrices for model estimation
X_train <- model.matrix(icourse ~ .-1, training)
Y_train <- training[, "icourse"] %>% unlist()

X_test <- model.matrix(icourse ~ .-1, testing)
Y_test <- testing[, "icourse"] %>% unlist()

glm <- glmnet(X_train,
              Y_train,
              family = 'binomial',
              alpha = 1,
              standardize = FALSE,
              intercept = FALSE,
              lambda = 0.0005)

glm_pr <- predict(glm, X_test, type = 'class')

1 - mean(glm_pr == Y_test)

coef(glm)


lin <- glm(icourse ~ .-1, family = 'binomial', data = training)
