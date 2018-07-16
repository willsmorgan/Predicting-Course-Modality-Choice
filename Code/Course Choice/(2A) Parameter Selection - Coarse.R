### Hyperparameter Selection - Coarse Search
### Author: William Morgan

## Purpose:
# Take results of first round of cross-validation to determine if a finer grid
# should be used or a new grid altogether. If it appears that a model is performing
# best at the boundary of a particular hyperparameter, we'll have evidence for 
# expanding the grid

#------------------------------------------------------------------------------#

# 0. Setup and Import
rm(list = ls())

libs <- c("tidyverse", "data.table", "magrittr")
lapply(libs, library, character.only = TRUE)

# Import
logit <- read_csv("Results/Coarse Search/logit.csv")
lin_svm <- read_csv("Results/Coarse Search/lin_svm.csv")
rbf_svm <- read_csv("Results/Coarse Search/rbf_svm.csv")
rf <- read_csv("Results/Coarse Search/rf.csv")
boost <- read_csv("Results/Coarse Search/boost.csv")

#------------------------------------------------------------------------------#

## 1. Individual Results

# Logit
logit %>%
  ggplot(aes(lambda, misclassification)) + 
  geom_point(aes(color = alpha)) +
  labs(x = "Penalty Strength",
       y = "CV-Misclassification Rate",
       title = "Elastic Net Results\n") +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Graphics/Coarse Search/logit.png")

# Linear SVM
lin_svm %>%
  ggplot(aes(log10(cost), misclassification)) +
  geom_point() +
  labs(x = "log_10(cost)",
       y = "CV-Misclassification Rate",
       title = "Linear SVM Results") +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Graphics/Coarse Search/lin_svm.png")

# RBF SVM
# rbf_svm %>%
#   ggplot(aes(log10(cost), misclassification)) +
#   geom_point(aes(color = log10(gamma))) +
#   labs(x = "log_10(cost)",
#        y = "CV-Misclassification Rate",
#        title = "RBF SVM Results") +
#   theme(plot.title = element_text(hjust = 0.5))
# 
# ggsave("Graphics/Coarse Search/rbf_svm.png")

# RF Results
rf %>%
  ggplot(aes(num_trees, misclassification)) +
  geom_point() +
  scale_x_continuous(breaks = seq(0, 1500, by = 250)) +
  labs(x = "Number of Trees",
       y = "CV-Misclassification Rate",
       title = "Random Forest Results") + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Graphics/Coarse Search/rf.png")

# Boosting Results
boost %>%
  ggplot(aes(factor(max_depth), misclassification)) +
  geom_point(aes(color = factor(round(gamma, 2)))) +
  labs(x = "Max Tree Depth",
       y = "CV-Misclassification Rate",
       title = "Boosting Results") + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Graphics/Coarse Search/boost.png")

#------------------------------------------------------------------------------#

## 2. Aggregate Results

# agg_results <- tibble(
#   method = c("Logit", "Linear SVM", "RBF SVM",
#              "Random Forest", "Boosting"),
#   result = c(mean(logit$misclassification), mean(lin_svm$misclassification),
#              mean(rbf_svm$misclassification), mean(rf$misclassification),
#              mean(boost$misclassification)))
  
agg_results <- tibble(
  method = c("Logit", "Linear SVM", "Random Forest", "Boosting"),
  result = c(mean(logit$misclassification), mean(lin_svm$misclassification),
             mean(rf$misclassification), mean(boost$misclassification)))

agg_results %>%
  ggplot(aes(reorder(method, -result), result)) +
  geom_bar(stat = 'identity') +
  labs(x = "Misclassification Rate on CV Sets",
       y = "Method",
       title = "Initial Model Performance") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylim(0, 0.5) +
  coord_flip()

ggsave("Graphics/Coarse Search/Overall.png")