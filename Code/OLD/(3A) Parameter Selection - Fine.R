### Hyperparameter Selection - Fine Search
### Author: William Morgan

## Purpose:
# Evaluate results from fine search to determine best-performing hyperparameter
# values for each model

#------------------------------------------------------------------------------#

# 0. Setup and Import
rm(list = ls())

libs <- c("tidyverse", "data.table", "magrittr")
lapply(libs, library, character.only = TRUE)

# Import
logit <- read_csv("Results/Fine Search/logit.csv")
rf <- read_csv("Results/Fine Search/rf.csv")
boost <- read_csv("Results/Fine Search/boost.csv")

#------------------------------------------------------------------------------#

## 1. Individual Results

# Logit
logit %>%
  ggplot(aes(alpha, misclassification)) + 
  geom_point(aes(color = lambda)) +
  labs(x = "Mixing Parameter",
       y = "CV-Misclassification Rate",
       title = "Penalized Logit Results\n") +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Graphics/Fine Search/logit.png")

# RF Results
rf %>%
  ggplot(aes(num_trees, misclassification)) +
  geom_point() +
  scale_x_continuous(breaks = seq(0, 1500, by = 200)) +
  labs(x = "Number of Trees",
       y = "CV-Misclassification Rate",
       title = "Random Forest Results") + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Graphics/Fine Search/rf.png")

# Boosting Results
boost %>%
  ggplot(aes(factor(max_depth), misclassification)) +
  geom_point(aes(color = round(gamma, 2))) +
  labs(x = "Max Tree Depth",
       y = "CV-Misclassification Rate",
       title = "Boosting Results",
       color = "Gamma") + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Graphics/Fine Search/boost.png")

#------------------------------------------------------------------------------#

## 2. Aggregate Results
agg_results <- tibble(
  method = c("Logit", "Random Forest", "Boosting"),
  result = c(mean(logit$misclassification),
             mean(rf$misclassification),
             mean(boost$misclassification)))

agg_results %>%
  ggplot(aes(reorder(method, -result), result)) +
  geom_bar(stat = 'identity') +
  labs(x = "Method",
       y = "CV'd Misclassificaiton Rate",
       title = "Initial Model Performances") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylim(0, 0.5) +
  coord_flip()

ggsave("Graphics/Fine Search/overall performances.png")