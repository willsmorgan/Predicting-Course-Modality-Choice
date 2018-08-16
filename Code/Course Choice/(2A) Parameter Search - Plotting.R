### Hyperparameter Selection - Fine Search
### Author: William Morgan

## Purpose:
# Evaluate results from search to determine best-performing hyperparameter
# values for each model

#------------------------------------------------------------------------------#
## 0. Setup and Import
libs <- c("tidyverse", "data.table", "magrittr")
lapply(libs, library, character.only = TRUE)

# Import
logit <- read_csv("Results/Parameter Search/logit.csv")
rf <- read_csv("Results/Parameter Search/rf.csv")
boost <- read_csv("Results/Parameter Search/boost.csv")

#------------------------------------------------------------------------------#
## 1. Individual Results

# Logit
ggplot(logit) +
  geom_line(aes(alpha, train_error, color = 'red')) +
  geom_line(aes(alpha, test_error, color = 'blue')) +
  labs(x = "Mixing Parameter",
       y = "Misclassification Error",
       title = "Logit Results") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.title = element_blank()) +
  scale_color_manual(values = c("red", "blue"),
                     labels = c("training", "testing"))

ggsave("Graphics/Parameter Search/logit.png")


# RF
ggplot(rf) +
  geom_line(aes(forest_size, train_error, color = 'red')) +
  geom_line(aes(forest_size, test_error, color = 'blue')) +
  labs(x = "Forest Size",
       y = "Misclassification Error",
       title = "Random Forest Results") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.title = element_blank()) +
  scale_color_manual(values = c("red", "blue"),
                     labels = c("training", "testing")) +
  ylim(0, 0.2)

ggsave("Graphics/Parameter Search/rf.png")

# Boost
temp <- boost %>%
  arrange(learning_rate, depth, desc(test_error)) %>%
  group_by(learning_rate, depth) %>%
  slice(1)

ggplot(temp, aes(factor(depth), test_error,
                 group = factor(learning_rate),
                 color = factor(round(learning_rate, 2)))) +
  geom_line() +
  labs(x = "Max Tree Depth",
       y = "Misclassification Error",
       title = "Boost Results",
       subtitle = "(Test Set Only)") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5)) +
  scale_color_discrete(name = "Learning Rate")

ggsave("Graphics/Parameter Search/boost.png")


#------------------------------------------------------------------------------#

## 2. Aggregate Results
agg_results <- tibble(
  method = c("Logit", "Random Forest", "Boosting"),
  result = c(min(logit$test_error),
             min(rf$test_error),
             min(boost$test_error)))

agg_results %>%
  ggplot(aes(reorder(method, -result), result)) +
  geom_bar(stat = 'identity') +
  labs(x = "Method",
       y = "Test Error Rate",
       title = "Model Performances on Test Set") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylim(0, 0.5) +
  coord_flip()

ggsave("Graphics/Parameter Search/overall performances.png")