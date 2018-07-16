### Feature Creation - Course Choice Model
### Author: William Morgan

## Purpose:
# Use the full peoplesoft data set to create features that will be relevant
# in predicting the likelihood of taking an iCourse. This will also serve as a 
# preliminary way of cutting down the size of the file that will be used in selecting
# the appropriate sample

## Course Features:
# Class size
# Class level (lower division vs. upper division)
# Cumulative average grade received in the course/modality combination 

## Student Features:
# multiple majors
# retention from previous term (are you more likely to take an icourse when you are returning from a break?)
# number of iCourses previously taken
# "STEM" degree - done by looking at CIP code of primary degree

# NOTE: This script is not meant to be run independently from the sample selection
# script. 

#------------------------------------------------------------------------------#
## 0. Setup
libs <- c("tidyverse", "magrittr")
lapply(libs, library, character.only = TRUE)
#------------------------------------------------------------------------------#

## 1. Feature Creation

# Course features
print("Starting Course Features...")
data %<>%
  mutate(upper_division = if_else(catalog_nbr >= 300, 1, 0)) %>%
  group_by(strm, class_nbr) %>%
  mutate(total_enrolled = n()) %>%
  ungroup()

avg_grades <- data %>%
  select(course, crse_modality, strm, cgrade) %>%
  filter(!is.na(cgrade)) %>%
  group_by(course, crse_modality) %>%
  arrange(strm) %>%
  mutate(cummean_grade = cummean(cgrade)) %>%
  group_by(course, crse_modality, strm) %>%
  slice(1) %>%
  group_by(course, crse_modality) %>%
  mutate(cummean_grade_prev = lag(cummean_grade)) %>%
  select(course, crse_modality, strm, cummean_grade_prev) %>%
  ungroup()

data %<>% inner_join(., avg_grades, by = c("course", "crse_modality", "strm"))

rm(avg_grades)

# Student Features (number of icourses taken, double major)
print("Starting Student Features...")
data %<>%
  mutate(icourse = if_else(crse_modality == "ICOURSE", 1, 0)) %>%
  group_by(emplid) %>%
  arrange(strm) %>%
  mutate(double_major = if_else(acad_major_2 != "", 1, 0),
         num_ico_taken = cumsum(icourse),
         num_ico_taken = lag(num_ico_taken)) %>%
  group_by(emplid, strm) %>%
  mutate(num_ico_taken = num_ico_taken[1],
         num_ico_taken = if_else(is.na(num_ico_taken), 0, num_ico_taken)) %>%
  ungroup()
  

# THIS FEATURE IS TAKEN OUT BECAUSE IT DROPS TOO MANY OBSERVATIONS
# # Student features (retained from previous term)
# data %<>%
#   group_by(emplid) %>%
#   mutate(prev_observed_term = lag(strm)) %>%
#   group_by(emplid, strm) %>%
#   mutate(prev_observed_term = as.character(prev_observed_term[1]),
#          prev_temp = str_sub(prev_observed_term, 4)) %>%
#   ungroup() %>%
#   mutate(prev_term_retention = case_when(
#            term == "Spring" & prev_temp %in% c("4", "7")   ~ 1,
#            term == "Summer" & prev_temp == "7"             ~ 1,
#            term == "Fall" & prev_temp %in% c("1", "9")     ~ 1,
#            term == "Winter" & prev_temp == "1"             ~ 0,
#            graduation == 1 | is.na(prev_observed_term)     ~ NA_real_,
#            TRUE                                            ~ NA_real_
#          )) %>%
#   select(
#     -prev_temp) %>%
#   ungroup()

