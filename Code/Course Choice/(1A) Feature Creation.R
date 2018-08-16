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


## Student Features:
# multiple majors
# number of iCourses previously taken
# proximity of enrolling to start date

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


# Student Features (number of icourses taken, double major, number of days before start date when stud enrolled)
print("Starting Student Features...")
data %<>%
  mutate(icourse = if_else(crse_modality == "ICOURSE", 1, 0),
         temp = paste0(acad_major_2, acad_cert_1, acad_minor_1),
         mult_degree = if_else(acad_major_1 != "" & temp != "", 1, 0),
         age2 = age_at_class_start^2,
         cdi2 = course_diff_index^2) %>%
  group_by(emplid) %>%
  arrange(strm) %>%
  mutate(num_ico_taken = cumsum(icourse),
         num_ico_taken = lag(num_ico_taken)) %>%
  group_by(emplid, strm) %>%
  mutate(num_ico_taken = num_ico_taken[1],
         num_ico_taken = if_else(is.na(num_ico_taken), 0, num_ico_taken)) %>%
  ungroup() %>%
  select(-temp) %>%
  rename(age = age_at_class_start,
         cdi = course_diff_index) %>%
  mutate_at(vars(enrl_add_dt, start_dt), lubridate::as_datetime) %>%
  mutate(days_before_strt = as.numeric((start_dt- enrl_add_dt)/7))
  

