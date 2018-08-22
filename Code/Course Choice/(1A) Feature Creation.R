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
### 0. Setup
libs <- c("tidyverse", "magrittr", "lubridate")
lapply(libs, library, character.only = TRUE)

#------------------------------------------------------------------------------#
### 1. Course Features
print("Starting Course Features...")

# Enrollment caps
enrl_caps <- fread("Data/enrollment_caps.csv") %>%
  rename_all(str_to_lower) %>%
  mutate(course = paste(subject, catalog_nbr, sep = '|')) %>%
  select(strm, class_nbr, enrl_cap)

data <- inner_join(data, enrl_caps, by = c("strm", "class_nbr"))

# UD/LD
data %<>%
  mutate(upper_division = if_else(catalog_nbr >= 300, 1, 0))

# Number of sections in a course by modality for a given term
sections <- data %>%
  select(course, strm, crse_modality, class_nbr) %>%
  group_by(course, strm, crse_modality) %>%
  mutate(sections_avail = n_distinct(class_nbr)) %>%
  slice(1) %>%
  ungroup() %>%
  select(-class_nbr)

# Reshape to wide
sections %<>%
  unite(temp, course, strm) %>%
  spread(crse_modality, sections_avail) %>%
  separate(., temp, c("course", "strm"), sep = "_") %>%
  mutate(strm = as.numeric(strm)) %>%
  select(course, strm, F2F, ICOURSE) %>%
  rename(num_sect_f2f = F2F,
         num_sect_ico = ICOURSE)

# join onto data
data %<>% inner_join(., sections, by = c("course", "strm"))

#------------------------------------------------------------------------------#
### 2. Student Features
print("Starting Student Features...")

# num_ico_taken, double major, number of days before start date
data %<>%
  mutate(icourse = if_else(crse_modality == "ICOURSE", 1, 0),
         temp = paste0(acad_major_2, acad_cert_1, acad_minor_1),
         mult_degree = if_else(acad_major_1 != "" & temp != "", 1, 0),
         age2 = age_at_class_start^2,
         cdi2 = course_diff_index^2,
         fed_efc = log(1 + fed_efc)) %>%
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
  mutate(days_before_strt = as.numeric((start_dt- enrl_add_dt)/24))
  

# Number of credits at date of enrollment
enrl <- fread("Data/student_enrollment_timestamps.csv") %>%
  rename_all(str_to_lower) %>%
  select(-session_code)

# Force non-drops to null drop date, get rid of CANC
enrl %<>%
  mutate(enrl_drop_dt = if_else(enrl_drop_dt == "1900-01-01 00:00:00", NA_character_, enrl_drop_dt)) %>%
  filter(enrl_status_reason != "CANC") %>%
  select(-enrl_status_reason) %>%
  as.data.table()

# Reshape to long
enrl <- melt(enrl,
             id.vars = c("emplid", "strm", "class_nbr", "unt_taken"),
             measure.vars = c("enrl_add_dt", "enrl_drop_dt"),
             variable.factor = FALSE)

enrl <- setnames(enrl, c("variable", "value"),
                 c("type", "timestamp")) 

# Change to datetime class
enrl <- enrl[, timestamp := as_datetime(timestamp)]

# Drop null timestamps, create col for tallying enrolled credits
enrl %<>%
  filter(!is.na(timestamp)) %>%
  mutate(taken_ind = if_else(type == "enrl_add_dt", 1, -1),
         val = unt_taken * taken_ind) 

# Cumsum enrolled credits
enrl %<>%
  arrange(emplid, strm, timestamp) %>%
  group_by(emplid, strm) %>%
  mutate(creds_at_enrl = cumsum(val),
         creds_at_enrl = lag(creds_at_enrl),
         creds_at_enrl = if_else(is.na(creds_at_enrl), 0, creds_at_enrl)) %>%
  ungroup() %>%
  select(-taken_ind, -val)

# Delete outliers
enrl %<>%
  mutate(low = if_else(creds_at_enrl < 0 , -1, 0),
         high = if_else(creds_at_enrl > 24 , 1, 0)) %>%   
  group_by(emplid, strm) %>%
  mutate(low = min(low),
         high = max(high)) %>%
  ungroup() %>%
  filter(low == 0 & high == 0)

# Final cleanup; grab last recorded add_dt for a student-term-course obs
enrl %<>%
  filter(type != "enrl_drop_dt") %>%
  select(emplid, strm, class_nbr, timestamp, creds_at_enrl) %>%
  arrange(emplid, strm, class_nbr, timestamp) %>%
  group_by(emplid, strm, class_nbr, timestamp) %>%
  slice(n()) %>%
  ungroup() %>%
  select(-timestamp)

data %<>% inner_join(., enrl, by = c("emplid", "strm", "class_nbr"))

if (create_OC) {
  ## Was the student living on campus
  on_campus <- fread("Data/on_campus_living.csv") %>%
    rename_all(str_to_lower) %>%
    group_by(emplid, strm) %>%
    slice(1) %>%
    ungroup() %>%
    mutate(on_campus = 1) %>%
    select(-booking_status, -check_in_dt)
  
  data <- left_join(data, on_campus, by = c("emplid", "strm")) %>%
    mutate(on_campus = if_else(strm >= 2154 & is.na(on_campus), 0, on_campus))
  
  rm(on_campus)
}


rm(enrl, enrl_caps, sections)