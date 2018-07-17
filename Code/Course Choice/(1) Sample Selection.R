### Sample Selection - Course Choice Model
### Author: William Morgan

## Purpose:
# Subset the full peoplesoft data set to extract relevant columns
# and observations for a model estimating a student's likelihood of enrolling
# in an iCourse

## List of drops:
# no post-baccs, law students, or graduate students
# students taking more than 24 credits
# students with unspecified gender
# courses with "ASU" as its subject
# courses worth more than 4 units (attempting to capture internships here)
# "Dynamic" session courses
# irregular grades (only want A-E + W)
# courses that are not lectures
# students past their 16th observed term
# missing residency values
# incoming_gpa > 4.33
# graduate courses
# terms before summer 11 and winter terms
# undefined ethnicity
# missing pell eligibility
# no hybrid courses


## Organization:
# 0. Setup - libraries, paths, import
# 1. Initial drops - drop specific observations based on unwanted values (see above list)
# 2. Course drops - keep courses that have at least one ICO and one F2F section in the same term
# 3. Feature creation - add new variables not already in data set
# 4. Final cleaning - missing values, outliers, and factor variables
# 5. Variable Summaries - log summaries of each variable for later inspection
# 6. Splitting and Standardizing

#------------------------------------------------------------------------------#

## 0. Setup
rm(list = ls())

libs <- c("tidyverse", "data.table", "magrittr", "caret")
lapply(libs, library, character.only = TRUE)

# Import
data <- fread("Data/full_data.csv")

# Set option for creating test set
create_test_set = TRUE

# Open Log
log <- file("Logs/Sample Selection.txt", open = "wt")
write_lines('Course Choice Sample Selection Log', path = log)

#------------------------------------------------------------------------------#
## 1. Feature creation

# Run script to create new features not already present in data
source("Code/Course Choice/(1A) Feature Creation.R")

#------------------------------------------------------------------------------#
## 2. Initial drops

cat("\nInitial number of observations: ", nrow(data), "\n", file = log)

data %<>%
  filter(strm >= 2114 & strm != 2149,
         acad_level %in% c("Freshman", "Sophomore", "Junior", "Senior"),
         course_load <= 24,
         gender != "U",
         stud_modality == "F2F",
         crse_modality %in% c("F2F", "ICOURSE"),
         catalog_nbr< 500,
         course_load <= 24,
         unt_taken <= 4,
         ssr_component == "LEC",
         nth_term <= 16,
         incoming_gpa <= 4.33,
         pell_eligibility != "",
         !ethnicity %in% c("NR", "Haw/Pac"),
         !is.na(first_gen),
         instruction_mode != "HY",
         crse_grade_off %in% c("A", "A-", "A+", "B", "B+", "B-",
                               "C", "C+", "C-", "D", "E", "W"),
         session_code %in% c("A", "B", "C"),
         !is.na(stem_degree))

cat("\nNumber of observations after initial drops: ", nrow(data), "\n", file = log)

#------------------------------------------------------------------------------#
## 3. Course Drops

# Run code to either create or load appropriate course list
source("Code/Course Choice/(1B) Course Eligibility.R")
course_list %<>% select(course, strm)

# Join course list
data %<>% semi_join(., course_list, by = c("course", "strm"))

rm(course_list)

cat("\nNumber of observations after selecting eligible courses: ", nrow(data), "\n", file = log)

#------------------------------------------------------------------------------#
## 4. Outliers, missing values, and factor variables

# Define levels for all factors
ac <- c("Freshman", "sophomore", "Junior", "Senior","Post-Bacc Undergraduate", "Non-degree Undergraduate")
cp <- c("TEMPE", "DTPHX", "POLY", "WEST")
eth <- c("White", "American I", "2 or More", "Asian", "Black", "Haw/Pac", "Hispanic", "NR")
sterm <- c("2114", "2117", "2121", "2124", "2127", "2131", "2134", "2137",
          "2141", "2144", "2147", "2151", "2154", "2157", "2161", "2164",
          "2167", "2171", "2174", "2177")
terms <- c("Fall", "Spring", "Summer")
years <- c("2011", "2012",  "2013", "2014", "2015", "2016", "2017", "2018")

# Factorize categorical variables and fix prev_term_gpa
data %<>%
  mutate(prev_term_gpa = if_else(is.na(prev_term_gpa), incoming_gpa, prev_term_gpa),  # replace missing prev_term_gpa with incoming gpa for freshmen
         prev_term_gpa2 = prev_term_gpa^2,
         acad_level = as_factor(acad_level, ac),
         campus = as_factor(campus, cp),
         ethnicity = as_factor(ethnicity, eth),
         strm = as_factor(as.character(strm), sterm),
         term = as_factor(term, terms),
         school_year = as_factor(as.character(school_year), years))


# Make sure indicator variables are factorized
ind_vars <- c("icourse", "first_gen", "in_state", "gender", "pell_eligible", 
              "required_course", "stem_degree", "took_course", "took_instructor", "transfer",
              "upper_division", "mult_degree", "include_in_gpa")

# Edit indicators to all be 1/0, then force to factor setting reference at 0
data %<>%
  mutate(pell_eligible = if_else(pell_eligibility == "Y", 1, 0),
         icourse = if_else(crse_modality == "ICOURSE", 1, 0),
         gender = if_else(gender == "M", 1, 0),
         include_in_gpa = if_else(include_in_gpa == "Y", 1, 0)) %>%
  mutate_at(vars(ind_vars), as.character) %>%
  mutate_at(vars(ind_vars), as_factor) %>%
  mutate_at(vars(ind_vars), function(x) relevel(x, "0"))

# Define columns to keep
cols <- c("acad_level", "age", "age2", "campus", "cdi", "cdi2", 
          "course_load", "ethnicity", "fed_efc", "first_gen", "gender", "icourse",
          "in_state", "include_in_gpa", "mult_degree",  "nth_term", "num_ico_taken",
          "pell_eligible", "prev_term_gpa", "prev_term_gpa2", "required_course", "stem_degree",
          "tot_req_crses_bot", "total_enrolled", "transfer", "trf_creds_first_term",
          "upper_division")

# Drop observations with missing values in any column
data %<>%
  select(one_of(cols)) %>%
  filter_all(all_vars(!is.na(.)))                        # HUGE sweep for all missing values

cat("\nNumber of observations after filtering on missing values: ", nrow(data), '\n', file = log)

#------------------------------------------------------------------------------#
## 5. Variable Summary Log

cat('\n\n', file = log)
cat("PRE-TRIMMING SUMMARIES\n", file = log)

## Make sure all factors have the reference level you want
data %>% 
  select_if(is.factor) %>%
  sapply(., levels)

# Send summaries of variables to the log
for (i in seq_along(data)){
  if (is.numeric(unlist(data[, i]))) {
    cat("\n", file = log)
    capture.output(summary(data[, i]), file = log)
  } else {
    cat("\n", names(data)[i], file = log)
    capture.output(table(data[, i]), file = log)
  }
}


## acad_level needs Freshmen as reference
## campus needs Tempe as reference
## fed_efc has upper values that need to be trimmed
## trnsfr_credit_accpt has negative values and absurdly high values

# Final cleaning
data %<>%
  mutate(acad_level = relevel(acad_level, "Freshman"),
         campus = relevel(campus, "TEMPE")) %>%
  filter(trf_creds_first_term >= 0,
         fed_efc <= quantile(fed_efc, 0.95),
         trf_creds_first_term <= quantile(trf_creds_first_term, 0.95)) %>%
  select(order(colnames(.)))

cat('\n\n\n', file = log)
cat("POST-TRIMMING SUMMARIES", "\n", file = log)

for (i in seq_along(data)){
  if (is.numeric(unlist(data[, i]))) {
    cat("\n", file = log)
    capture.output(summary(data[, i]), file = log)
  } else {
    cat("\n", names(data)[i], file = log)
    capture.output(table(data[, i]), file = log)
  }
}

# Close log
close(log)

#------------------------------------------------------------------------------#
## 6. Splitting and Standardizing

# Create partition and split for train/test
train_index <- createDataPartition(data$icourse,
                                   p = 0.7,
                                   list = FALSE)

training_set <- data[train_index, ]
val_set <- data[-train_index, ]

# Create second partition for test/val set
if (create_test_set) {
  test_index <- createDataPartition(val_set$icourse,
                                   p = 0.5,
                                   list = FALSE)
  
  testing_set <- val_set[test_index, ]
  val_set <- val_set[-test_index, ]
}

# Stdize to the training set and apply to testing
stdize_vals <- preProcess(training_set, method = c("center", "scale"))

training_set <- predict(stdize_vals, training_set)
val_set <- predict(stdize_vals, val_set)

if (create_test_set) {
  testing_set <- predict(stdize_vals, testing_set)
  saveRDS(testing_set, "Data/course choice testing.Rds")
}

# Export
saveRDS(training_set, "Data/course choice training.Rds")
saveRDS(val_set, "Data/course choice validation.Rds")
