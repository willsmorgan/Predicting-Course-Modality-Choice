### Data Setup - Course Choice Model
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


## Organization:
# 0. Setup - libraries, paths, import
# 1. Initial drops - drop specific observations based on unwanted values (see above list)
# 2. Course drops - keep courses that have at least one ICO and one F2F section in the same term
# 3. Final cleaning - missing values, outliers, and factor variables
# 4. Variable Summaries - log summaries of each variable for later inspection
# 5. Splitting and Standardizing

#------------------------------------------------------------------------------#

## 0. Setup
rm(list = ls())

libs <- c("tidyverse", "data.table", "magrittr", "caret")
lapply(libs, library, character.only = TRUE)

# PS rootpath
ps_path <-  "R:\\Data\\ASU Core\\Student Data\\Peoplesoft Data\\"

# Import
data <- fread("Data/full_data.csv")

# Set option for creating test set
create_test_set = TRUE

#------------------------------------------------------------------------------#
## 1. Initial drops
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
         ethnicity != "NR",
         !is.na(first_gen),
         crse_grade_off %in% c("A", "A-", "A+", "B", "B+", "B-",
                               "C", "C+", "C-", "D", "E", "W"),
         session_code %in% c("A", "B", "C"))
         
#------------------------------------------------------------------------------#
## 2. Course Drops

# Run code to either create or load appropriate course list
source("Code/Course Eligibility.R")
course_list %<>% select(course, strm)

# Join course list
data %<>% inner_join(., course_list, by = c("course", "strm"))

#------------------------------------------------------------------------------#
## 3. Outliers, missing values, and factor variables

# Define levels for all factors
ac <- c("Freshman", "sophomore", "Junior", "Senior","Post-Bacc Undergraduate", "Non-degree Undergraduate")
cp <- c("TEMPE", "DTPHX", "POLY", "WEST")
eth <- c("White", "American I", "2 or More", "Asian", "Black", "Haw/Pac", "Hispanic", "NR")
sesh <- c("C", "B", "A")
sterm <- c("2114", "2117", "2121", "2124", "2127", "2131", "2134", "2137",
          "2141", "2144", "2147", "2151", "2154", "2157", "2161", "2164",
          "2167", "2171", "2174", "2177")

# Factorize categorical variables and fix prev_term_gpa
data %<>%
  mutate(prev_term_gpa = if_else(is.na(prev_term_gpa), incoming_gpa, prev_term_gpa),  # replace missing prev_term_gpa with incoming gpa for freshmen
         acad_level = as_factor(acad_level, ac),
         campus = as_factor(campus, cp),
         ethnicity = as_factor(ethnicity, eth),
         session = as_factor(session_code, sesh),
         strm = as_factor(as.character(strm), sterm))


# Make sure indicator variables are factorized
ind_vars <- c("icourse", "first_gen", "in_state", "gender", "pell_eligible", 
              "required_course", "took_course", "took_instructor", "transfer")

data %<>%
  mutate(pell_eligible = if_else(pell_eligibility == "Y", 1, 0),
         icourse = if_else(crse_modality == "ICOURSE", 1, 0),
         gender = if_else(gender == "M", 1, 0)) %>%
  mutate_at(vars(ind_vars), as.character) %>%
  mutate_at(vars(ind_vars), as_factor) %>%
  mutate_at(vars(ind_vars), function(x) relevel(x, "0"))

# Define columns to keep
cols <- c("acad_level","age_at_class_start", "campus","course_diff_index",
          "course_load", "ethnicity", "fed_efc", "first_gen", "gender", "icourse", "in_state",
          "nth_term", "pell_eligible", "prev_term_gpa", "required_course", "session",
          "strm", "took_instructor", "took_course", "transfer", "trnsfr_credit_accpt")

# Drop observations with missing values in any column
data %<>%
  select(cols) %>%
  filter_all(all_vars(!is.na(.)))                        # HUGE sweep for all missing values

#------------------------------------------------------------------------------#

## 4. Variable Summary Log

var_log <- file("Course Choice sample selection log.txt", open = "wt")
sink(var_log, type = "output")

cat("PRE-TRIMMING SUMMARIES", "\n")

## Make sure all factors have the reference level you want
data %>% 
  select_if(is.factor) %>%
  sapply(., levels)

# Check for outlandish values with a summary of all variables
sapply(data, function(x) {
  if (is.numeric(x)) {
    summary(x)
  } else {
    table(x)
  }
})

## acad_level needs Freshmen as reference
## campus needs Tempe as reference
## fed_efc has upper values that need to be trimmed
## trnsfr_credit_accpt has negative values and absurdly high values


# Final cleaning
data %<>%
  mutate(acad_level = relevel(acad_level, "Freshman"),
         campus = relevel(campus, "TEMPE")) %>%
  filter(trnsfr_credit_accpt >= 0,
         fed_efc <= quantile(fed_efc, 0.95),
         trnsfr_credit_accpt <= quantile(trnsfr_credit_accpt, 0.95)) %>%
  select(order(colnames(.)))

cat("POST-TRIMMING SUMMARIES", "\n")

sapply(data, function(x) {
  if (is.numeric(x)) {
    summary(x)
  } else {
    table(x)
  }
})

# close log
sink()

#------------------------------------------------------------------------------#
## 5. Splitting and Standardizing

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
