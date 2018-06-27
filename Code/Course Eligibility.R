### Course Selection for Choice model

## Use the full data file to define a subset of courses that are reasonable enough
## to include in an exploratory analysis of iCourse takers. "Reasonable" is up for
## interpretation, but what seems intuitive is to choose courses that are 
## offered in both modalities. Because this will vary from term to term we will be   
## selecting items on a course-term level. Essentially, if a given course is offered
## as ICO and F2F in the same term, it will be included in this data set

#------------------------------------------------------------------------------#

## 0. Setup
#rm(list = ls(all = TRUE))

libs <- c("data.table", "tidyverse", "magrittr")
lapply(libs, library, character.only = T)

ps <- 'R:/Data/ASU Core/Student Data/Peoplesoft Data/'

#------------------------------------------------------------------------------#

## 1. Find Course-terms

# If the file exists, create it; otherwise exit this script
if (!file.exists("Data/choice model course list.csv")) {
    cat("File not found, creating list of eligible courses")
  
  # Use local data if available
  if (file.exists("Data/full_data.csv")) {
    fd <- fread("Data/full_data.csv",
                select = c("course", "strm", "crse_modality", "catalog_nbr"),
                verbose = FALSE)
  } else {
    fd <- fread(paste0(ps, "full_data.csv"),
                select = c("course", "strm", "crse_modality", "catalog_nbr"),
                verbose = FALSE)
  }
  
  # Create list of eligible courses
  course_list <- fd %>%
    filter(crse_modality %in% c("F2F", "ICOURSE"),
           catalog_nbr < 500) %>%
    group_by_all() %>%
    summarise(enrl_count = n()) %>%
    ungroup() %>%
    spread(crse_modality, enrl_count) %>%
    filter(!is.na(F2F) & !is.na(ICOURSE)) %>%
    mutate(enrl_total = F2F + ICOURSE) %>%
    select(-catalog_nbr)
  
  fwrite(course_list, "Data/choice model course list.csv")
  rm(fd)
  
} else {
  course_list <- fread("Data/choice model course list.csv") 
}


