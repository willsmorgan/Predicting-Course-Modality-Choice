### Course Selection for Choice model

## Use the working data file to define a subset of courses that are reasonable enough
## to include in an exploratory analysis of iCourse takers. "Reasonable" is up for
## interpretation, but what seems intuitive is to choose courses that are 
## offered in both modalities. Because this will vary from term to term we will be   
## selecting items on a course-term level. Essentially, if a given course is offered
## as ICO and F2F in the same term, it will be included in this data set

#------------------------------------------------------------------------------#
## 0. Setup
libs <- c("data.table", "dplyr", "magrittr", "purrr", "stringr",
          "lubridate", "tidyr")
lapply(libs, library, character.only = TRUE)

#------------------------------------------------------------------------------#
## 1. Find Course-terms
print("Creating list of eligible courses")

# Create initial list of courses that have enrollments in both modalities
course_list <- data %>%
  select(course, strm, crse_modality, catalog_nbr) %>%
  group_by_all() %>%
  summarise(enrl_count = n()) %>%
  ungroup() %>%
  spread(crse_modality, enrl_count) %>%
  filter(!is.na(F2F) & !is.na(ICOURSE)) %>%
  mutate(enrl_total = F2F + ICOURSE) %>%
  select(-catalog_nbr)

# Add to that list by ensuring at least one open seat in each modality
enrl_caps <- fread("Data/enrollment_caps.csv") %>%
  rename_all(str_to_lower) %>%
  select(order(colnames(.)))

enrl_caps %<>% filter(!is.na(class_nbr),
                      catalog_nbr < 500,
                      location %in% c("DTPHX", "ICOURSE", "POLY", "TEMPE", "WEST")) %>%
  mutate(crse_modality = if_else(location == "ICOURSE", "ICOURSE", "F2F"),
         course = paste(subject, catalog_nbr, sep = '|')) %>%
  select(-location, -subject, -catalog_nbr) %>%
  arrange(strm, course, crse_modality)

# Open seat counts by modality
enrl_caps %<>%
  group_by(course, strm, crse_modality) %>%
  summarise(avail_seats = sum(enrl_cap),
            taken_seats = sum(enrl_tot)) %>%
  ungroup() %>%
  mutate(open_seats = avail_seats - taken_seats) %>%
  select(-avail_seats, -taken_seats)

enrl_caps %<>%
  mutate(open_seats_f2f = if_else(crse_modality == "F2F", open_seats, NA_integer_),
         open_seats_ico = if_else(crse_modality == "ICOURSE", open_seats, NA_integer_))

enrl_caps %<>% 
  group_by(course, strm) %>%
  mutate(open_seats_f2f = sum(open_seats_f2f, na.rm = TRUE),
         open_seats_ico = sum(open_seats_ico, na.rm = TRUE)) %>%
  slice(1) %>%
  select(-crse_modality, -open_seats) %>%
  filter(open_seats_f2f > 0,
         open_seats_ico > 0)

# Merge lists
course_list <- inner_join(course_list, enrl_caps, by = c("course", "strm"))
  
#------------------------------------------------------------------------------#
