library(hansard)
library(dplyr)
library(glue)
start_date <- "2023-01-01"
end_date <- "2023-01-20"
file_path <- glue("haystacks_expanded/data/01_raw/parlquest_{start_date}_{end_date}.csv")

x <- all_answered_questions(start_date = start_date, end_date = end_date)

# Get the class of each column
column_classes <- sapply(x, class)

# Identify which columns are of type 'list'
list_columns <- names(column_classes[column_classes == "list"])

print(list_columns)

# Drop those columns
x <- x %>%
  select(-all_of(list_columns))

write.csv(x, file=file_path, row.names=FALSE)



