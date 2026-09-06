
library(Rlgt)
library(forecast)
library(tidyverse)
library(moments)




























































read_tsf <- function(filepath) {
  # Read lines with explicit encoding
  lines <- readLines(filepath, encoding = "UTF-8", warn = FALSE)
  
  # Find where data starts
  data_line <- which(grepl("@data", lines, fixed = TRUE))
  
  if (length(data_line) == 0) {
    # Try without encoding
    lines <- readLines(filepath, warn = FALSE)
    data_line <- which(grepl("@data", lines, fixed = TRUE))
  }
  
  # Parse header information
  header_lines <- lines[1:(data_line - 1)]
  
  # Extract frequency, horizon, and start timestamp from header
  frequency <- NULL
  start_timestamp <- NULL
  horizon <- NULL
  has_start_timestamp_attr <- FALSE
  
  # Fixed frequency extraction in the main read_tsf function
  for (line in header_lines) {
    if (grepl("@frequency", line, fixed = TRUE)) {
      # Improved frequency extraction - handle multiple possible formats
      # Remove @frequency and any colons, spaces, then trim
      freq_part <- gsub("@frequency", "", line, fixed = TRUE)
      freq_part <- gsub(":", "", freq_part, fixed = TRUE)
      freq_part <- trimws(freq_part)
      
      if (nchar(freq_part) > 0) {
        frequency <- tolower(freq_part)
      }
    }
    
    if (grepl("@start_timestamp", line, fixed = TRUE)) {
      # This indicates the file format has start timestamps in header
      has_start_timestamp_attr <- TRUE
      # Similar fix for start_timestamp
      start_part <- gsub("@start_timestamp", "", line, fixed = TRUE)
      start_part <- gsub(":", "", start_part, fixed = TRUE)
      start_part <- trimws(start_part)
      
      if (nchar(start_part) > 0) {
        start_timestamp <- start_part
      }
    }
    
    if (grepl("@horizon", line, fixed = TRUE)) {
      # Similar fix for horizon
      horizon_part <- gsub("@horizon", "", line, fixed = TRUE)
      horizon_part <- gsub(":", "", horizon_part, fixed = TRUE)
      horizon_part <- trimws(horizon_part)
      
      if (nchar(horizon_part) > 0) {
        horizon <- as.numeric(horizon_part)
      }
    }
  }
  
  # Parse data lines
  data_lines <- lines[(data_line + 1):length(lines)]
  data_lines <- data_lines[data_lines != ""]
  
  # Detect data format by examining the first data line
  sample_line <- data_lines[1]
  parts <- strsplit(sample_line, ":")[[1]]
  
  # Determine format based on second field
  format_type <- "unknown"
  if (length(parts) >= 3) {
    second_field <- trimws(parts[2])
    
    # Check if second field looks like a date/timestamp
    if (grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}", second_field) || 
        grepl("^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}", second_field)) {
      format_type <- "with_timestamps"
    } else if (is.numeric(suppressWarnings(as.numeric(second_field)))) {
      format_type <- "with_horizon"
    }
  }
  
  # Parse each series based on detected format
  all_data <- list()
  
  for (line in data_lines) {
    # Split by colon
    parts <- strsplit(line, ":")[[1]]
    if (length(parts) >= 3) {
      series_name <- parts[1]
      
      if (format_type == "with_timestamps") {
        # Format: series_name:start_timestamp:values
        series_start_timestamp <- parts[2]
        values <- as.numeric(unlist(strsplit(parts[3], ",")))
        series_horizon <- if (!is.null(horizon)) horizon else 18  # Use header horizon or default
        
        # Generate timestamps from series start timestamp
        if (nchar(trimws(series_start_timestamp)) > 0 && 
            !grepl("1900-01-01", series_start_timestamp)) {
          # Use series-specific start timestamp (convert to Date objects)
          timestamps <- generate_timestamps_from_original(length(values), frequency, series_start_timestamp)
        } else {
          # For placeholder timestamps (1900-01-01), default to 1990-01-01 format
          if (grepl("1900-01-01", series_start_timestamp)) {
            # Keep the same format as original but start from 1990
            base_format <- series_start_timestamp
            new_timestamp <- sub("1900-01-01", "1990-01-01", base_format)
            timestamps <- generate_timestamps_from_original(length(values), frequency, new_timestamp)
          } else {
            timestamps <- generate_timestamps_from_original(length(values), frequency, "1990-01-01 00-00-00")
          }
        }
        
      } else {
        # Format: series_name:horizon:values
        series_horizon <- as.numeric(parts[2])
        values <- as.numeric(unlist(strsplit(parts[3], ",")))
        
        # Generate timestamps from 1990 or header start timestamp
        start_date <- if (!is.null(start_timestamp)) start_timestamp else "1990-01-01"
        timestamps <- generate_timestamps_from_date(length(values), frequency, start_date)
      }
      
      # Create dataframe
      df <- data.frame(
        series_name = series_name,
        start_timestamp = timestamps,  # Now these will be Date objects
        series_value = values,
        horizon = series_horizon,
        stringsAsFactors = FALSE
      )
      
      all_data[[length(all_data) + 1]] <- df
    }
  }
  
  # Combine all
  result <- do.call(rbind, all_data)
  
  # Print summary information
  cat("Format detected:", format_type, "\n")
  cat("Frequency:", frequency, "\n")
  cat("Series count:", length(unique(result$series_name)), "\n")
  cat("Timestamp class:", class(result$start_timestamp), "\n")
  
  return(result)
}

# Helper function to generate timestamps preserving original format but returning Date objects
generate_timestamps_from_original <- function(n_values, frequency, start_timestamp) {
  # Parse the original timestamp format
  # Example: "1990-01-01 00-00-00" or "1990-01-01"
  
  if (is.null(start_timestamp) || nchar(start_timestamp) == 0) {
    start_timestamp <- "1990-01-01 00-00-00"
  }
  
  # Extract date and time parts
  if (grepl(" ", start_timestamp)) {
    parts <- strsplit(start_timestamp, " ")[[1]]
    date_part <- parts[1]
    time_part <- parts[2]
    has_time <- TRUE
  } else {
    date_part <- start_timestamp
    time_part <- "00-00-00"
    has_time <- FALSE
  }
  
  # Parse date
  date_components <- strsplit(date_part, "-")[[1]]
  start_year <- as.numeric(date_components[1])
  start_month <- as.numeric(date_components[2])
  start_day <- as.numeric(date_components[3])
  
  # Parse time if present (format: HH-MM-SS)
  start_hour <- 0
  start_min <- 0
  start_sec <- 0
  if (has_time && nchar(time_part) > 0) {
    time_components <- strsplit(time_part, "-")[[1]]
    if (length(time_components) >= 1) start_hour <- as.numeric(time_components[1])
    if (length(time_components) >= 2) start_min <- as.numeric(time_components[2])
    if (length(time_components) >= 3) start_sec <- as.numeric(time_components[3])
  }
  
  # Create base datetime
  base_datetime <- as.POSIXct(paste(start_year, start_month, start_day, start_hour, start_min, start_sec, sep = "-"), 
                              format = "%Y-%m-%d-%H-%M-%S")
  
  # Only set default if frequency is truly NULL - don't override valid frequencies
  if (is.null(frequency)) {
    warning("Frequency is NULL, defaulting to monthly")
    frequency <- "monthly"  # Keep monthly as default for NULL cases
  }
  
  # Generate timestamp sequence based on frequency
  timestamps <- switch(tolower(frequency),  # Added tolower() for case insensitivity
                       "hourly" = {
                         # For hourly data, use POSIXct and return hourly sequence
                         seq(from = base_datetime, length.out = n_values, by = "hour")
                       },
                       "daily" = {
                         # Convert to Date for daily and longer frequencies
                         base_date <- as.Date(base_datetime)
                         seq(from = base_date, length.out = n_values, by = "day")
                       },
                       "weekly" = {
                         base_date <- as.Date(base_datetime)
                         seq(from = base_date, length.out = n_values, by = "week")
                       },
                       "monthly" = {
                         base_date <- as.Date(base_datetime)
                         seq(from = base_date, length.out = n_values, by = "month")
                       },
                       "quarterly" = {
                         base_date <- as.Date(base_datetime)
                         seq(from = base_date, length.out = n_values, by = "3 months")
                       },
                       "yearly" = {
                         base_date <- as.Date(base_datetime)
                         seq(from = base_date, length.out = n_values, by = "year")
                       },
                       # Default case handles unknown frequencies
                       {
                         warning(paste("Unknown frequency:", frequency, "- defaulting to monthly"))
                         base_date <- as.Date(base_datetime)
                         seq(from = base_date, length.out = n_values, by = "month")
                       }
  )
  
  return(timestamps)
}

# Helper function for backward compatibility
generate_timestamps_from_date <- function(n_values, frequency, start_date) {
  # Parse start date (format: "1990-01-01" or "1990")
  if (is.null(start_date) || nchar(start_date) == 0) {
    start_date <- "1990-01-01"
  }
  
  # Handle different date formats
  if (grepl("-", start_date)) {
    date_parts <- strsplit(start_date, "-")[[1]]
    start_year <- as.numeric(date_parts[1])
    start_month <- if(length(date_parts) >= 2) as.numeric(date_parts[2]) else 1
    start_day <- if(length(date_parts) >= 3) as.numeric(date_parts[3]) else 1
  } else {
    start_year <- as.numeric(start_date)
    start_month <- 1
    start_day <- 1
  }
  
  # Create base date or datetime
  if (tolower(frequency) == "hourly") {
    # For hourly, create POSIXct starting at midnight
    base_datetime <- as.POSIXct(paste(start_year, start_month, start_day, "00", "00", "00", sep = "-"), 
                                format = "%Y-%m-%d-%H-%M-%S")
  } else {
    base_date <- as.Date(paste(start_year, start_month, start_day, sep = "-"))
  }
  
  # Only set default if frequency is truly NULL
  if (is.null(frequency)) {
    warning("Frequency is NULL, defaulting to monthly")
    frequency <- "monthly"
  }
  
  # Generate timestamp sequence based on frequency
  timestamps <- switch(tolower(frequency),  # Added tolower() for case insensitivity
                       "hourly" = {
                         seq(from = base_datetime, length.out = n_values, by = "hour")
                       },
                       "daily" = {
                         seq(from = base_date, length.out = n_values, by = "day")
                       },
                       "weekly" = {
                         seq(from = base_date, length.out = n_values, by = "week")
                       },
                       "monthly" = {
                         seq(from = base_date, length.out = n_values, by = "month")
                       },
                       "quarterly" = {
                         seq(from = base_date, length.out = n_values, by = "3 months")
                       },
                       "yearly" = {
                         seq(from = base_date, length.out = n_values, by = "year")
                       },
                       # Default case
                       {
                         warning(paste("Unknown frequency:", frequency, "- defaulting to monthly"))
                         seq(from = base_date, length.out = n_values, by = "month")
                       }
  )
  
  return(timestamps)
}



m3_yearly_df<-read_tsf("") #file here

write.csv(m3_yearly_df, "m3_yearly.csv")



# Split dataset into train and test sets based on prediction length
split_train_test <- function(data, prediction_length, 
                             series_column = "series_name",
                             timestamp_column = "start_timestamp", 
                             value_column = "series_value") {
  
  # Validate required columns
  required_cols <- c(series_column, timestamp_column, value_column)
  if (!all(required_cols %in% names(data))) {
    stop(paste("Missing required columns. Expected:", paste(required_cols, collapse = ", ")))
  }
  
  # Standardize column names for processing
  data_std <- data
  names(data_std)[names(data_std) == series_column] <- "series_name"
  names(data_std)[names(data_std) == timestamp_column] <- "start_timestamp"
  names(data_std)[names(data_std) == value_column] <- "series_value"
  
  # Get unique series
  unique_series <- unique(data_std$series_name)
  n_series <- length(unique_series)
  
  message(paste("Splitting", n_series, "series with prediction length:", prediction_length))
  
  # Initialize lists to store results
  train_data_list <- list()
  test_data_list <- list()
  excluded_series <- character()
  
  # Process each series
  for (i in seq_along(unique_series)) {
    series_name <- unique_series[i]
    series_data <- data_std[data_std$series_name == series_name, ]
    
    # Sort by timestamp to ensure correct order
    series_data <- series_data[order(series_data$start_timestamp), ]
    
    n_obs <- nrow(series_data)
    
    # Check if series is long enough
    if (n_obs <= prediction_length) {
      excluded_series <- c(excluded_series, series_name)
      message(paste("Warning: Series", series_name, "has", n_obs, 
                    "observations but needs >", prediction_length, "- excluding"))
      next
    }
    
    # Calculate split point
    train_end_idx <- n_obs - prediction_length
    
    # Split the data
    train_series <- series_data[1:train_end_idx, ]
    test_series <- series_data[(train_end_idx + 1):n_obs, ]
    
    # Add split identifiers
    train_series$split <- "train"
    test_series$split <- "test"
    
    # Store in lists
    train_data_list[[i]] <- train_series
    test_data_list[[i]] <- test_series
  }
  
  # Combine all series
  train_df <- do.call(rbind, train_data_list)
  test_df <- do.call(rbind, test_data_list)
  
  # Restore original column names
  if (series_column != "series_name") {
    names(train_df)[names(train_df) == "series_name"] <- series_column
    names(test_df)[names(test_df) == "series_name"] <- series_column
  }
  if (timestamp_column != "start_timestamp") {
    names(train_df)[names(train_df) == "start_timestamp"] <- timestamp_column
    names(test_df)[names(test_df) == "start_timestamp"] <- timestamp_column
  }
  if (value_column != "series_value") {
    names(train_df)[names(train_df) == "series_value"] <- value_column
    names(test_df)[names(test_df) == "series_value"] <- value_column
  }
  
  # Create summary
  successful_series <- length(unique_series) - length(excluded_series)
  
  message(paste("\nSplit completed:"))
  message(paste("- Successful series:", successful_series))
  message(paste("- Excluded series:", length(excluded_series)))
  message(paste("- Train observations:", nrow(train_df)))
  message(paste("- Test observations:", nrow(test_df)))
  
  return(list(
    train = train_df,
    test = test_df,
    excluded_series = excluded_series,
    summary = list(
      prediction_length = prediction_length,
      total_series = n_series,
      successful_series = successful_series,
      excluded_series = length(excluded_series),
      train_obs = nrow(train_df),
      test_obs = nrow(test_df)
    )
  ))
}

# Helper function to validate split results
validate_split <- function(split_result, original_data, series_name = NULL) {
  
  if (is.null(series_name)) {
    series_name <- unique(original_data$series_name)[1]
  }
  
  # Get original, train, and test data for one series
  orig_series <- original_data[original_data$series_name == series_name, ]
  train_series <- split_result$train[split_result$train$series_name == series_name, ]
  test_series <- split_result$test[split_result$test$series_name == series_name, ]
  
  cat("=== Validation for series:", series_name, "===\n")
  cat("Original length:", nrow(orig_series), "\n")
  cat("Train length:", nrow(train_series), "\n")
  cat("Test length:", nrow(test_series), "\n")
  cat("Train + Test:", nrow(train_series) + nrow(test_series), "\n")
  cat("Matches original:", nrow(orig_series) == (nrow(train_series) + nrow(test_series)), "\n")
  
  if (nrow(train_series) > 0 && nrow(test_series) > 0) {
    cat("Last train timestamp:", as.character(max(train_series$start_timestamp)), "\n")
    cat("First test timestamp:", as.character(min(test_series$start_timestamp)), "\n")
  }
  
  return(invisible(TRUE))
}

# Example usage function
example_usage <- function() {
  # Example: Split with 18-month prediction horizon
  split_result <- split_train_test(
    data = cif_df,
    prediction_length = 18
  )
  
  # Access the results
  train_data <- split_result$train
  test_data <- split_result$test
  
  # Check what was excluded
  print(split_result$excluded_series)
  
  # Validate a specific series
  validate_split(split_result, cif_df, "T1")
  
  return(split_result)
}






# Split your data with 18-month prediction horizon
m3_yearly_split_result <- split_train_test(
  data = m3_yearly_df,
  prediction_length = 6
)

# Access the train and test sets
m3_yearly_train_data <- m3_yearly_split_result$train
m3_yearly_test_data <- m3_yearly_split_result$test

# Check the summary
m3_yearly_split_result$summary

# Validate the split for a specific series
validate_split(m3_yearly_split_result, m3_yearly_df, "T645")






