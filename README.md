# Netflix Habits Analysis

## Overview

This project aims to analyze my own Netflix viewing history data to extract insights about viewing patterns, including the series watched, season and episode information, as well as the overall viewing behavior. The dataset used for this analysis contains information about titles watched, dates, and other metadata related to Netflix usage. 

The objective is to clean, process, and visualize the data to explore trends and patterns in the viewing history.

## Motivation

The motivation for this project is to gain insights into personal viewing habits on Netflix, focusing on:
- Identifying the most-watched series and episodes.
- Analyzing the frequency of watching specific seasons and episodes.
- Detecting trends and preferences in Netflix content consumption.
- Exploring how viewing habits may have changed over time.

## Data Source

The dataset for this project was extracted from Netflix's viewing history by scraping it using selenium, which can be downloaded directly from the user's Netflix account page.

- **Data Source**: [Netflix Viewing History (CSV format)](https://https://github.com/jjnazlica/ceydanazlica-netflix/NetflixViewingHistory.csv)
- **Data Columns**:
  - **Title**: The name of the watched series or movie.
  - **Date**: The date when the title was watched.

The original data contains information about the titles, but some entries may not have clear season or episode details for movies. For these cases, the title remains as a series name with no season or episode information.

## Data Processing (First Step)

1. **Data Loading**: 
   - The dataset is read from a CSV file (`NetflixViewingHistory.csv`) using the `pandas` library.

2. **Title Extraction**:
   - Titles containing season and episode information are split using a regular expression pattern.
   - If a title includes "Season", it is split into the `Series`, `Season`, and `Episode` columns.
   - Titles without "Season" (i.e., movies) are placed into the `Series` column with default values for `Season` and `Episode`.

3. **Missing Values Handling**:
   - For titles without season and episode details (movies/one time shows), the `Season` is filled with `'Movies'`, and also `Episode` is set to `'Movies'`.
   
4. **Data Type Conversion**:
   - The `Season` column is converted into a categorical data type for more efficient handling of season data.

5. **Data Merging**:
   - The processed `Series`, `Season`, and `Episode` columns are merged with the original data, and the cleaned DataFrame is generated.

## Findings

- Will be Implemented

## Limitations and Future Work

### Limitations:
- **Missing Data**: Some data entries was incomplete or corrupted.
- **Limited Information**: The dataset does not include information about the duration of views or other behavioral insights.
  
### Future Work:

- Will be Implemented

## Files and Directory Structure

