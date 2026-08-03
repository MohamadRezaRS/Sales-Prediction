# Sales Prediction Pipeline 

## Overview

**This repository contains a professional data mining pipeline designed to predict future retail sales based on historical data.**

### Academic Context: This project was originally developed in Fall 2024 as an academic project for the Introduction to Data Mining course at K.N.Toosi University of Technology.

# Project Structure

The project is divided into two main Python scripts to separate data engineering from machine learning, ensuring clean, readable, and reusable code:

`01_data_preparation.py`: Handles the ETL (Extract, Transform, Load) process. It merges the raw store and sales datasets, extracts temporal features (Year, Month, Day, WeekOfYear), handles missing values via median imputation, and chronologically sorts the data. Outputs a clean processed_data.csv.

`02_modeling_evaluation.py`: Ingests the processed data, scales the features using StandardScaler, and performs a strict 70/30 chronological train-test split. It trains both a Linear Regression and a Random Forest Regressor model, evaluating them using R² Score and a custom RMSPE (Root Mean Square Percentage Error) function. Finally, it extracts and prints the most important features driving sales.

