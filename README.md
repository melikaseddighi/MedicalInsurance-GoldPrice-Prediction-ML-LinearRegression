# Medical Insurance and Gold Price Prediction

This project is a machine learning application that predicts **Medical Insurance Charges** and **Gold Prices** using linear regression models. The application uses the `Tkinter` library to create a graphical user interface (GUI) where users can input values and visualize predictions, model performance, and data trends.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Data](#data)
  - [Medical Insurance Dataset](#medical-insurance-dataset)
  - [Gold Price Dataset](#gold-price-dataset)
- [Model Details](#model-details)
  - [Medical Insurance Prediction](#medical-insurance-prediction)
  - [Gold Price Prediction](#gold-price-prediction)

## Features

- **Medical Insurance Prediction**: Predicts insurance charges based on user inputs such as age, BMI, number of children, smoking habits, and region.
- **Gold Price Prediction**: Predicts the price of gold based on economic indicators like crude oil price, interest rates, and USD/INR exchange rate.
- **Visualizations**: Generates plots for actual vs. predicted values and visualizes data relationships through scatter and box plots.
- **Mean Squared Error**: Displays the model’s performance by showing the Mean Squared Error (MSE) for both prediction models.
- **User-friendly Interface**: Built with `Tkinter` for easy interaction with the models.

## Project Structure

```
.
├── insurance.csv                # Medical Insurance dataset
├── GF.csv                       # Gold Price dataset
├── main.py                      # Python script with Tkinter GUI and ML models
└── README.md                    # Project documentation
```

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - seaborn
  - matplotlib
  - scikit-learn
  - Tkinter (comes pre-installed with Python)


## Usage

1. Run the Python script:

   ```bash
   python main.py
   ```

2. The application GUI will open with two tabs:

   - **Medical Insurance Charges**: Enter details like age, BMI, smoking status, etc., and click "Predict" to get an estimated charge.
   - **Gold Price Prediction**: Enter economic factors like crude oil price, interest rate, etc., and click "Predict" to get an estimated gold price.

3. Use the buttons to plot data visualizations and view the model's Mean Squared Error.

## Data

### Medical Insurance Dataset

This dataset (`insurance.csv`) contains the following fields:

- `age`: Age of the individual
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of children/dependents
- `smoker`: Smoking status (yes/no)
- `region`: Residential region (northeast, northwest, southeast, southwest)
- `charges`: Final medical insurance charges billed by health insurance

### Gold Price Dataset

This dataset (`GF.csv`) includes the following economic indicators:

- `Crude_Oil`: Price of crude oil
- `Interest_Rate`: Interest rates
- `USD_INR`: USD/INR exchange rate
- `Sensex`: Indian stock market index
- `CPI`: Consumer Price Index
- `USD_Index`: USD index
- `Gold_Price`: Price of gold

## Model Details

### Medical Insurance Prediction

A **Linear Regression** model is trained on the insurance dataset to predict medical charges. The input features include age, BMI, smoking habits, children, region, and sex.

### Gold Price Prediction

A **Linear Regression** model is trained on the gold price dataset using crude oil price, interest rate, USD/INR rate, Sensex, CPI, and USD index to predict gold prices.
