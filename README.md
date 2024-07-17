# Laptop Price Predictor

## Overview

The Laptop Price Predictor is a web application built with Streamlit that estimates the price of a laptop based on its specifications. Using a machine learning model trained on historical laptop data, this app provides accurate price predictions based on user input.

## Features

- **User-friendly Interface**: Intuitive dropdowns and input fields for selecting laptop configurations.
- **Price Prediction**: Get instant price estimates based on selected specifications.
- **Custom Styling**: A modern design for a seamless user experience.

## Technologies Used

- **Python**: The primary programming language for the backend.
- **Streamlit**: A framework for building interactive web applications.
- **XGBoost**: A powerful gradient boosting library used for regression tasks.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Pickle**: For loading the pre-trained model and dataset.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/laptop-price-predictor.git
   cd laptop-price-predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the pre-trained model (`pipe.pkl`) and dataset (`df.pkl`) in the project directory.

## Running the Application

To run the application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open a new tab in your web browser where you can interact with the application.

## Usage

1. Select the laptop specifications from the provided options.
2. Click the **Predict Price** button.
3. The predicted price will be displayed below the input fields.

## Contributing

If you'd like to contribute to the project, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Thanks to the open-source community for the libraries and tools used in this project.
```

Feel free to modify any section to better match your project specifics!