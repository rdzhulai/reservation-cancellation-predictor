# Hotel Reservation Cancellation Predictor

## Overview

This project aims to predict hotel reservation cancellations using Support Vector Machine (SVM). The main objective is to develop a predictive model that helps hotels anticipate cancellations, allowing them to manage resources and optimize bookings effectively.

## Dataset Information

The dataset used in this project is the "Hotel Reservations Classification Dataset" available on Kaggle. It contains various features related to hotel reservations, including:

- **Booking_ID**: Unique identifier for each booking
- **no_of_adults**: Number of adults
- **no_of_children**: Number of children
- **no_of_weekend_nights**: Number of weekend nights (Saturday or Sunday) the guest stayed or booked
- **no_of_week_nights**: Number of week nights (Monday to Friday) the guest stayed or booked
- **type_of_meal_plan**: Type of meal plan booked by the customer
- **required_car_parking_space**: Whether the customer needs a car parking space (0 - No, 1 - Yes)
- **room_type_reserved**: Type of room reserved by the customer (encoded)
- **lead_time**: Number of days between the booking date and the arrival date
- **arrival_year**: Year of arrival
- **arrival_month**: Month of arrival
- **arrival_date**: Date of the month
- **market_segment_type**: Market segment designation
- **repeated_guest**: Whether the customer is a repeated guest (0 - No, 1 - Yes)
- **no_of_previous_cancellations**: Number of previous bookings canceled by the customer
- **no_of_previous_bookings_not_canceled**: Number of previous bookings not canceled by the customer
- **avg_price_per_room**: Average price per day of the booking (in euros)
- **no_of_special_requests**: Total number of special requests made by the customer (e.g., high floor, view from room)
- **booking_status**: Whether the booking was canceled (1) or not (0)

The dataset can be accessed [here](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset).

## Project Structure

The project is structured as follows:

- **data/**: Contains the dataset and any data preprocessing scripts.
- **notebooks/**: Jupyter notebooks for data exploration, analysis, and model development.
- **src/**: Source code for data processing, feature engineering, model training, and evaluation.
- **docs/**: Documentation and additional resources.

## Setup and Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Navigate to the project directory:
    ```bash
    cd project_directory
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

Usage

To run the project, follow these steps:

    Model Training and Evaluation:
        Open and run the Jupyter notebook for training and evaluating the model:
    ```bash
    jupyter notebook src/train_test_hotel_reservations.ipynb
    ```


## Evaluation Metrics

The performance of the predictive model is evaluated using the following metrics:

- **Accuracy**: The proportion of correctly predicted outcomes out of the total predictions.
- **Precision**: The proportion of correctly predicted positive outcomes out of the total predicted positives.
- **Recall**: The proportion of correctly predicted positive outcomes out of the actual positives.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.

The model's performance on the dataset is as follows:

- **Accuracy**: 80.10%
- **Precision**: 81.82%
- **Recall**: 90.04%
- **F1 Score**: 85.73%.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for more details.

## References

- [Dataset on Kaggle](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)

For more detailed information, please refer to the project's documentation in the `docs/` directory.
