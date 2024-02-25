# Hotel Booking Cancellation Prediction

This repository contains a machine learning project for predicting hotel booking cancellations.

## Dataset Description

The dataset used in this project contains information about hotel bookings, including features such as lead time, arrival date, number of guests, etc and the descriptions are explained in detail in beginning of the `notebook/notebook.ipynb`. The raw dataset can be found in `data/hotel.csv`, and a preprocessed version is available in `data/preprocessed_data.csv`.
The exploratory data analysis report performed on the raw data is available in `eda/data_profile_report.html`.

## Modeling Approach

We trained several machine learning models using scikit-learn, including logistic regression, random forest, and support vector machine (SVM). The models were trained on the preprocessed data, and their performance was evaluated using various evaluation metrics.

## Instructions for Use

To reproduce the results:

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Explore the notebooks in the `notebooks/` directory for data preprocessing, model training, and evaluation.
4. Run the notebooks or scripts to train the models and make predictions.

## Evaluation Metrics

We used accuracy, precision, recall, and F1-score as evaluation metrics to assess the performance of the models. Detailed evaluation results can be found in the model evaluation notebook.

## Future Work

In future iterations of this project, we plan to explore more advanced modeling techniques, such as neural networks, and conduct further analysis to improve prediction accuracy.

## Contact Information

For any questions or feedback, feel free to contact me at [your_email@example.com](mailto:your_email@example.com) or connect with me on [LinkedIn](https://www.linkedin.com/in/your_username/).
