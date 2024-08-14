## Cars Price Prediction with Python
### What does it do?
Python implementation of data analysis and machine learning programs using a used vehicle purchase and sale dataset. The goal is to predict the selling price of vehicles.

### How to Use?

#### Clone the Repository
```bash
$ git clone https://github.com/alexandreclem/cars-price-prediction.git
```
#### Dependencies
- To install the dependencies, use the **requirements.txt** file present in the project folder.
    - Within the **project** folder, Run:
        ```bash
        $ pip install -r requirements.txt
        ``` 

#### Implementations & Datasets
- **src/eda.ipynb**
    - Exploratory data analysis to better understand the dataset and its features, as well as answer business questions

- **src/ml_model.py**
    - Preprocessing and feature engineering were performed, followed by the application of the Random Forest Regressor algorithm to predict the sales prices
- **src/predicted.csv**
    - Store the predicted sales price obtained from the model
- **src/datasets**
    - **src/datasets/cars_train.csv**
        - Labeled data used to train the model
    - **src/datasets/cars_test.csv**
        - Unlabeled data used to test the model
