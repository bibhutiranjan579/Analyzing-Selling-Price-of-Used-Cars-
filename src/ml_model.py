'''
> I used the original labeled dataset to separate it into training and testing so I could use supervised learning
> The training and unseen datasets went through:
    - Preprocessing:
        . Remove obviously not useful features
        . Handle missing values
        . The target (preco) was transform to float, removing the points within the number. The price in this dataset seems very weird
        . Define datatypes: Continuous and Categorical features
        . Remove categorical features with a lot of categories -> Prevent overfitting due to one-hot encoding
        . Remove categorical features with low entropy
        . Split the training data (original labeled dataset) into training and testing data
    - Feature Engineering
        . Normalize the continuous features: 'hodometro' and 'preco'
        . Apply one-hot encoding in the categorical features -> The majority of the categorical features doesnt have inherent order. Except, maybe for the 'ano_de_fabricacao' feature
        
> The chosen model was Random Forest Regressor
    - The problem is a regression problem. We want to predict cars sale price, which is a continuous variable.
    - Random Forests Regressors:
        . Pros:
            > Handles Both Numerical and Categorical Features: 
            > Handles many features
            > Less prone to overfitting
        . Cons:
            > Excessive Memory Usage
    - Performance metric chosen: MSE 
        . Stands for Mean Squared Error, and it is a common performance metric used to evaluate the accuracy of regression models. 
        . It measures the average squared difference between the predicted values and the actual values in the dataset.
'''


# Data manipulation
import pandas as pd
import re

# Data visualization
import matplotlib.pyplot as plt

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# ML Model 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def data_overview(dataframe):
    print('> Data Overview\n')
    print('- Dataset shape: ', dataframe.shape, '\n')

    print('- Dimensions: ', dataframe.ndim, '\n')

    print('- Overall Information:')
    print(dataframe.info(verbose=True, show_counts=True), '\n')

    print('- Features Types:\n', dataframe.dtypes, '\n')

    print('- Duplicated Rows:\n', dataframe.duplicated().any(), '\n')

    print('- Unique Values:\n', dataframe.nunique(), '\n')

    print('- Missing Values:\n', dataframe.isnull().sum(), '\n')


def preprocessing(cars_data_train, cars_data_unseen, test_percentage):
    # Removing the useless features
    useless_features = ['id', 'elegivel_revisao', 'veiculo_alienado', 'revisoes_dentro_agenda'] # Due to a lot of missing values and lack of interpretability i decided to remove the revisoes_dentro_agenda variable
    cars_data_train = cars_data_train.drop(columns=useless_features)
    cars_data_unseen = cars_data_unseen.drop(columns=useless_features)

    # Handling missing values
    missing_cols = ['num_fotos', 'dono_aceita_troca', 'veiculo_único_dono', 'revisoes_concessionaria', 'ipva_pago', 'veiculo_licenciado', 'garantia_de_fábrica']

    # Removing 'garantia_de_fábrica' due to over 85% missing values in both datasets. This information was obtained in the EDA 
    cars_data_train = cars_data_train.drop(columns=['garantia_de_fábrica'])
    cars_data_unseen = cars_data_unseen.drop(columns=['garantia_de_fábrica'])

    missing_cols = [x for x in missing_cols if x != 'garantia_de_fábrica']

    # My guessing is that these variables are binary, except the num_fotos feature. So, filling the missing values with a 'unk' (unknown) string value seems to be fine
    missing_cols = [x for x in missing_cols if x != 'num_fotos']

    for col in missing_cols:
        cars_data_train[col].fillna('unk', inplace=True)
        cars_data_unseen[col].fillna('unk', inplace=True)

    cars_data_train['num_fotos'].fillna(cars_data_train['num_fotos'].mean(), inplace=True)   # Filling with the mean
    cars_data_unseen['num_fotos'].fillna(cars_data_unseen['num_fotos'].mean(), inplace=True) # Filling with the mean

    # Types of features
    features = cars_data_train.columns.tolist()
    continuous_features = ['hodometro', 'preco'] # 2 Continuous Features
    categorical_features = [x for x in features if x not in continuous_features] # 23 Categorical Features

    # Converting the categorical features into categorical dtype
    for var in categorical_features:
        cars_data_train[var] = cars_data_train[var].astype('category')
    
    # Converting the preco variable to float    
    cars_data_train['preco'] = cars_data_train['preco'].str.replace('\.', '', regex=True).astype('float64') # 1st: Remove the dots from the variable and 2nd convert to float

    # Remove categorical features with many categories. This information was obtained in the EDA
    cars_data_train = cars_data_train.drop(columns=['versao', 'modelo', 'cidade_vendedor'])
    cars_data_unseen = cars_data_unseen.drop(columns=['versao', 'modelo', 'cidade_vendedor'])

    # Remove categorical features with low entropy. This information was obtained in the EDA
    cars_data_train = cars_data_train.drop(columns=['blindado'])
    cars_data_unseen = cars_data_unseen.drop(columns=['blindado'])

    # Removing possible duplicate rows created due to the removal of columns
    cars_data_train = cars_data_train.drop_duplicates()
    cars_data_train = cars_data_train.reset_index(drop=True) # adjusting the indices after performing the removal of duplicate rows

    # Split the data into training and testing sets (80% train, 20% test)
    target = cars_data_train['preco']
    cars_data_train = cars_data_train.drop(columns=['preco'])
    train_set, test_set, train_target, test_target = train_test_split(cars_data_train, target, test_size=test_percentage, random_state=42)
    
    train_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)
    train_target.reset_index(drop=True, inplace=True)
    test_target.reset_index(drop=True, inplace=True)
    cars_data_unseen.reset_index(drop=True, inplace=True)

    return train_set, test_set, train_target, test_target, cars_data_unseen


def feature_engineering(train_set, test_set, train_target, test_target, cars_data_unseen):
    # Normalizing (Min-Max scaling) the continuous features
    hodometro_scaler = MinMaxScaler()
    preco_scaler = MinMaxScaler()
    scalers = {
            'hodometro': hodometro_scaler,
            'preco': preco_scaler
    }

    # Normalizing the 'hodometro' feature
    hodometro_scaler = hodometro_scaler.fit(train_set[['hodometro']])
    train_set['hodometro'] = hodometro_scaler.transform(train_set[['hodometro']])
    test_set['hodometro'] = hodometro_scaler.transform(test_set[['hodometro']])
    cars_data_unseen['hodometro'] = hodometro_scaler.transform(cars_data_unseen[['hodometro']])

    train_target = pd.DataFrame({'preco': train_target}) # Convertion needed in order to use MinMaxScaler         
    test_target = pd.DataFrame({'preco': test_target})   # Convertion needed in order to use MinMaxScaler                  

    # Normalizing the 'preco' target
    preco_scaler = preco_scaler.fit(train_target[['preco']])
    train_target['preco'] = preco_scaler.transform(train_target[['preco']])
    test_target['preco'] = preco_scaler.transform(test_target[['preco']])

    train_target = train_target['preco'].squeeze()       # Convertion back to the series format
    test_target = test_target['preco'].squeeze()         # Convertion back to the series format

    # One hot encoding on the categorical features
    features_list = train_set.columns.tolist()
    categorical_features_list = [x for x in features_list if x != 'hodometro'] # List of columns to apply one-hot encoding

    # Extract the selected features as a new DataFrame
    categorical_features_train = train_set[categorical_features_list]
    categorical_features_test = test_set[categorical_features_list]
    categorical_features_unseen = cars_data_unseen[categorical_features_list]

    # Create an instance of OneHotEncoder and Fit only using the Training set
    ohe = OneHotEncoder(handle_unknown = 'ignore', sparse_output=False)
    ohe.fit(categorical_features_train)

    # Transform the selected features
    encoded_features_train = ohe.transform(categorical_features_train)
    encoded_features_test = ohe.transform(categorical_features_test)
    encoded_features_unseen = ohe.transform(categorical_features_unseen)

    # Convert the result back to a pandas DataFrame
    encoded_train = pd.DataFrame(encoded_features_train, columns=ohe.get_feature_names_out(categorical_features_list))
    encoded_test = pd.DataFrame(encoded_features_test, columns=ohe.get_feature_names_out(categorical_features_list))
    encoded_unseen = pd.DataFrame(encoded_features_unseen, columns=ohe.get_feature_names_out(categorical_features_list))

    # Concatenate the encoded features with the rest of the DataFrame
    train_set = pd.concat([train_set.drop(columns=categorical_features_list), encoded_train], axis=1)
    test_set = pd.concat([test_set.drop(columns=categorical_features_list), encoded_test], axis=1)
    cars_data_unseen = pd.concat([cars_data_unseen.drop(columns=categorical_features_list), encoded_unseen], axis=1)

    return train_set, test_set, train_target, test_target, cars_data_unseen, scalers


def denormalize_series(series, scalers, name):
    # Convert test_target, test_pred and unseen_pred back to DataFrame with a column named 'preco'
    series = pd.DataFrame({name: series})

    # Inverse transform the scaled target values to get the original values
    series[name] = scalers[name].inverse_transform(series[[name]])
    
    # Convert back to pandas Series format
    series = series[name].squeeze()

    return series
    

def ml_model_plots(test_target, test_pred, unseen_pred, scalers):
    test_target = denormalize_series(test_target, scalers, name='preco')
    test_pred = denormalize_series(test_pred, scalers, name='preco')
    unseen_pred = denormalize_series(unseen_pred, scalers, name='preco')

    # Plot the Target Price x Predicted Price
    instances = [i for i in range(len(test_target))]

    # Target Price x Predicted Price
    plt.figure()
    plt.plot(instances, test_target, label='Target Price')
    plt.plot(instances, test_pred, label='Predicted Price')

    # Add labels, title, and legend
    plt.xlabel('instance')
    plt.ylabel('price')
    plt.title('Target Price x Predicted Price - Test Set')
    plt.legend()

    # Plot the Price predicted for the Unseen Data
    instances = [i for i in range(len(unseen_pred))]

    # Create the second figure
    plt.figure()
    plt.plot(instances, unseen_pred)

    # Add labels, title, and legend
    plt.xlabel('instance')
    plt.ylabel('price')
    plt.title('Predicted Price - Unseen Data')

    plt.show()


def save_unseen_pred(unseen_pred, scalers):
    unseen_pred = denormalize_series(unseen_pred, scalers, name='preco')

    # Going back to the original format. Like this: 824.197.638.908.304
    unseen_pred = unseen_pred.astype('int64').astype(str)
    for index, value in unseen_pred.items():
        unseen_pred.iloc[index] = re.sub(r'(?<!^)(?=(\d{3})+$)', r'.', value)
        
    unseen_pred = pd.DataFrame({'preco': unseen_pred})    
    unseen_pred.to_csv('./predicted.csv', index=True, index_label='id')

    
def ml_model(train_set, test_set, train_target, test_target, cars_data_unseen, scalers):
    # Random Forest Regressor Model
    rf_regressor = RandomForestRegressor(
        n_estimators=100,           # Number of trees in the forest
        criterion='squared_error',  # The function to measure the quality of a split 
        max_features=0.15,          # Number of features to consider when looking for the best split
        random_state=42             # Ensure reproducibility
    )

    # Train the Model
    rf_regressor.fit(train_set, train_target)

    # Test the Model
    test_pred = rf_regressor.predict(test_set)

    # Evaluate the Model
    mse = mean_squared_error(test_target, test_pred)
    print(f'Test Set - Mean Squared Error: {mse:.2f}')
    
    # Predict - Unseen Data
    unseen_pred = rf_regressor.predict(cars_data_unseen)

    ml_model_plots(test_target, test_pred, unseen_pred, scalers)    
    save_unseen_pred(unseen_pred, scalers)
    

def main():
    # Loading the Data
    cars_data_train = pd.read_csv('./datasets/cars_train.csv', sep=';', encoding='utf-8') # LABELED DATA
    cars_data_unseen = pd.read_csv('./datasets/cars_test.csv', sep=';', encoding='utf-8') # UNSEEN DATA

    # Performing Preprocessing and Feature Engineering
    train_set, test_set, train_target, test_target, cars_data_unseen = preprocessing(cars_data_train, cars_data_unseen, test_percentage=0.2)
    train_set_norm, test_set_norm, train_target_norm, test_target_norm, cars_data_unseen_norm, scalers = feature_engineering(train_set, test_set, train_target, test_target, cars_data_unseen)

    # Apply the Machine Learning Model
    ml_model(train_set_norm, test_set_norm, train_target_norm, test_target_norm, cars_data_unseen_norm, scalers)

    
if __name__ == '__main__':
    main()
