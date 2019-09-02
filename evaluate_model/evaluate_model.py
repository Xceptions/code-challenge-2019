import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import click

@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
@click.option('--model-pkl')
@click.option('--pers-vars-pkl')

def evaluate_model(in_csv, out_dir, model_pkl, pers_vars_pkl):
    """
    1. Fetch the test dataset
    2. Fetch the model file
    3. Fetch the persistence file
    4. Prepare the test dataset just like the train
    5. Fill in the persistence variables
    6. Train the model.
    """

    with open(model_pkl, 'rb') as model:
        xgb_model = pickle.load(model)
    with open(pers_vars_pkl, 'rb') as pers:
        persistence = pickle.load(pers)
    data = pd.read_csv(in_csv)

    """
    Read the persistence variables and get the test data that
    are predictable
    """
    country_persistence = persistence[0]
    province_persistence = persistence[1]
    check_country = [x for x in country_persistence]
    check_province = [x for x in province_persistence]
    data = data.loc[data['country'].isin(check_country)]
    data = data.loc[data['province'].isin(check_province)]


    """
    Drop duplicates, selection model features, drop null values
    """
    test_data = data.drop_duplicates(['taster_name', 'title', 'description'])
    test_data = test_data.replace([np.inf, -np.inf], np.nan)
    test_data = test_data.fillna(data.mean())
    test_data = test_data[['price', 'points', 'country', 'province']].copy()
    test_data = test_data.dropna()


    """
    Fill in the engineered features
    """
    map_country_feat = {
        'price_per_country_mean': 0,
        'price_per_country_mean_diff': 1,
        'price_per_country_median': 2,
        'price_per_country_median_diff': 3,
        'points_per_country_mean': 4,
        'points_per_country_median': 5,
    }
    map_province_feat = {
        'price_per_province_mean': 0,
        'price_per_province_mean_diff': 1,
        'price_per_province_median': 2,
        'price_per_province_median_diff': 3,
        'points_per_province_mean': 4,
        'points_per_province_median': 5,
    }

    test_data['price_per_country_mean'] = test_data['country'].apply(
        lambda x: country_persistence[x][
            map_country_feat['price_per_country_mean']
        ]
    )
    test_data['price_per_country_mean_diff'] = test_data['price'] - test_data[
                                                        'price_per_country_mean']
    test_data['price_per_country_median'] = test_data['country'].apply(
        lambda x: country_persistence[x][
            map_country_feat['price_per_country_median']
        ]
    )
    test_data['price_per_country_median_diff'] = test_data['price'] - test_data[
                                                        'price_per_country_median']
    test_data['price_per_province_mean'] = test_data['province'].apply(
        lambda x: province_persistence[x][
            map_province_feat['price_per_province_mean']
        ]
    )
    test_data['price_per_province_mean_diff'] = test_data['price'] - test_data[
                                                        'price_per_province_mean']
    test_data['price_per_province_median'] = test_data['province'].apply(
        lambda x: province_persistence[x][
            map_province_feat['price_per_province_median']
        ]
    )
    test_data['price_per_province_median_diff'] = test_data['price'] - test_data[
                                                    'price_per_province_median']
    test_data['points_per_country_mean'] = test_data['country'].apply(
        lambda x: country_persistence[x][
            map_country_feat['points_per_country_mean']
        ]
    )
    test_data['points_per_country_median'] = test_data['country'].apply(
        lambda x: country_persistence[x][
            map_country_feat['points_per_country_median']
        ]
    )
    test_data['points_per_province_mean'] = test_data['province'].apply(
        lambda x: province_persistence[x][
            map_province_feat['points_per_province_mean']
        ]
    )
    test_data['points_per_province_median'] = test_data['province'].apply(
        lambda x: province_persistence[x][
            map_province_feat['points_per_province_median']
        ]
    )

    """
    Perform your predictions.
    These predictions will be used for the model evaluation
    """
    test_features = [
        'price',
        'price_per_country_mean',
        'price_per_country_mean_diff',
        'price_per_country_median',
        'price_per_country_median_diff',
        'price_per_province_mean',
        'price_per_province_mean_diff',
        'price_per_province_median',
        'price_per_province_median_diff',
        'points_per_country_mean',
        'points_per_country_median',
        'points_per_province_mean',
        'points_per_province_median',
    ]
    test_data_vals = test_data[test_features].values
    predictions = xgb_model.predict(test_data_vals)

    # save the values gotten from the predictions as a csv file