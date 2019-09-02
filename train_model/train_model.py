from pathlib import Path

import click
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
import pickle

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')

def train_model(in_csv, out_dir):
    """
    1. Fetch the data
    2. Deduplicate Data
    3. Feature Selection
    4. Handle Missing Data
    5. Feature Engineering
    6. Train
    7. Save

    Parameters
    ----------
    in_csv: str
        the input csv file
    out_dir:
        directory where file should be saved to.

    Returns
    -------
    Model.pkl
    Persistence.pkl
    """

    # Fetch the data
    data = pd.read_csv(in_csv)

    # deduplicate data
    dedup_cols = ['taster_name', 'title', 'description']
    data = data.drop_duplicates(dedup_cols)

    # select the features to build the model on
    cols = ['price', 'points', 'country', 'province']
    data = data[cols].copy()

    # handle missing data
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.mean())

    # drop the rows where country and province are null
    # since our model is based on them
    data = data.dropna()

    # generate useful features
    COUNTRY = data.groupby('country')
    PROVINCE = data.groupby('province')
    data['price_per_country_mean'] = COUNTRY['price'].transform('mean')
    data['price_per_country_mean_diff'] = data['price'] - data[
                                            'price_per_country_mean']
    data['price_per_country_median'] = COUNTRY['price'].transform(
                                                            'median'
                                                        )
    data['price_per_country_median_diff'] = data['price'] - data[
                                            'price_per_country_median']
    data['price_per_province_mean'] = PROVINCE['price'].transform('mean')
    data['price_per_province_mean_diff'] = data['price'] - data[
                                            'price_per_province_mean']
    data['price_per_province_median'] = PROVINCE['price'].transform(
                                                            'median'
                                                        )
    data['price_per_province_median_diff'] = data['price'] - data[
                                            'price_per_province_median']
    data['points_per_country_mean'] = COUNTRY['points'].transform('mean')
    data['points_per_country_median'] = COUNTRY['points'].transform(
                                                            'median'
                                                        )
    data['points_per_province_mean'] = PROVINCE['points'].transform('mean')
    data['points_per_province_median'] = PROVINCE['points'].transform(
                                                                'median'
                                                            )

    # select useful features for model persistence
    country_cols = [
        'country',
        'price_per_country_mean',
        'price_per_country_mean_diff',
        'price_per_country_median',
        'price_per_country_median_diff',
        'points_per_country_mean',
        'points_per_country_median',
    ]
    province_cols = [
        'province',
        'price_per_province_mean',
        'price_per_province_mean_diff',
        'price_per_province_median',
        'price_per_province_median_diff',
        'points_per_province_mean',
        'points_per_province_median',
    ]
    var_persistence = data.drop_duplicates(['country', 'province'])
    country_persistence = var_persistence[country_cols].copy()
    province_persistence = var_persistence[province_cols].copy()
    country_dict = country_persistence.set_index('country').T.to_dict('list')
    province_dict = province_persistence.set_index('province').T.to_dict('list')
    # contains our persistence variables
    var_persistence = [country_dict, province_dict]

    # train model
    train_features = [
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
        'points_per_province_median'
    ]
    target_feature = 'points'
    train_data = data[train_features].values
    target_data = data[target_feature].values

    # train the model using cross validation of 10 folds
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(train_data):
        X_train, X_valid = train_data[train_index], train_data[test_index]
        y_train, y_valid = target_data[train_index], target_data[test_index]
        xgb_model = xgb.XGBRegressor(
                        n_estimators=1000,
                        max_depth=20,
                        importance_type="gain",
                        learning_rate=0.01,
                        n_jobs=4
                    )
        xgb_model.fit(X_train, y_train,
                    early_stopping_rounds=5,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="rmse",
                    verbose=True)

if __name__ == '__main__':
    train_model()
