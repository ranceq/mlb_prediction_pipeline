import argparse
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    # Load all years' features except current year for training
    X_list, y_list = [], []
    for fname in os.listdir(args.features_dir):
        if fname.startswith('features_'):
            year = int(fname.split('_')[1].split('.')[0])
            df = pd.read_parquet(os.path.join(args.features_dir, fname))
            X_list.append(df.drop('home_win', axis=1))
            y_list.append(df['home_win'])
    X = pd.concat(X_list)
    y = pd.concat(y_list)

    tss = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        {
            'n_estimators':[100,200], 'max_depth':[3,5],
            'learning_rate':[0.01,0.1], 'subsample':[0.8,1.0]
        },
        cv=tss, scoring='accuracy'
    )
    grid.fit(X, y)
    best = grid.best_estimator_
    # Save model
    best.save_model(os.path.join(args.run_dir, 'model.json'))

    # Evaluate on latest year
    latest = sorted([f for f in os.listdir(args.features_dir)])[-1]
    df_test = pd.read_parquet(os.path.join(args.features_dir, latest))
    X_test = df_test.drop('home_win', axis=1)
    y_test = df_test['home_win']
    preds = best.predict(X_test)
    print('Test year:', latest)
    print('Acc:', accuracy_score(y_test, preds))
    print('F1 :', f1_score(y_test, preds))