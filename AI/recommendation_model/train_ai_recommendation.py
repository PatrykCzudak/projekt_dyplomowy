import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier, Pool

csv_path = Path('dataset.csv')
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values(['Symbol', 'Date'])


df['Return'] = df.groupby('Symbol')['Close'].pct_change()
df['Momentum'] = df.groupby('Symbol')['Close'].transform(lambda x: x / x.shift(20) - 1)
df['Return_lag1'] = df.groupby('Symbol')['Return'].shift(1)
df['Return_lag5'] = df.groupby('Symbol')['Return'].shift(5)
df['RollingMean_5'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).mean())
df['RollingStd_5'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).std())
exp12 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
exp26 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df['MACD'] = exp12 - exp26
df['Bollinger_Upper'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).mean() + 2 * x.rolling(window=20).std())
df['Bollinger_Lower'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).mean() - 2 * x.rolling(window=20).std())
df['Bollinger_Width'] = df['Bollinger_Upper'] - df['Bollinger_Lower']


def stochastic_oscillator(high, low, close, k=14):
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    return 100 * (close - lowest_low) / (highest_high - lowest_low)

df['Stochastic_14'] = df.groupby('Symbol').apply(
    lambda group: stochastic_oscillator(group['Close'], group['Close'], group['Close'], 14)
).reset_index(level=0, drop=True)


df['Volatility_20'] = df.groupby('Symbol')['Return'].transform(lambda x: x.rolling(window=20).std())


features = ['Return', 'Momentum', 'Return_lag1', 'Return_lag5', 'RollingMean_5', 'RollingStd_5',
            'MACD', 'Bollinger_Width', 'Stochastic_14', 'Volatility_20']
df = df.dropna(subset=features)


df['Future_Return_5d'] = df.groupby('Symbol')['Return'].shift(-1).rolling(window=5).mean()
df['Target'] = (df['Future_Return_5d'] > 0).astype(int)
df = df.dropna(subset=['Target'])


features_num = features
features_cat = ['Symbol']
target = 'Target'

X = df[features_num + features_cat]
y = df[target]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_num),
        ('cat', 'passthrough', features_cat)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(
        verbose=0,
        random_state=42
    ))
])


param_grid = {
    'classifier__iterations': [200, 500],
    'classifier__learning_rate': [0.01, 0.05],
    'classifier__depth': [4, 6, 8]
}

tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='accuracy',
    cv=tscv,
    verbose=2,
    n_jobs=-1
)

#Trenowanie
grid_search.fit(X, y, classifier__cat_features=[X.columns.get_loc(col) for col in features_cat])

print(f"Best params: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_:.4f}")


best_model = grid_search.best_estimator_.named_steps['classifier']
feature_names = (
    features_num +
    features_cat
)
importances = best_model.get_feature_importance(Pool(X, label=y, cat_features=[X.columns.get_loc(col) for col in features_cat]))
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nFeature Importance:")
print(feat_imp)


model_path = Path('recommendation_pipeline.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

print(f"Model zapisany w {model_path.resolve()}")
