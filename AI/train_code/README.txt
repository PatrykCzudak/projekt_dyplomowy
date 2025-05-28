"""Klasyfikacja ryzyka akcji przy użyciu sieci neuronowych (TensorFlow/Keras).

Założenia
---------
* Źródłowe dane pochodzą z plików CSV wygenerowanych przez `yahoo_pipeline.py`
  (każdy ticker osobno lub scalone w `dataset.csv`).
* Ryzyko definiujemy heurystycznie na podstawie 20-dniowej historycznej zmienności
  *annualized*:
    - **wysokie**  (label = 2)  → vol > 0.40
    - **średnie**  (label = 1)  → 0.20 < vol ≤ 0.40
    - **niskie**   (label = 0)  → vol ≤ 0.20
* Cechy (feature engineering):
    - dzienna stopa zwrotu (`return`)            = Close.pct_change()
    - 5-dniowa średnia krocząca stopy zwrotu     = `return.rolling(5).mean()`
    - 20-dniowa średnia krocząca stopy zwrotu    = `return.rolling(20).mean()`
    - dzienna zmienność (`abs_return`)           = abs(return)
    - 20-dniowa historyczna zmienność (`hv20`)   = return.rolling(20).std() * sqrt(252)

Pipeline
--------
1. **build_feature_df(csv_dir, min_len)** → łączy surowe ceny w jeden DataFrame z cechami i etykietą.
2. **prepare_dataset(df)**                 → czyszczenie NaN, podział train/test, skalowanie.
3. **build_model(input_dim)**             → tworzy model Keras *Sequential*.
4. **train_model(model, X_train, y_train)**
5. **evaluate(model, X_test, y_test)**
6. Model zapisuje się do `risk_model.h5`, a scaler do `scaler.pkl`.

Uruchomienie
------------
```bash
pip install pandas numpy scikit-learn tensorflow
python risk_classifier.py --csv_dir data/ --epochs 25
```
"""