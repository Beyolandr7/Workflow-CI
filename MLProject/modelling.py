import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    # Path dataset disesuaikan secara dinamis agar kebal terhadap perbedaan CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'academic_performance_preprocessing', 'processed_dataset.csv')
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['final_exam_score'])
    y = df['final_exam_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate and log metrics
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Model berhasil dilatih dengan Run ID: {run_id}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")