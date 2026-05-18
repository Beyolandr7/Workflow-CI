import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'academic_performance_preprocessing', 'processed_dataset.csv')
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['final_exam_score'])
    y = df['final_exam_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
            
        print(f"Model berhasil dilatih dengan Run ID: {run_id}")