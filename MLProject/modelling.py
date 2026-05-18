import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    # Path dataset disesuaikan dengan posisi file saat dijalankan
    df = pd.read_csv('academic_performance_preprocessing/processed_dataset.csv')
    
    X = df.drop(columns=['final_exam_score'])
    y = df['final_exam_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # BARIS TAMBAHAN: Memaksa MLflow menyimpan folder 'model'
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model berhasil dilatih dengan Run ID: {run.info.run_id}")