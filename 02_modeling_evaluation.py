import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def RMSPE(y_true, y_pred):
    """
    Calculates the Root Mean Square Percentage Error.
    Converts inputs to numpy arrays to prevent pandas index misalignment.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mask to avoid division by zero
    m = y_true != 0
    return np.sqrt(np.mean(np.square((y_true[m] - y_pred[m]) / y_true[m]))) * 100



def train_and_evaluate():
    
    
    
    df = pd.read_csv('processed_data.csv')


    
    X = df.drop(['Sales'], axis=1)
    y = df['Sales']

    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    #Split (70/30)
    train_size = int(len(X_scaled) * 0.7)
    
    X_train = X_scaled.iloc[:train_size]
    X_test = X_scaled.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    
    linear_model = LinearRegression()
    forest_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    
    linear_model.fit(X_train, y_train)
    
    
    forest_model.fit(X_train, y_train)

    print("\nEvaluating models on test data")
    y_pred_lin = linear_model.predict(X_test)
    y_pred_forest = forest_model.predict(X_test)

    print("\nLinear Regression Performance")
    print(f"R2 SCORE: {r2_score(y_test, y_pred_lin):.4f}")
    print(f"RMSPE:    {RMSPE(y_test, y_pred_lin):.2f}%")

    print("\nRandom Forest Performance")
    print(f"R2 SCORE: {r2_score(y_test, y_pred_forest):.4f}")
    print(f"RMSPE:    {RMSPE(y_test, y_pred_forest):.2f}%")

    print("\nExtracting Feature Importance (Random Forest)")
    important_features = forest_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': X.columns, 
        'Importance': important_features
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print("\nTop Features driving sales:")
    print(importance_df.to_string(index=False))

if __name__ == "__main__":
    train_and_evaluate()

