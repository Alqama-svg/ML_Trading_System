"""
ML Model Implementation for Trading System
Days 1-2: ML Model Implementation
• Feature engineering for financial data
• Time series forecasting models
• Reinforcement learning for trading
• Model validation techniques
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLTradingModels:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def prepare_training_data(self, data, target_column='target'):
<<<<<<< HEAD
        """
        Prepare training data with proper labels
        """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
        if data is None or data.empty:
            raise ValueError("No data provided for training")
            
        # Create target variable (next day return > 0)
        data = data.copy()
        data['future_return'] = data['Close'].shift(-1) / data['Close'] - 1
        data[target_column] = (data['future_return'] > 0.001).astype(int)  # 0.1% threshold
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Define feature columns (exclude target and price columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return', target_column]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No feature columns found. Ensure features are created first.")
            
        self.feature_columns = feature_cols
        
        X = data[feature_cols]
        y = data[target_column]
        
        return X, y
    
    def train_models(self, X, y):
        """
        Train multiple ML models
        """
        print("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        # Train and evaluate models
        model_scores = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            model_scores[name] = accuracy
            
            print(f"{name} accuracy: {accuracy:.4f}")
            
            # Store model
            self.models[name] = model
        
        # Select best model
        best_model_name = max(model_scores.items(), key=lambda x: x[1])[0]
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"Best model: {best_model_name} with accuracy: {model_scores[best_model_name]:.4f}")
        
        self.is_trained = True
        return model_scores
    
    def predict(self, features):
<<<<<<< HEAD
        """
        Make predictions using the best model
        """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
        if not self.is_trained:
            # Return random prediction if not trained
            return {
                'prediction': np.random.choice([0, 1]),
                'probability': np.random.uniform(0.4, 0.9),
                'signal': 'HOLD'
            }
        
        try:
            # Ensure features match training columns
            if isinstance(features, pd.DataFrame):
                features = features[self.feature_columns]
            
            # Scale features if using logistic regression
            if self.best_model_name == 'logistic_regression':
                features_scaled = self.scaler.transform(features)
                pred = self.best_model.predict(features_scaled)[0]
                prob = self.best_model.predict_proba(features_scaled)[0].max()
            else:
                pred = self.best_model.predict(features)[0]
                prob = self.best_model.predict_proba(features)[0].max()
            
            # Convert to trading signal
            if pred == 1 and prob > 0.6:
                signal = 'BUY'
            elif pred == 0 and prob > 0.6:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'prediction': pred,
                'probability': prob,
                'signal': signal
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'prediction': 0,
                'probability': 0.5,
                'signal': 'HOLD'
            }
    
    def save_models(self, filepath='models/'):
<<<<<<< HEAD
        """
        Save trained models to disk
        """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
        os.makedirs(filepath, exist_ok=True)
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'best_model_name': getattr(self, 'best_model_name', 'random_forest'),
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        with open(f'{filepath}/trading_models.pkl', 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Models saved to {filepath}/trading_models.pkl")
    
    def load_models(self, filepath='models/trading_models.pkl'):
<<<<<<< HEAD
        """
        Load trained models from disk
        """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.best_model_name = model_data.get('best_model_name', 'random_forest')
            self.best_model = self.models.get(self.best_model_name)
            self.feature_columns = model_data.get('feature_columns', [])
            self.is_trained = model_data.get('is_trained', False)
            
            print(f"Models loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

class TimeSeriesForecaster:
<<<<<<< HEAD
    """
    Time series forecasting for price prediction
    """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.model = None
        self.scaler = StandardScaler()
        
    def create_sequences(self, data, target_col='Close'):
<<<<<<< HEAD
        """
        Create sequences for time series prediction
        """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
        sequences = []
        targets = []
        
        for i in range(self.window_size, len(data)):
            sequences.append(data[i-self.window_size:i])
            targets.append(data[i])
        
        return np.array(sequences), np.array(targets)
    
    def train(self, price_data):
<<<<<<< HEAD
        """
        Train time series model (simplified version)
        """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
        # Normalize data
        scaled_data = self.scaler.fit_transform(price_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Simple linear model for demonstration
        from sklearn.linear_model import LinearRegression
        
        # Flatten sequences for linear regression
        X_flat = X.reshape(X.shape[0], -1)
        
        self.model = LinearRegression()
        self.model.fit(X_flat, y)
        
        return self.model.score(X_flat, y)
    
    def predict_next(self, recent_data):
<<<<<<< HEAD
        """
        Predict next price
        """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
        if self.model is None:
            return None
            
        # Scale recent data
        scaled_data = self.scaler.transform(recent_data.reshape(-1, 1)).flatten()
        
        if len(scaled_data) < self.window_size:
            return None
        
        # Get last window
        last_sequence = scaled_data[-self.window_size:].reshape(1, -1)
        
        # Predict
        scaled_prediction = self.model.predict(last_sequence)[0]
        
        # Inverse transform
        prediction = self.scaler.inverse_transform([[scaled_prediction]])[0][0]
        
        return prediction

class ReinforcementLearningTrader:
<<<<<<< HEAD
    """
    Simple RL trader implementation
    """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.positions = 0
        self.action_space = ['BUY', 'SELL', 'HOLD']
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        
    def get_state(self, features):
        """
        Convert features to state representation
        """
        # Simplified state representation
        if isinstance(features, pd.Series):
            return tuple(np.round(features.values, 2))
        return tuple(np.round(features, 2))
    
    def get_action(self, state):
<<<<<<< HEAD
        """
        Get action using epsilon-greedy policy
        """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_table(self, state, action, reward, next_state):
<<<<<<< HEAD
        """
        Update Q-table using Q-learning
        """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

def validate_model(model, X_test, y_test):
<<<<<<< HEAD
    """
    Model validation with various metrics
    """
=======
>>>>>>> f1909685739746bbe77927120694a7980b73754a
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return {
        'accuracy': accuracy,
        'classification_report': report
<<<<<<< HEAD
    }
=======
    }
>>>>>>> f1909685739746bbe77927120694a7980b73754a
