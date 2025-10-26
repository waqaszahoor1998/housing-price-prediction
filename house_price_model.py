"""
California House Price Prediction System
======================================

Complete machine learning system for predicting house prices with high accuracy.
Combines feature engineering, ensemble methods, and optimization techniques.

Usage: python house_price_model.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class HousePriceModel:
    """Complete house price prediction system with high accuracy"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                             'Population', 'AveOccup', 'Latitude', 'Longitude']
        self.enhanced_features = None
        
    def load_data(self):
        """Load California housing dataset"""
        print("Loading California Housing Dataset...")
        housing = fetch_california_housing()
        
        self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.data['target'] = housing.target
        
        print(f"Dataset: {len(self.data):,} houses, {len(self.feature_names)} features")
        return self.data
    
    def engineer_features(self):
        """Create advanced features for better prediction"""
        print("Engineering features...")
        
        # Basic engineered features
        self.data['rooms_per_household'] = self.data['AveRooms'] / self.data['AveOccup']
        self.data['bedrooms_per_room'] = self.data['AveBedrms'] / self.data['AveRooms']
        self.data['population_per_household'] = self.data['Population'] / self.data['AveOccup']
        self.data['income_per_room'] = self.data['MedInc'] / self.data['AveRooms']
        self.data['income_per_person'] = self.data['MedInc'] / self.data['AveOccup']
        
        # Geographic features
        self.data['distance_from_coast'] = np.sqrt(
            (self.data['Latitude'] - 34.0)**2 + (self.data['Longitude'] + 118.0)**2
        )
        self.data['is_coastal'] = (self.data['Latitude'] > 34) & (self.data['Longitude'] < -118)
        self.data['is_northern'] = self.data['Latitude'] > 36
        self.data['is_southern'] = self.data['Latitude'] < 34
        
        # Age and income features
        self.data['is_new_house'] = self.data['HouseAge'] < 10
        self.data['is_old_house'] = self.data['HouseAge'] > 30
        self.data['is_high_income'] = self.data['MedInc'] > 6.0
        self.data['is_low_income'] = self.data['MedInc'] < 3.0
        self.data['is_large_house'] = self.data['AveRooms'] > 6.0
        self.data['is_small_house'] = self.data['AveRooms'] < 4.0
        
        # Population density
        self.data['is_dense_area'] = self.data['Population'] > 2000
        self.data['is_sparse_area'] = self.data['Population'] < 500
        self.data['log_population'] = np.log1p(self.data['Population'])
        
        # Update feature list
        self.enhanced_features = [col for col in self.data.columns if col != 'target']
        
        print(f"Enhanced dataset: {len(self.data):,} houses, {len(self.enhanced_features)} features")
        return self.data
    
    def clean_data(self):
        """Clean data by removing outliers"""
        print("Cleaning data...")
        
        initial_size = len(self.data)
        
        # Remove outliers using IQR method
        for col in self.enhanced_features:
            if col in ['is_coastal', 'is_northern', 'is_southern', 'is_new_house', 
                      'is_old_house', 'is_high_income', 'is_low_income', 'is_large_house', 
                      'is_small_house', 'is_dense_area', 'is_sparse_area']:
                continue
                
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            
            self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
        
        final_size = len(self.data)
        removed = initial_size - final_size
        print(f"Removed {removed} outliers ({removed/initial_size*100:.1f}%)")
        print(f"Final dataset: {final_size:,} houses")
        
        return self.data
    
    def train_model(self):
        """Train ensemble model for high accuracy"""
        print("Training ensemble model...")
        
        # Prepare data
        X = self.data[self.enhanced_features]
        y = self.data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define optimized models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            ),
            'Ridge': Ridge(alpha=0.1),
            'Lasso': Lasso(alpha=0.01),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5)
        }
        
        # Train individual models
        individual_scores = {}
        trained_models = {}
        
        print("Training individual models:")
        for name, model in models.items():
            print(f"  - {name}...", end=" ")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            model.fit(X_train_scaled, y_train)
            
            # Test score
            y_pred = model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_pred)
            
            individual_scores[name] = test_r2
            trained_models[name] = model
            
            print(f"CV: {cv_scores.mean():.3f}, Test: {test_r2:.3f}")
        
        # Create ensemble from best models
        best_models = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        ensemble_models = [(name, trained_models[name]) for name, _ in best_models]
        
        self.model = VotingRegressor(ensemble_models)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate ensemble
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nEnsemble model trained successfully!")
        print(f"Accuracy: {r2:.1%} (R² = {r2:.3f})")
        print(f"RMSE: ${rmse * 100000:,.2f}")
        print(f"MAE: ${mae * 100000:,.2f}")
        
        # Show improvement
        best_individual = max(individual_scores.values())
        improvement = r2 - best_individual
        print(f"Improvement over best individual: +{improvement:.3f} ({improvement*100:.1f}%)")
        
        return r2, rmse, mae
    
    def predict_price(self, med_income, house_age, avg_rooms, avg_bedrooms, 
                     population, avg_occupancy, latitude, longitude):
        """Predict house price for given features"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Create feature vector
        features = self._create_feature_vector(
            med_income, house_age, avg_rooms, avg_bedrooms,
            population, avg_occupancy, latitude, longitude
        )
        
        # Scale and predict
        features_scaled = self.scaler.transform([features])
        predicted_price = self.model.predict(features_scaled)[0]
        
        return predicted_price * 100000
    
    def _create_feature_vector(self, med_income, house_age, avg_rooms, avg_bedrooms,
                              population, avg_occupancy, latitude, longitude):
        """Create feature vector with all engineered features"""
        # Basic features
        features = [med_income, house_age, avg_rooms, avg_bedrooms, 
                   population, avg_occupancy, latitude, longitude]
        
        # Engineered features
        features.extend([
            avg_rooms / avg_occupancy,  # rooms_per_household
            avg_bedrooms / avg_rooms,   # bedrooms_per_room
            population / avg_occupancy, # population_per_household
            med_income / avg_rooms,     # income_per_room
            med_income / avg_occupancy, # income_per_person
            np.sqrt((latitude - 34.0)**2 + (longitude + 118.0)**2),  # distance_from_coast
            int(latitude > 34 and longitude < -118),  # is_coastal
            int(latitude > 36),         # is_northern
            int(latitude < 34),         # is_southern
            int(house_age < 10),        # is_new_house
            int(house_age > 30),        # is_old_house
            int(med_income > 6.0),      # is_high_income
            int(med_income < 3.0),      # is_low_income
            int(avg_rooms > 6.0),       # is_large_house
            int(avg_rooms < 4.0),       # is_small_house
            int(population > 2000),     # is_dense_area
            int(population < 500),      # is_sparse_area
            np.log1p(population)        # log_population
        ])
        
        return features
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            return None
        
        if hasattr(self.model.estimators_[0][1], 'feature_importances_'):
            importances = self.model.estimators_[0][1].feature_importances_
            return dict(zip(self.enhanced_features, importances))
        return None
    
    def save_model(self, model_path='house_price_model.pkl', scaler_path='scaler.pkl'):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved: {model_path}")
        print(f"Scaler saved: {scaler_path}")

def main():
    """Main function - complete house price prediction system"""
    print("=" * 60)
    print("CALIFORNIA HOUSE PRICE PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize model
    model = HousePriceModel()
    
    # Complete pipeline
    model.load_data()
    model.engineer_features()
    model.clean_data()
    accuracy, rmse, mae = model.train_model()
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    importance = model.get_feature_importance()
    if importance:
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, imp) in enumerate(sorted_features[:10], 1):
            print(f"  {i:2d}. {feature:<20}: {imp:.1%}")
    
    # Example predictions
    print("\nExample Predictions:")
    examples = [
        ("Luxury Coastal Home", [8.0, 15, 7.0, 1.2, 1000, 2.5, 37.8, -122.4]),
        ("Affordable Inland Home", [2.5, 35, 4.5, 1.0, 2000, 3.5, 34.0, -118.0]),
        ("Family Suburban Home", [5.0, 25, 6.0, 1.1, 1500, 3.0, 35.5, -120.0])
    ]
    
    for name, features in examples:
        price = model.predict_price(*features)
        print(f"  {name:<20}: ${price:,.0f}")
    
    # Save model
    model.save_model()
    
    print("\n" + "=" * 60)
    print("SYSTEM READY!")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.1%} | Average Error: ±${mae * 100000:,.0f}")

if __name__ == "__main__":
    main()
