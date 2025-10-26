"""
House Price Prediction Tool
==========================

Simple tool to predict house prices using the trained model.

Usage: python predict.py
"""

import joblib
import numpy as np

def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('house_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("Model not found. Please run 'python house_price_model.py' first.")
        return None, None

def predict_house_price(med_income, house_age, avg_rooms, avg_bedrooms, 
                       population, avg_occupancy, latitude, longitude):
    """Predict house price for given features"""
    model, scaler = load_model()
    if model is None:
        return None
    
    # Create feature vector (same as in training)
    features = [med_income, house_age, avg_rooms, avg_bedrooms, 
               population, avg_occupancy, latitude, longitude]
    
    # Add engineered features
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
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    predicted_price = model.predict(features_scaled)[0]
    
    return predicted_price * 100000

def interactive_prediction():
    """Interactive house price prediction"""
    print("House Price Prediction Tool")
    print("=" * 40)
    
    print("\nEnter house details (press Enter for defaults):")
    
    med_income = float(input("Median income (default 3.87): ") or "3.87")
    house_age = float(input("House age in years (default 28): ") or "28")
    avg_rooms = float(input("Average rooms (default 5.43): ") or "5.43")
    avg_bedrooms = float(input("Average bedrooms (default 1.10): ") or "1.10")
    population = float(input("Population (default 1425): ") or "1425")
    avg_occupancy = float(input("Average occupancy (default 3.07): ") or "3.07")
    latitude = float(input("Latitude (default 34.26): ") or "34.26")
    longitude = float(input("Longitude (default -118.49): ") or "-118.49")
    
    predicted_price = predict_house_price(med_income, house_age, avg_rooms, avg_bedrooms,
                                        population, avg_occupancy, latitude, longitude)
    
    if predicted_price is not None:
        print(f"\nPredicted House Price: ${predicted_price:,.2f}")
        print(f"Price Range: ${predicted_price * 0.8:,.0f} - ${predicted_price * 1.2:,.0f}")

def quick_examples():
    """Show quick example predictions"""
    print("House Price Examples")
    print("=" * 25)
    
    examples = [
        ("Luxury Coastal Home", [8.0, 15, 7.0, 1.2, 1000, 2.5, 37.8, -122.4]),
        ("Affordable Inland Home", [2.5, 35, 4.5, 1.0, 2000, 3.5, 34.0, -118.0]),
        ("Family Suburban Home", [5.0, 25, 6.0, 1.1, 1500, 3.0, 35.5, -120.0])
    ]
    
    for name, features in examples:
        price = predict_house_price(*features)
        if price is not None:
            print(f"{name:<20}: ${price:,.0f}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Interactive prediction")
    print("2. Quick examples")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        interactive_prediction()
    elif choice == "2":
        quick_examples()
    else:
        print("Invalid choice. Running quick examples...")
        quick_examples()