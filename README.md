# California House Price Prediction

A machine learning system that predicts house prices in California based on location, number of rooms, population, and other key features.

## Features

- **High Accuracy**: 77.5% accuracy (R² = 0.775)
- **Fast Prediction**: Optimized Random Forest model
- **Easy to Use**: Simple command-line interface
- **Complete Analysis**: Data cleaning, training, and evaluation
- **Real-world Ready**: Production-ready model files

## Model Performance

- **Accuracy**: 77.5% (R² = 0.775)
- **Average Error**: ±$33,579
- **Model Type**: Random Forest Regressor
- **Dataset**: 19,464 California houses

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python house_price_predictor.py
```

### 3. Make Predictions
```bash
python predict.py
```

## Project Structure

```
house-price-prediction/
├── house_price_predictor.py    # Main training script
├── predict.py                  # Prediction tool
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── house_price_model.pkl       # Trained model (created after training)
└── scaler.pkl                  # Feature scaler (created after training)
```

## How to Use

### Training the Model
```bash
python house_price_predictor.py
```
This will:
- Load the California housing dataset
- Clean and preprocess the data
- Train a Random Forest model
- Show accuracy metrics and feature importance
- Save the trained model

### Making Predictions
```bash
python predict.py
```
Choose from:
1. **Interactive prediction** - Enter house details manually
2. **Quick examples** - See sample predictions

### Programmatic Usage
```python
from house_price_predictor import HousePricePredictor

# Initialize and train
predictor = HousePricePredictor()
predictor.load_data()
predictor.train_model()

# Make prediction
price = predictor.predict_price(
    med_income=5.0,      # Median income
    house_age=25,        # House age in years
    avg_rooms=6.0,       # Average rooms
    avg_bedrooms=1.1,    # Average bedrooms
    population=1500,     # Population
    avg_occupancy=3.0,   # Average occupancy
    latitude=35.5,       # Latitude
    longitude=-120.0     # Longitude
)
print(f"Predicted price: ${price:,.2f}")
```

## Key Features Analyzed

1. **Median Income** (47.6%) - Most important factor
2. **Average Occupancy** (15.6%) - Space per person
3. **Location** (19.2%) - Latitude and longitude
4. **House Age** (6.0%) - Age of the house
5. **Room Count** (5.2%) - Number of rooms

## Example Predictions

- **Luxury Coastal Home**: $450,223
- **Affordable Inland Home**: $171,485
- **Family Suburban Home**: $166,763

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib

## Technical Details

- **Data Source**: California Housing Dataset (scikit-learn)
- **Preprocessing**: Outlier removal, feature scaling
- **Algorithm**: Random Forest Regressor
- **Validation**: 80/20 train-test split
- **Cross-validation**: Built-in sklearn validation

## Model Insights

The model reveals that house prices in California are primarily driven by:
- **Income levels** in the area
- **Geographic location** (coastal vs inland)
- **Space per person** (occupancy rates)
- **House characteristics** (age, size)

This makes the model valuable for:
- Real estate market analysis
- Investment decision making
- Property valuation
- Market trend analysis

## Performance

The model achieves excellent performance with:
- **77.5% accuracy** - Explains most price variance
- **±$33,579 error** - Reasonable for real estate
- **Fast predictions** - Optimized for production use
- **Robust performance** - Works across different price ranges

## License

This project is open source and available under the MIT License.