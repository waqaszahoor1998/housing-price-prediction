"""
Enhanced Data Visualization for California House Price Prediction
=================================================================

Creates clear, professional visualizations with improved styling and insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

print("Creating Enhanced Visualizations...")
print("=" * 60)

# Load data
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['Price'] = housing.target * 100000  # Convert to actual dollars

# Feature engineering (matching the model)
data['rooms_per_household'] = data['AveRooms'] / data['AveOccup']
data['bedrooms_per_room'] = data['AveBedrms'] / data['AveRooms']
data['income_per_person'] = data['MedInc'] / data['AveOccup']

print(f"Loaded {len(data):,} records with {len(data.columns)} features")

# ============================================================================
# 1. CORRELATION HEATMAP - Clear Version
# ============================================================================
print("\n1. Creating Correlation Heatmap...")

fig, ax = plt.subplots(figsize=(14, 12))

# Select main features for correlation
corr_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                 'Population', 'AveOccup', 'Latitude', 'Longitude', 'Price']
corr_matrix = data[corr_features].corr()

# Create heatmap with better styling
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show lower triangle only
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1.5, cbar_kws={"shrink": 0.8},
            ax=ax, annot_kws={'size': 9, 'weight': 'bold'})

ax.set_title('Feature Correlation Matrix\n(California Housing Dataset)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('correlation_matrix_enhanced.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: correlation_matrix_enhanced.png")
plt.close()

# ============================================================================
# 2. DISTRIBUTION ANALYSIS - Enhanced
# ============================================================================
print("\n2. Creating Distribution Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

features_to_plot = ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup', 'Price']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

for idx, (feature, color) in enumerate(zip(features_to_plot, colors)):
    ax = axes[idx]
    
    # Histogram
    ax.hist(data[feature], bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add statistics
    mean_val = data[feature].mean()
    median_val = data[feature].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    ax.set_title(f'{feature} Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('data_distributions_enhanced.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: data_distributions_enhanced.png")
plt.close()

# ============================================================================
# 3. FEATURE VS PRICE RELATIONSHIPS - Clear Scatter Plots
# ============================================================================
print("\n3. Creating Feature-Price Relationships...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

key_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 
                'income_per_person', 'rooms_per_household']
colors_scatter = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#E67E22']

for idx, (feature, color) in enumerate(zip(key_features, colors_scatter)):
    ax = axes[idx]
    
    # Create scatter plot with transparency
    ax.scatter(data[feature], data['Price'], alpha=0.4, s=20, color=color, edgecolors='black', linewidths=0.3)
    
    # Add trend line
    z = np.polyfit(data[feature], data['Price'], 1)
    p = np.poly1d(z)
    ax.plot(data[feature], p(data[feature]), "r--", linewidth=2, label='Trend')
    
    # Calculate correlation
    corr = data[feature].corr(data['Price'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')
    
    ax.set_xlabel(feature, fontsize=11, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax.set_title(f'{feature} vs Price', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Feature Impact on House Prices', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('feature_price_relationships_enhanced.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: feature_price_relationships_enhanced.png")
plt.close()

# ============================================================================
# 4. FEATURE IMPORTANCE - Vertical Bar Chart
# ============================================================================
print("\n4. Creating Feature Importance Chart...")

# Train a quick model for feature importance
X = data[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
          'AveOccup', 'Latitude', 'Longitude', 'rooms_per_household', 
          'income_per_person']]
y = data['Price']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)

# Get feature importance
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                     'AveOccup', 'Latitude', 'Longitude', 'Rooms/Household', 
                     'Income/Person']
importances = model.feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1]

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#E74C3C'] + ['steelblue'] * (len(indices) - 1)  # Highlight most important
bars = ax.barh(range(len(indices)), importances[indices], color=colors)

# Add value labels
for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
    ax.text(imp + 0.01, i, f'{imp:.1%}', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[idx] for idx in indices])
ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance - What Drives House Prices?', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: feature_importance_enhanced.png")
plt.close()

# ============================================================================
# 5. MODEL EVALUATION - Predictions vs Actual
# ============================================================================
print("\n5. Creating Model Evaluation Chart...")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_train = RobustScaler()
X_train_scaled = scaler_train.fit_transform(X_train)
X_test_scaled = scaler_train.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Create subplots
fig = plt.figure(figsize=(16, 6))

# Subplot 1: Scatter plot - Predictions vs Actual
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(y_test, y_pred, alpha=0.6, s=30, color='#3498DB', edgecolors='black', linewidth=0.3)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, label='Perfect Prediction')

ax1.set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
ax1.set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Add metrics text
metrics_text = f'R² Score: {r2:.3f}\nRMSE: ${rmse:,.0f}\nMAE: ${mae:,.0f}'
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
         fontsize=11, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Subplot 2: Residual plot
ax2 = plt.subplot(1, 2, 2)
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6, s=30, color='#E67E22', edgecolors='black', linewidth=0.3)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)

ax2.set_xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
ax2.set_title('Residual Analysis', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.suptitle('Model Performance Evaluation', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_evaluation_enhanced.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: model_evaluation_enhanced.png")
plt.close()

# ============================================================================
# 6. PRICE DISTRIBUTION BY CATEGORIES
# ============================================================================
print("\n6. Creating Price Distribution by Income Levels...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Income levels
data['Income_Level'] = pd.cut(data['MedInc'], 
                               bins=[0, 2, 4, 6, 10, 100],
                               labels=['Very Low\n(<$200k)', 'Low\n($200-400k)', 
                                       'Medium\n($400-600k)', 'High\n($600k-1M)', 
                                       'Very High\n(>$1M)'])

# Box plot by income level
ax1 = axes[0, 0]
data.boxplot(column='Price', by='Income_Level', ax=ax1, 
             grid=True, patch_artist=True,
             boxprops=dict(facecolor='lightblue', color='black'),
             medianprops=dict(color='red', linewidth=2))
ax1.set_title('Price Distribution by Income Level', fontsize=13, fontweight='bold')
ax1.set_xlabel('Income Level', fontsize=11)
ax1.set_ylabel('Price ($)', fontsize=11)

# Age groups
data['Age_Group'] = pd.cut(data['HouseAge'], 
                           bins=[0, 15, 30, 45, 60, 100],
                           labels=['Very New\n(<15)', 'New\n(15-30)', 
                                   'Middle\n(30-45)', 'Old\n(45-60)', 
                                   'Very Old\n(60+)'])

ax2 = axes[0, 1]
data.boxplot(column='Price', by='Age_Group', ax=ax2, 
             grid=True, patch_artist=True,
             boxprops=dict(facecolor='lightcoral', color='black'),
             medianprops=dict(color='blue', linewidth=2))
ax2.set_title('Price Distribution by House Age', fontsize=13, fontweight='bold')
ax2.set_xlabel('Age Group', fontsize=11)
ax2.set_ylabel('Price ($)', fontsize=11)

# Geographic regions (by latitude)
data['Region'] = pd.cut(data['Latitude'], 
                        bins=[32, 34, 36, 38],
                        labels=['Southern\nCA', 'Central\nCA', 'Northern\nCA'])

ax3 = axes[1, 0]
data.boxplot(column='Price', by='Region', ax=ax3, 
             grid=True, patch_artist=True,
             boxprops=dict(facecolor='lightgreen', color='black'),
             medianprops=dict(color='purple', linewidth=2))
ax3.set_title('Price Distribution by Region', fontsize=13, fontweight='bold')
ax3.set_xlabel('Region', fontsize=11)
ax3.set_ylabel('Price ($)', fontsize=11)

# Statistics summary
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
DATA SUMMARY
{'='*40}

Total Records: {len(data):,}
Average Price: ${data['Price'].mean():,.2f}
Median Price: ${data['Price'].median():,.2f}
Min Price: ${data['Price'].min():,.2f}
Max Price: ${data['Price'].max():,.2f}

Standard Deviation: ${data['Price'].std():,.2f}

KEY INSIGHTS:
• Median Income is the strongest
  predictor of house prices
• Newer houses tend to be more
  expensive
• Geographic location significantly
  impacts pricing
• Room count and occupancy
  patterns affect value

MODEL PERFORMANCE:
R² Score: {r2:.3f}
RMSE: ${rmse:,.0f}
MAE: ${mae:,.0f}
"""
ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Comprehensive Price Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: comprehensive_analysis.png")
plt.close()

print("\n" + "=" * 60)
print("[SUCCESS] All visualizations created successfully!")
print("=" * 60)
print("\nGenerated Files:")
print("  1. correlation_matrix_enhanced.png")
print("  2. data_distributions_enhanced.png")
print("  3. feature_price_relationships_enhanced.png")
print("  4. feature_importance_enhanced.png")
print("  5. model_evaluation_enhanced.png")
print("  6. comprehensive_analysis.png")

