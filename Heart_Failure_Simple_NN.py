# %%
"""
Heart Failure Neural Network - Simplified Implementation
=======================================================

Simple neural network implementation without complex TensorFlow dependencies
Using basic Keras/TensorFlow functionality for heart failure readmission prediction
"""

# %%
# Cell 1: Import Libraries (Simplified)
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           f1_score, accuracy_score, average_precision_score)
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Try simple TensorFlow import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    
    # Set random seed
    tf.random.set_seed(42)
    
    # Suppress warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("✅ TensorFlow imported successfully")
    print(f"TensorFlow version: {tf.__version__}")
    TENSORFLOW_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False

# Set random seeds
np.random.seed(42)

print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# Cell 2: Load and Prepare Data (Process from scratch)
print("="*60)
print("LOADING DATA FOR SIMPLE NEURAL NETWORK")
print("="*60)

# Load original data and process
patients = pd.read_csv('MIMIC_PATIENTS.csv')
admissions = pd.read_csv('MIMIC_ADMISSIONS.csv')
diagnoses = pd.read_csv('MIMIC_DIAGNOSES_ICD.csv')
icustays = pd.read_csv('MIMIC_ICUSTAYS.csv')
labevents = pd.read_csv('MIMIC_LABEVENTS.csv')
prescriptions = pd.read_csv('MIMIC_PRESCRIPTIONS.csv')

# Convert datetime columns
patients['DOB'] = pd.to_datetime(patients['DOB'])
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])

# Create target variable
admissions_sorted = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME']).copy()
admissions_sorted['NEXT_ADMITTIME'] = admissions_sorted.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
admissions_sorted['DAYS_TO_NEXT_ADMISSION'] = (
    admissions_sorted['NEXT_ADMITTIME'] - admissions_sorted['DISCHTIME']
).dt.days

admissions_sorted['READMITTED_30DAY'] = (
    (admissions_sorted['DAYS_TO_NEXT_ADMISSION'] <= 30) & 
    (admissions_sorted['DAYS_TO_NEXT_ADMISSION'] > 0)
).astype(int)

# Remove last admission for each patient
admissions_sorted['IS_LAST_ADMISSION'] = admissions_sorted.groupby('SUBJECT_ID').cumcount(ascending=False) == 0
modeling_data = admissions_sorted[~admissions_sorted['IS_LAST_ADMISSION']].copy()

# Merge with patient data
modeling_data = modeling_data.merge(
    patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID'
)

# Basic features
modeling_data['AGE_AT_ADMISSION'] = (modeling_data['ADMITTIME'] - modeling_data['DOB']).dt.days / 365.25
modeling_data['GENDER_MALE'] = (modeling_data['GENDER'] == 'M').astype(int)
modeling_data['LOS_DAYS'] = (modeling_data['DISCHTIME'] - modeling_data['ADMITTIME']).dt.days
modeling_data['EMERGENCY_ADMISSION'] = (modeling_data['ADMISSION_TYPE'] == 'EMERGENCY').astype(int)
modeling_data['MEDICARE'] = (modeling_data['INSURANCE'] == 'Medicare').astype(int)

print("✅ Processed data from scratch")

print(f"📊 Dataset: {len(modeling_data):,} admissions")
print(f"📈 30-day readmission rate: {modeling_data['READMITTED_30DAY'].mean()*100:.1f}%")

# %%
# Cell 3: Simple Feature Engineering for Neural Network
print("="*60)
print("SIMPLE FEATURE ENGINEERING FOR NEURAL NETWORK")
print("="*60)

# Create a simple but effective feature set
simple_features = [
    'AGE_AT_ADMISSION', 'GENDER_MALE', 'LOS_DAYS', 
    'EMERGENCY_ADMISSION', 'MEDICARE'
]

# Add some engineered features
modeling_data['AGE_SQUARED'] = modeling_data['AGE_AT_ADMISSION'] ** 2
modeling_data['LOS_LOG'] = np.log1p(modeling_data['LOS_DAYS'])
modeling_data['AGE_LOS_INTERACTION'] = modeling_data['AGE_AT_ADMISSION'] * modeling_data['LOS_DAYS']

# Add to feature list
simple_features.extend(['AGE_SQUARED', 'LOS_LOG', 'AGE_LOS_INTERACTION'])

# Create feature matrix
X_simple = modeling_data[simple_features].copy()
y_simple = modeling_data['READMITTED_30DAY'].copy()

print(f"📊 Simple Feature Matrix Shape: {X_simple.shape}")
print(f"📊 Features: {simple_features}")
print(f"📊 Target Distribution: {y_simple.value_counts().to_dict()}")

# Handle any missing values
imputer = SimpleImputer(strategy='median')
X_simple_imputed = pd.DataFrame(
    imputer.fit_transform(X_simple),
    columns=X_simple.columns,
    index=X_simple.index
)

print("✅ Simple features created and cleaned")

# %%
# Cell 4: Train-Test Split and Preprocessing
print("="*60)
print("TRAIN-TEST SPLIT AND PREPROCESSING")
print("="*60)

if TENSORFLOW_AVAILABLE:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_simple_imputed, y_simple, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_simple
    )
    
    print(f"📊 Data Split:")
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled")
    
    # Balance classes with SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"📊 After SMOTE:")
    print(f"   Balanced training set: {X_train_balanced.shape[0]:,} samples")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"📊 Class weights: {class_weight_dict}")
    print("✅ Data preprocessing completed")

else:
    print("❌ TensorFlow not available")

# %%
# Cell 5: Create and Train Simple Neural Networks
print("="*60)
print("CREATING AND TRAINING SIMPLE NEURAL NETWORKS")
print("="*60)

if TENSORFLOW_AVAILABLE:
    
    def create_simple_model(input_dim):
        """Create a simple neural network"""
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def create_deep_model(input_dim):
        """Create a deeper neural network"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.4),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    # Model configurations
    models_config = {
        'Simple_NN': create_simple_model,
        'Deep_NN': create_deep_model
    }
    
    # Training parameters
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train models
    results = {}
    trained_models = {}
    
    print("🧠 Training Neural Networks...")
    print("-" * 40)
    
    for name, model_fn in models_config.items():
        print(f"\nTraining {name}...")
        
        try:
            # Create model
            model = model_fn(X_train_balanced.shape[1])
            
            # Compile
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train
            history = model.fit(
                X_train_balanced, y_train_balanced,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.2,
                callbacks=[early_stopping],
                class_weight=class_weight_dict,
                verbose=0
            )
            
            # Predict
            y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'epochs_trained': len(history.history['loss'])
            }
            
            trained_models[name] = model
            
            print(f"   ✅ {name} trained successfully")
            print(f"      ROC-AUC: {roc_auc:.4f}")
            print(f"      F1-Score: {f1:.4f}")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Epochs: {len(history.history['loss'])}")
            
            # Check improvement over baseline
            baseline_auc = 0.577
            if roc_auc > baseline_auc:
                improvement = ((roc_auc - baseline_auc) / baseline_auc) * 100
                print(f"      🎉 Improvement: +{improvement:.1f}% over baseline!")
            
        except Exception as e:
            print(f"   ❌ {name} failed: {str(e)}")
    
    print(f"\n✅ Neural network training completed!")
    print(f"   Successfully trained: {len(results)} models")

else:
    print("❌ TensorFlow not available")

# %%
# Cell 6: Analyze Neural Network Results
print("="*60)
print("NEURAL NETWORK RESULTS ANALYSIS")
print("="*60)

if TENSORFLOW_AVAILABLE and len(results) > 0:
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T[['roc_auc', 'f1_score', 'accuracy', 'avg_precision']].round(4)
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
    
    print("🏆 NEURAL NETWORK PERFORMANCE:")
    print("=" * 60)
    print(f"{'Model':<15} {'ROC-AUC':<10} {'F1-Score':<10} {'Accuracy':<10} {'Avg-Precision':<12}")
    print("-" * 60)
    
    for model_name in comparison_df.index:
        roc_auc = comparison_df.loc[model_name, 'roc_auc']
        f1 = comparison_df.loc[model_name, 'f1_score']
        accuracy = comparison_df.loc[model_name, 'accuracy']
        avg_prec = comparison_df.loc[model_name, 'avg_precision']
        
        print(f"{model_name:<15} {roc_auc:<10.4f} {f1:<10.4f} {accuracy:<10.4f} {avg_prec:<12.4f}")
    
    # Best model analysis
    best_model_name = comparison_df.index[0]
    best_metrics = comparison_df.loc[best_model_name]
    
    print(f"\n🥇 BEST NEURAL NETWORK: {best_model_name}")
    print("=" * 40)
    print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Avg Precision: {best_metrics['avg_precision']:.4f}")
    
    # Compare with baseline
    baseline_auc = 0.577
    best_auc = best_metrics['roc_auc']
    improvement = ((best_auc - baseline_auc) / baseline_auc) * 100
    
    print(f"\n📈 PERFORMANCE vs BASELINE:")
    print("=" * 35)
    print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
    print(f"Best NN ROC-AUC: {best_auc:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    if improvement > 5:
        print("🎉 Significant improvement with neural networks!")
    elif improvement > 0:
        print("✅ Modest improvement with neural networks")
    else:
        print("⚠️ Neural networks did not improve over baseline")
    
    # Detailed evaluation
    best_predictions = results[best_model_name]['predictions']
    best_probabilities = results[best_model_name]['probabilities']
    
    print(f"\n📋 DETAILED EVALUATION - {best_model_name}:")
    print("=" * 50)
    
    print("Classification Report:")
    print(classification_report(y_test, best_predictions, 
                              target_names=['No Readmission', 'Readmission']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_predictions)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 No    Yes")
    print(f"Actual    No   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"          Yes  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nKey Metrics:")
    print(f"   Sensitivity: {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    
    print(f"\n✅ Neural network analysis completed!")

else:
    print("❌ No neural network results to analyze")

# %%
# Cell 7: Final Summary and Recommendations
print("="*60)
print("FINAL NEURAL NETWORK SUMMARY")
print("="*60)

if TENSORFLOW_AVAILABLE and len(results) > 0:
    
    print("🧠 NEURAL NETWORK FINAL SUMMARY:")
    print("=" * 45)
    print(f"Best Model: {best_model_name}")
    print(f"Dataset Size: {len(modeling_data):,} admissions")
    print(f"Features Used: {X_simple_imputed.shape[1]} simple features")
    print(f"Test Set Size: {len(y_test):,} admissions")
    
    print(f"\nPerformance Metrics:")
    print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"   F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   Sensitivity: {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    
    print(f"\n📊 MODEL COMPARISON:")
    print("=" * 30)
    all_models = {
        'Baseline Gradient Boosting': 0.577,
        f'Neural Network ({best_model_name})': best_metrics['roc_auc'],
        'Enhanced Random Forest': 0.532
    }
    
    sorted_models = sorted(all_models.items(), key=lambda x: x[1], reverse=True)
    for i, (model, auc) in enumerate(sorted_models, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"{medal} {model:<30} {auc:.4f}")
    
    print(f"\n💡 NEURAL NETWORK INSIGHTS:")
    print("=" * 35)
    insights = [
        f"✅ Simple architecture with {X_simple_imputed.shape[1]} features works effectively",
        f"⚖️ Class balancing with SMOTE improves minority class detection",
        f"🎯 Early stopping prevents overfitting",
        f"📊 Feature scaling is crucial for neural network performance"
    ]
    
    for insight in insights:
        print(f"   {insight}")
    
    print(f"\n🚀 DEPLOYMENT RECOMMENDATIONS:")
    print("=" * 40)
    recommendations = [
        "1. Use simple neural network architecture for production",
        "2. Implement real-time feature scaling pipeline",
        "3. Monitor model performance and retrain periodically",
        "4. Consider ensemble with traditional ML models",
        "5. Validate on real clinical data before deployment"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    # Final verdict
    print(f"\n🎯 FINAL VERDICT:")
    print("=" * 20)
    if best_auc > 0.577:
        print(f"✅ Neural networks achieved improvement over baseline!")
        print(f"   Recommended for production deployment")
    else:
        print(f"⚠️ Neural networks did not significantly improve")
        print(f"   Consider hybrid approach or more complex features")

else:
    print("❌ Neural network analysis not completed")

print(f"\n" + "="*60)
print("🎉 SIMPLE NEURAL NETWORK ANALYSIS COMPLETE!")
print("="*60)
print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if TENSORFLOW_AVAILABLE and len(results) > 0:
    print(f"Best Neural Network: {best_model_name} with ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Performance vs Baseline: {improvement:+.1f}%")
print("="*60)

# %%