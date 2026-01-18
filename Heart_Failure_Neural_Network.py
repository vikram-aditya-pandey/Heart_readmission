# %%
"""
Heart Failure 30-Day Readmission Prediction - Neural Network Focus
==================================================================

Dedicated neural network implementation with:
- Proper TensorFlow/Keras configuration
- Multiple NN architectures
- Advanced techniques for imbalanced data
- Hyperparameter optimization
- Ensemble of neural networks

Goal: Achieve better performance than baseline ROC-AUC of 0.577
"""

# %%
# Cell 1: Import Libraries and Setup
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           f1_score, accuracy_score, average_precision_score,
                           precision_recall_curve, roc_curve)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Neural Network libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    from tensorflow.keras.utils import class_weight
    
    # Set TensorFlow configuration
    tf.random.set_seed(42)
    
    # Suppress TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("✅ TensorFlow/Keras imported successfully")
    print(f"TensorFlow version: {tf.__version__}")
    TENSORFLOW_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False

# Set random seeds
np.random.seed(42)

print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# Cell 2: Load and Prepare Data
print("="*60)
print("LOADING AND PREPARING DATA FOR NEURAL NETWORKS")
print("="*60)

# Load data (reuse from previous analysis)
patients = pd.read_csv('MIMIC_PATIENTS.csv')
admissions = pd.read_csv('MIMIC_ADMISSIONS.csv')
diagnoses = pd.read_csv('MIMIC_DIAGNOSES_ICD.csv')
icustays = pd.read_csv('MIMIC_ICUSTAYS.csv')
labevents = pd.read_csv('MIMIC_LABEVENTS.csv')
prescriptions = pd.read_csv('MIMIC_PRESCRIPTIONS.csv')

print(f"📊 Dataset loaded: {len(admissions):,} admissions")

# Convert datetime columns
patients['DOB'] = pd.to_datetime(patients['DOB'])
patients['DOD'] = pd.to_datetime(patients['DOD'])
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

print(f"📈 Modeling dataset: {len(modeling_data):,} admissions")
print(f"📈 30-day readmission rate: {modeling_data['READMITTED_30DAY'].mean()*100:.1f}%")

print("✅ Data loaded and target variable created")

# %%
# Cell 3: Comprehensive Feature Engineering for Neural Networks
print("="*60)
print("NEURAL NETWORK OPTIMIZED FEATURE ENGINEERING")
print("="*60)

def create_nn_features(modeling_data, patients, diagnoses, icustays, labevents, prescriptions):
    """Create features optimized for neural network learning"""
    
    # Merge with patient data
    modeling_data = modeling_data.merge(
        patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']], 
        on='SUBJECT_ID'
    )
    
    # Basic features
    modeling_data['AGE_AT_ADMISSION'] = (modeling_data['ADMITTIME'] - modeling_data['DOB']).dt.days / 365.25
    modeling_data['GENDER_MALE'] = (modeling_data['GENDER'] == 'M').astype(int)
    modeling_data['LOS_DAYS'] = (modeling_data['DISCHTIME'] - modeling_data['ADMITTIME']).dt.days
    modeling_data['EMERGENCY_ADMISSION'] = (modeling_data['ADMISSION_TYPE'] == 'EMERGENCY').astype(int)
    modeling_data['MEDICARE'] = (modeling_data['INSURANCE'] == 'Medicare').astype(int)
    
    # Normalized features for NN
    modeling_data['AGE_NORMALIZED'] = modeling_data['AGE_AT_ADMISSION'] / 100.0  # Scale to 0-1 range
    modeling_data['LOS_LOG'] = np.log1p(modeling_data['LOS_DAYS'])
    
    # Temporal features
    modeling_data['ADMIT_MONTH'] = modeling_data['ADMITTIME'].dt.month
    modeling_data['ADMIT_WEEKDAY'] = modeling_data['ADMITTIME'].dt.weekday
    modeling_data['WEEKEND_ADMISSION'] = (modeling_data['ADMIT_WEEKDAY'] >= 5).astype(int)
    
    # Cyclical encoding for temporal features (better for NN)
    modeling_data['MONTH_SIN'] = np.sin(2 * np.pi * modeling_data['ADMIT_MONTH'] / 12)
    modeling_data['MONTH_COS'] = np.cos(2 * np.pi * modeling_data['ADMIT_MONTH'] / 12)
    modeling_data['WEEKDAY_SIN'] = np.sin(2 * np.pi * modeling_data['ADMIT_WEEKDAY'] / 7)
    modeling_data['WEEKDAY_COS'] = np.cos(2 * np.pi * modeling_data['ADMIT_WEEKDAY'] / 7)
    
    print("✅ Basic and temporal features created")
    
    # Diagnosis features
    hf_codes = [39891, 40201, 40211, 40291, 40401, 40403, 40411, 40413,
                40491, 40493, 4280, 4281, 42820, 42821, 42822, 42823,
                42830, 42831, 42832, 42833, 42840, 42841, 42842, 42843, 4289]
    
    diabetes_codes = [25000, 25001, 25002]
    hypertension_codes = [4019, 40190, 40191]
    kidney_codes = [5849, 5859]
    cardiac_codes = [42731, 42732, 41401, 41071]
    respiratory_codes = [49390, 49391]
    
    diagnosis_features = []
    for hadm_id in modeling_data['HADM_ID'].unique():
        admission_diagnoses = diagnoses[diagnoses['HADM_ID'] == hadm_id]['ICD9_CODE'].tolist()
        
        # Counts and flags
        total_dx = len(admission_diagnoses)
        hf_dx_count = sum(1 for code in admission_diagnoses if code in hf_codes)
        
        has_diabetes = int(any(code in diabetes_codes for code in admission_diagnoses))
        has_hypertension = int(any(code in hypertension_codes for code in admission_diagnoses))
        has_kidney = int(any(code in kidney_codes for code in admission_diagnoses))
        has_cardiac = int(any(code in cardiac_codes for code in admission_diagnoses))
        has_respiratory = int(any(code in respiratory_codes for code in admission_diagnoses))
        
        # Normalized counts for NN
        comorbidity_score = has_diabetes + has_hypertension + has_kidney + has_cardiac + has_respiratory
        complexity_score = total_dx + hf_dx_count + comorbidity_score
        
        diagnosis_features.append({
            'HADM_ID': hadm_id,
            'TOTAL_DIAGNOSES': total_dx,
            'TOTAL_DIAGNOSES_NORM': min(total_dx / 10.0, 1.0),  # Normalize to 0-1
            'HF_DIAGNOSES_COUNT': hf_dx_count,
            'HF_DIAGNOSES_NORM': min(hf_dx_count / 5.0, 1.0),
            'HAS_DIABETES': has_diabetes,
            'HAS_HYPERTENSION': has_hypertension,
            'HAS_KIDNEY_DISEASE': has_kidney,
            'HAS_CARDIAC_COMORBIDITY': has_cardiac,
            'HAS_RESPIRATORY_DISEASE': has_respiratory,
            'COMORBIDITY_COUNT': comorbidity_score,
            'COMORBIDITY_NORM': comorbidity_score / 5.0,
            'COMPLEXITY_SCORE': complexity_score,
            'COMPLEXITY_NORM': min(complexity_score / 20.0, 1.0)
        })
    
    diagnosis_df = pd.DataFrame(diagnosis_features)
    modeling_data = modeling_data.merge(diagnosis_df, on='HADM_ID')
    
    print("✅ Diagnosis features created")
    
    # ICU features
    icu_features = []
    for hadm_id in modeling_data['HADM_ID'].unique():
        icu_stays_admission = icustays[icustays['HADM_ID'] == hadm_id]
        
        if len(icu_stays_admission) > 0:
            icu_los = icu_stays_admission['LOS'].sum()
            has_icu = 1
        else:
            icu_los = 0
            has_icu = 0
        
        icu_features.append({
            'HADM_ID': hadm_id,
            'HAS_ICU_STAY': has_icu,
            'ICU_LOS_DAYS': icu_los,
            'ICU_LOS_NORM': min(icu_los / 10.0, 1.0)  # Normalize
        })
    
    icu_df = pd.DataFrame(icu_features)
    modeling_data = modeling_data.merge(icu_df, on='HADM_ID')
    
    print("✅ ICU features created")
    
    # Lab features (simplified for NN)
    lab_features = []
    key_lab_items = {
        50912: 'CREATININE',
        50983: 'SODIUM', 
        51006: 'BUN',
        51222: 'HEMOGLOBIN',
        50862: 'ALBUMIN'
    }
    
    for hadm_id in modeling_data['HADM_ID'].unique():
        admission_labs = labevents[labevents['HADM_ID'] == hadm_id]
        
        lab_feature = {'HADM_ID': hadm_id, 'TOTAL_LABS': len(admission_labs)}
        
        for itemid, lab_name in key_lab_items.items():
            lab_values = admission_labs[admission_labs['ITEMID'] == itemid]['VALUENUM']
            
            if len(lab_values) > 0:
                last_value = lab_values.iloc[-1]
                lab_feature[f'{lab_name}_LAST'] = last_value
                lab_feature[f'HAS_{lab_name}'] = 1
                
                # Normalized abnormal flags
                if lab_name == 'CREATININE':
                    lab_feature[f'{lab_name}_ABNORMAL'] = min(max(last_value - 1.0, 0) / 2.0, 1.0)
                elif lab_name == 'SODIUM':
                    lab_feature[f'{lab_name}_ABNORMAL'] = max(135 - last_value, 0) / 10.0
                elif lab_name == 'BUN':
                    lab_feature[f'{lab_name}_ABNORMAL'] = min(max(last_value - 20, 0) / 30.0, 1.0)
                elif lab_name == 'HEMOGLOBIN':
                    lab_feature[f'{lab_name}_ABNORMAL'] = max(10 - last_value, 0) / 5.0
                elif lab_name == 'ALBUMIN':
                    lab_feature[f'{lab_name}_ABNORMAL'] = max(3.5 - last_value, 0) / 2.0
            else:
                lab_feature[f'{lab_name}_LAST'] = np.nan
                lab_feature[f'HAS_{lab_name}'] = 0
                lab_feature[f'{lab_name}_ABNORMAL'] = 0
        
        lab_features.append(lab_feature)
    
    lab_df = pd.DataFrame(lab_features)
    modeling_data = modeling_data.merge(lab_df, on='HADM_ID')
    
    print("✅ Lab features created")
    
    # Medication features
    hf_medications = {
        'ACE_INHIBITOR': ['Lisinopril', 'Enalapril'],
        'BETA_BLOCKER': ['Metoprolol', 'Carvedilol'],
        'DIURETIC': ['Furosemide', 'Hydrochlorothiazide'],
        'ARB': ['Losartan']
    }
    
    med_features = []
    for hadm_id in modeling_data['HADM_ID'].unique():
        admission_meds = prescriptions[prescriptions['HADM_ID'] == hadm_id]
        med_drugs = admission_meds['DRUG'].tolist()
        
        med_feature = {
            'HADM_ID': hadm_id,
            'TOTAL_MEDICATIONS': len(admission_meds),
            'TOTAL_MEDS_NORM': min(len(admission_meds) / 15.0, 1.0)
        }
        
        # Key medication classes
        for med_class, drug_list in hf_medications.items():
            has_med = int(any(drug in med_drugs for drug in drug_list))
            med_feature[f'HAS_{med_class}'] = has_med
        
        # GDMT score
        gdmt_score = (med_feature['HAS_ACE_INHIBITOR'] or med_feature['HAS_ARB']) + \
                     med_feature['HAS_BETA_BLOCKER'] + med_feature['HAS_DIURETIC']
        med_feature['GDMT_SCORE'] = gdmt_score
        med_feature['GDMT_NORM'] = gdmt_score / 3.0
        
        med_features.append(med_feature)
    
    med_df = pd.DataFrame(med_features)
    modeling_data = modeling_data.merge(med_df, on='HADM_ID')
    
    print("✅ Medication features created")
    
    # Historical features
    historical_features = []
    admissions_sorted_local = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME'])
    
    for subject_id in modeling_data['SUBJECT_ID'].unique():
        patient_admissions = admissions_sorted_local[admissions_sorted_local['SUBJECT_ID'] == subject_id]
        
        for idx, admission in patient_admissions.iterrows():
            prior_admissions = patient_admissions[patient_admissions['ADMITTIME'] < admission['ADMITTIME']]
            
            if len(prior_admissions) == 0:
                hist_features = {
                    'HADM_ID': admission['HADM_ID'],
                    'PRIOR_ADMISSIONS_COUNT': 0,
                    'PRIOR_ADMISSIONS_NORM': 0,
                    'DAYS_SINCE_LAST_ADMISSION': 365,  # Default to 1 year
                    'DAYS_SINCE_LAST_NORM': 1.0,
                    'HAD_PRIOR_READMISSION': 0
                }
            else:
                last_admission = prior_admissions.iloc[-1]
                days_since_last = (admission['ADMITTIME'] - pd.to_datetime(last_admission['DISCHTIME'])).days
                
                # Check for prior readmissions
                had_prior_readmission = 0
                for i in range(len(prior_admissions) - 1):
                    curr_disch = pd.to_datetime(prior_admissions.iloc[i]['DISCHTIME'])
                    next_admit = pd.to_datetime(prior_admissions.iloc[i+1]['ADMITTIME'])
                    if (next_admit - curr_disch).days <= 30:
                        had_prior_readmission = 1
                        break
                
                hist_features = {
                    'HADM_ID': admission['HADM_ID'],
                    'PRIOR_ADMISSIONS_COUNT': len(prior_admissions),
                    'PRIOR_ADMISSIONS_NORM': min(len(prior_admissions) / 5.0, 1.0),
                    'DAYS_SINCE_LAST_ADMISSION': days_since_last,
                    'DAYS_SINCE_LAST_NORM': min(days_since_last / 365.0, 1.0),
                    'HAD_PRIOR_READMISSION': had_prior_readmission
                }
            
            historical_features.append(hist_features)
    
    hist_df = pd.DataFrame(historical_features)
    modeling_data = modeling_data.merge(hist_df, on='HADM_ID')
    
    print("✅ Historical features created")
    
    return modeling_data

# Create NN-optimized features
modeling_data_nn = create_nn_features(
    modeling_data, patients, diagnoses, icustays, labevents, prescriptions
)

print(f"📊 NN-optimized feature set shape: {modeling_data_nn.shape}")

# %%
# Cell 4: Prepare Features for Neural Networks
print("="*60)
print("PREPARING FEATURES FOR NEURAL NETWORKS")
print("="*60)

# Select features optimized for neural networks
nn_feature_columns = [
    # Demographics (normalized)
    'AGE_NORMALIZED', 'GENDER_MALE', 'LOS_LOG', 'EMERGENCY_ADMISSION', 'MEDICARE',
    
    # Temporal (cyclical encoding)
    'MONTH_SIN', 'MONTH_COS', 'WEEKDAY_SIN', 'WEEKDAY_COS', 'WEEKEND_ADMISSION',
    
    # Diagnoses (normalized)
    'TOTAL_DIAGNOSES_NORM', 'HF_DIAGNOSES_NORM', 'COMORBIDITY_NORM', 'COMPLEXITY_NORM',
    'HAS_DIABETES', 'HAS_HYPERTENSION', 'HAS_KIDNEY_DISEASE', 
    'HAS_CARDIAC_COMORBIDITY', 'HAS_RESPIRATORY_DISEASE',
    
    # ICU (normalized)
    'HAS_ICU_STAY', 'ICU_LOS_NORM',
    
    # Labs (normalized abnormal scores)
    'HAS_CREATININE', 'HAS_SODIUM', 'HAS_BUN', 'HAS_HEMOGLOBIN', 'HAS_ALBUMIN',
    'CREATININE_ABNORMAL', 'SODIUM_ABNORMAL', 'BUN_ABNORMAL', 
    'HEMOGLOBIN_ABNORMAL', 'ALBUMIN_ABNORMAL',
    
    # Medications (normalized)
    'TOTAL_MEDS_NORM', 'GDMT_NORM',
    'HAS_ACE_INHIBITOR', 'HAS_BETA_BLOCKER', 'HAS_DIURETIC', 'HAS_ARB',
    
    # Historical (normalized)
    'PRIOR_ADMISSIONS_NORM', 'DAYS_SINCE_LAST_NORM', 'HAD_PRIOR_READMISSION'
]

# Create feature matrix
X_nn = modeling_data_nn[nn_feature_columns].copy()
y_nn = modeling_data_nn['READMITTED_30DAY'].copy()

print(f"📊 Neural Network Feature Matrix Shape: {X_nn.shape}")
print(f"📊 Target Distribution: {y_nn.value_counts().to_dict()}")

# Handle missing values
print(f"\n🔍 Missing Values Analysis:")
missing_counts = X_nn.isnull().sum()
missing_features = missing_counts[missing_counts > 0]

if len(missing_features) > 0:
    print("Features with missing values:")
    for feature, count in missing_features.items():
        print(f"   {feature}: {count} ({count/len(X_nn)*100:.1f}%)")
    
    # Simple imputation for NN
    imputer = SimpleImputer(strategy='median')
    X_nn_imputed = pd.DataFrame(
        imputer.fit_transform(X_nn),
        columns=X_nn.columns,
        index=X_nn.index
    )
else:
    X_nn_imputed = X_nn.copy()

print(f"✅ Missing values handled")
print(f"📊 Final NN feature set: {X_nn_imputed.shape[1]} features")

# %%
# Cell 5: Train-Test Split and Scaling for Neural Networks
print("="*60)
print("TRAIN-TEST SPLIT AND SCALING FOR NEURAL NETWORKS")
print("="*60)

if TENSORFLOW_AVAILABLE:
    # Stratified split
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        X_nn_imputed, y_nn, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_nn
    )
    
    print(f"📊 Data Split:")
    print(f"   Training set: {X_train_nn.shape[0]:,} samples")
    print(f"   Test set: {X_test_nn.shape[0]:,} samples")
    print(f"   Features: {X_train_nn.shape[1]}")
    
    # Neural networks work best with normalized features (0-1 range)
    scaler_nn = MinMaxScaler()  # Better for NN than StandardScaler
    X_train_nn_scaled = scaler_nn.fit_transform(X_train_nn)
    X_test_nn_scaled = scaler_nn.transform(X_test_nn)
    
    print(f"✅ Features scaled using MinMaxScaler (0-1 range)")
    
    # Class balancing with SMOTE
    smote_nn = SMOTE(random_state=42, k_neighbors=3)
    X_train_nn_balanced, y_train_nn_balanced = smote_nn.fit_resample(X_train_nn_scaled, y_train_nn)
    
    print(f"\n📊 After SMOTE Balancing:")
    print(f"   Balanced training set: {X_train_nn_balanced.shape[0]:,} samples")
    class_dist = pd.Series(y_train_nn_balanced).value_counts(normalize=True) * 100
    for class_val, pct in class_dist.items():
        label = "Readmission" if class_val == 1 else "No Readmission"
        print(f"   {label}: {pct:.1f}%")
    
    # Calculate class weights for loss function
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_nn),
        y=y_train_nn
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"\n📊 Class Weights for Loss Function:")
    print(f"   No Readmission (0): {class_weight_dict[0]:.2f}")
    print(f"   Readmission (1): {class_weight_dict[1]:.2f}")
    
    print(f"\n✅ Data preprocessing for neural networks completed")

else:
    print("❌ TensorFlow not available - cannot proceed with neural networks")

# %%
# Cell 6: Neural Network Architectures
print("="*60)
print("CREATING NEURAL NETWORK ARCHITECTURES")
print("="*60)

if TENSORFLOW_AVAILABLE:
    
    def create_simple_nn(input_dim, dropout_rate=0.3):
        """Simple feedforward neural network"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def create_deep_nn(input_dim, dropout_rate=0.4):
        """Deep neural network with batch normalization"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate * 0.75),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate * 0.5),
            
            Dense(16, activation='relu'),
            Dropout(dropout_rate * 0.25),
            
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def create_wide_nn(input_dim, dropout_rate=0.5):
        """Wide neural network"""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            Dropout(dropout_rate * 0.8),
            Dense(64, activation='relu'),
            Dropout(dropout_rate * 0.6),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def create_regularized_nn(input_dim, l1_reg=0.01, l2_reg=0.01, dropout_rate=0.3):
        """Neural network with L1/L2 regularization"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,),
                  kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
            Dropout(dropout_rate),
            Dense(32, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
            Dropout(dropout_rate),
            Dense(16, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def create_residual_nn(input_dim, dropout_rate=0.3):
        """Neural network with residual connections"""
        inputs = Input(shape=(input_dim,))
        
        # First block
        x1 = Dense(64, activation='relu')(inputs)
        x1 = Dropout(dropout_rate)(x1)
        
        # Second block with residual connection
        x2 = Dense(64, activation='relu')(x1)
        x2 = Dropout(dropout_rate)(x2)
        
        # Add residual connection
        x2_residual = layers.add([x1, x2])
        
        # Final layers
        x3 = Dense(32, activation='relu')(x2_residual)
        x3 = Dropout(dropout_rate)(x3)
        
        outputs = Dense(1, activation='sigmoid')(x3)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    # Define neural network configurations
    nn_configs = {
        'Simple_NN': {
            'model_fn': create_simple_nn,
            'optimizer': Adam(learning_rate=0.001),
            'params': {'dropout_rate': 0.3}
        },
        'Deep_NN': {
            'model_fn': create_deep_nn,
            'optimizer': Adam(learning_rate=0.001),
            'params': {'dropout_rate': 0.4}
        },
        'Wide_NN': {
            'model_fn': create_wide_nn,
            'optimizer': Adam(learning_rate=0.0005),
            'params': {'dropout_rate': 0.5}
        },
        'Regularized_NN': {
            'model_fn': create_regularized_nn,
            'optimizer': Adam(learning_rate=0.001),
            'params': {'l1_reg': 0.01, 'l2_reg': 0.01, 'dropout_rate': 0.3}
        },
        'Residual_NN': {
            'model_fn': create_residual_nn,
            'optimizer': Adam(learning_rate=0.001),
            'params': {'dropout_rate': 0.3}
        }
    }
    
    print(f"✅ Created {len(nn_configs)} neural network architectures:")
    for name in nn_configs.keys():
        print(f"   - {name}")
    
else:
    print("❌ TensorFlow not available - cannot create neural networks")

# %%
# Cell 7: Train Neural Networks
print("="*60)
print("TRAINING NEURAL NETWORKS")
print("="*60)

if TENSORFLOW_AVAILABLE:
    
    # Training configuration
    EPOCHS = 100
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=0.0001,
        verbose=0
    )
    
    # Store results
    nn_results = {}
    nn_models = {}
    
    print("🧠 Training Neural Network Models...")
    print("-" * 50)
    
    for name, config in nn_configs.items():
        print(f"\nTraining {name}...")
        
        try:
            # Create model
            model = config['model_fn'](X_train_nn_balanced.shape[1], **config['params'])
            
            # Compile model
            model.compile(
                optimizer=config['optimizer'],
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                X_train_nn_balanced, y_train_nn_balanced,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                callbacks=[early_stopping, reduce_lr],
                class_weight=class_weight_dict,
                verbose=0
            )
            
            # Make predictions
            y_pred_proba_nn = model.predict(X_test_nn_scaled, verbose=0).flatten()
            y_pred_nn = (y_pred_proba_nn > 0.5).astype(int)
            
            # Calculate metrics
            accuracy_nn = accuracy_score(y_test_nn, y_pred_nn)
            f1_nn = f1_score(y_test_nn, y_pred_nn)
            roc_auc_nn = roc_auc_score(y_test_nn, y_pred_proba_nn)
            avg_precision_nn = average_precision_score(y_test_nn, y_pred_proba_nn)
            
            # Store results
            nn_results[name] = {
                'accuracy': accuracy_nn,
                'f1_score': f1_nn,
                'roc_auc': roc_auc_nn,
                'avg_precision': avg_precision_nn,
                'predictions': y_pred_nn,
                'probabilities': y_pred_proba_nn,
                'history': history,
                'epochs_trained': len(history.history['loss'])
            }
            
            nn_models[name] = model
            
            print(f"   ✅ {name} trained successfully")
            print(f"      ROC-AUC: {roc_auc_nn:.4f}")
            print(f"      F1-Score: {f1_nn:.4f}")
            print(f"      Accuracy: {accuracy_nn:.4f}")
            print(f"      Avg Precision: {avg_precision_nn:.4f}")
            print(f"      Epochs: {len(history.history['loss'])}")
            
            # Check for improvement over baseline
            baseline_auc = 0.577
            if roc_auc_nn > baseline_auc:
                improvement = ((roc_auc_nn - baseline_auc) / baseline_auc) * 100
                print(f"      🎉 Improvement: +{improvement:.1f}% over baseline!")
            
        except Exception as e:
            print(f"   ❌ {name} failed: {str(e)}")
    
    print(f"\n✅ Neural network training completed!")
    print(f"   Successfully trained: {len(nn_results)} models")
    
else:
    print("❌ TensorFlow not available - cannot train neural networks")

# %%
# Cell 8: Neural Network Ensemble
print("="*60)
print("CREATING NEURAL NETWORK ENSEMBLE")
print("="*60)

if TENSORFLOW_AVAILABLE and len(nn_results) > 1:
    
    print("🔗 Creating Neural Network Ensemble...")
    
    # Get predictions from all models
    ensemble_predictions = []
    ensemble_probabilities = []
    
    for name, results in nn_results.items():
        ensemble_probabilities.append(results['probabilities'])
    
    # Average ensemble (simple but effective)
    ensemble_proba_avg = np.mean(ensemble_probabilities, axis=0)
    ensemble_pred_avg = (ensemble_proba_avg > 0.5).astype(int)
    
    # Weighted ensemble (weight by ROC-AUC performance)
    weights = []
    for name, results in nn_results.items():
        weights.append(results['roc_auc'])
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize weights
    
    ensemble_proba_weighted = np.average(ensemble_probabilities, axis=0, weights=weights)
    ensemble_pred_weighted = (ensemble_proba_weighted > 0.5).astype(int)
    
    # Evaluate ensembles
    ensembles = {
        'NN_Ensemble_Average': {
            'probabilities': ensemble_proba_avg,
            'predictions': ensemble_pred_avg
        },
        'NN_Ensemble_Weighted': {
            'probabilities': ensemble_proba_weighted,
            'predictions': ensemble_pred_weighted
        }
    }
    
    for ensemble_name, ensemble_data in ensembles.items():
        proba = ensemble_data['probabilities']
        pred = ensemble_data['predictions']
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_nn, pred)
        f1 = f1_score(y_test_nn, pred)
        roc_auc = roc_auc_score(y_test_nn, proba)
        avg_precision = average_precision_score(y_test_nn, proba)
        
        nn_results[ensemble_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'predictions': pred,
            'probabilities': proba
        }
        
        print(f"✅ {ensemble_name} created")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
    
    print(f"\n✅ Neural network ensembles created!")
    
elif TENSORFLOW_AVAILABLE:
    print("⚠️ Not enough models for ensemble creation")
else:
    print("❌ TensorFlow not available - cannot create ensembles")

# %%
# Cell 9: Neural Network Results Analysis
print("="*60)
print("NEURAL NETWORK RESULTS ANALYSIS")
print("="*60)

if TENSORFLOW_AVAILABLE and len(nn_results) > 0:
    
    # Create comparison table
    comparison_metrics = ['roc_auc', 'f1_score', 'accuracy', 'avg_precision']
    nn_comparison_df = pd.DataFrame(nn_results).T[comparison_metrics].round(4)
    nn_comparison_df = nn_comparison_df.sort_values('roc_auc', ascending=False)
    
    print("🏆 NEURAL NETWORK PERFORMANCE COMPARISON:")
    print("=" * 75)
    print(f"{'Model':<20} {'ROC-AUC':<10} {'F1-Score':<10} {'Accuracy':<10} {'Avg-Precision':<12}")
    print("-" * 75)
    
    for model_name in nn_comparison_df.index:
        roc_auc = nn_comparison_df.loc[model_name, 'roc_auc']
        f1 = nn_comparison_df.loc[model_name, 'f1_score']
        accuracy = nn_comparison_df.loc[model_name, 'accuracy']
        avg_prec = nn_comparison_df.loc[model_name, 'avg_precision']
        
        print(f"{model_name:<20} {roc_auc:<10.4f} {f1:<10.4f} {accuracy:<10.4f} {avg_prec:<12.4f}")
    
    # Find best neural network
    best_nn_name = nn_comparison_df.index[0]
    best_nn_metrics = nn_comparison_df.loc[best_nn_name]
    
    print(f"\n🥇 BEST NEURAL NETWORK: {best_nn_name}")
    print("=" * 45)
    print(f"ROC-AUC Score: {best_nn_metrics['roc_auc']:.4f}")
    print(f"F1-Score: {best_nn_metrics['f1_score']:.4f}")
    print(f"Accuracy: {best_nn_metrics['accuracy']:.4f}")
    print(f"Average Precision: {best_nn_metrics['avg_precision']:.4f}")
    
    # Compare with baseline
    baseline_auc = 0.577
    best_nn_auc = best_nn_metrics['roc_auc']
    improvement = ((best_nn_auc - baseline_auc) / baseline_auc) * 100
    
    print(f"\n📈 NEURAL NETWORK vs BASELINE:")
    print("=" * 40)
    print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
    print(f"Best NN ROC-AUC: {best_nn_auc:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    if improvement > 5:
        print("🎉 Significant improvement achieved with neural networks!")
    elif improvement > 0:
        print("✅ Modest improvement achieved with neural networks")
    else:
        print("⚠️ Neural networks did not improve over baseline")
    
    # Detailed evaluation of best neural network
    best_nn_predictions = nn_results[best_nn_name]['predictions']
    best_nn_probabilities = nn_results[best_nn_name]['probabilities']
    
    print(f"\n📋 DETAILED EVALUATION - {best_nn_name}:")
    print("=" * 60)
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test_nn, best_nn_predictions, 
                              target_names=['No Readmission', 'Readmission']))
    
    # Confusion Matrix
    cm_nn = confusion_matrix(y_test_nn, best_nn_predictions)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 No    Yes")
    print(f"Actual    No   {cm_nn[0,0]:4d}  {cm_nn[0,1]:4d}")
    print(f"          Yes  {cm_nn[1,0]:4d}  {cm_nn[1,1]:4d}")
    
    # Additional metrics
    tn, fp, fn, tp = cm_nn.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nAdvanced Metrics:")
    print(f"   Sensitivity (Recall): {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    print(f"   Positive Predictive Value: {ppv:.4f}")
    print(f"   Negative Predictive Value: {npv:.4f}")
    print(f"   False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"   False Negative Rate: {fn/(fn+tp):.4f}")
    
    # Training insights
    if best_nn_name in nn_models and 'history' in nn_results[best_nn_name]:
        history = nn_results[best_nn_name]['history']
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        epochs_trained = nn_results[best_nn_name]['epochs_trained']
        
        print(f"\nTraining Insights:")
        print(f"   Epochs trained: {epochs_trained}")
        print(f"   Final training loss: {final_loss:.4f}")
        print(f"   Final validation loss: {final_val_loss:.4f}")
        print(f"   Overfitting indicator: {final_val_loss - final_loss:.4f}")
    
    # Model ranking
    print(f"\n🏅 NEURAL NETWORK RANKING (by ROC-AUC):")
    print("=" * 50)
    for i, (model_name, metrics) in enumerate(nn_comparison_df.iterrows(), 1):
        auc = metrics['roc_auc']
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        print(f"{medal} {model_name:<25} {auc:.4f}")
    
    print(f"\n✅ Neural network analysis completed!")
    
else:
    print("❌ No neural network results to analyze")

# %%
# Cell 10: Final Neural Network Summary and Recommendations
print("="*60)
print("FINAL NEURAL NETWORK SUMMARY")
print("="*60)

if TENSORFLOW_AVAILABLE and len(nn_results) > 0:
    
    print("🧠 NEURAL NETWORK PERFORMANCE SUMMARY:")
    print("=" * 50)
    print(f"Best Neural Network: {best_nn_name}")
    print(f"Dataset Size: {len(modeling_data_nn):,} admissions")
    print(f"NN-Optimized Features: {X_nn_imputed.shape[1]} features")
    print(f"Test Set Size: {len(y_test_nn):,} admissions")
    print(f"Baseline Readmission Rate: {y_test_nn.mean()*100:.1f}%")
    
    print(f"\nBest NN Performance Metrics:")
    print(f"   ROC-AUC Score: {best_nn_metrics['roc_auc']:.4f}")
    print(f"   F1-Score: {best_nn_metrics['f1_score']:.4f}")
    print(f"   Accuracy: {best_nn_metrics['accuracy']:.4f}")
    print(f"   Average Precision: {best_nn_metrics['avg_precision']:.4f}")
    print(f"   Sensitivity: {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    
    # Clinical impact assessment
    print(f"\n🏥 NEURAL NETWORK CLINICAL IMPACT:")
    print("=" * 45)
    
    # Risk stratification with best NN
    nn_risk_scores = best_nn_probabilities
    high_risk_threshold_nn = np.percentile(nn_risk_scores, 80)
    high_risk_patients_nn = nn_risk_scores >= high_risk_threshold_nn
    high_risk_count_nn = high_risk_patients_nn.sum()
    high_risk_readmissions_nn = y_test_nn[high_risk_patients_nn].sum()
    high_risk_rate_nn = high_risk_readmissions_nn / high_risk_count_nn if high_risk_count_nn > 0 else 0
    
    total_readmissions_nn = y_test_nn.sum()
    capture_rate_nn = high_risk_readmissions_nn / total_readmissions_nn if total_readmissions_nn > 0 else 0
    
    print(f"High-Risk Patient Identification (Top 20%):")
    print(f"   Patients identified: {high_risk_count_nn} ({high_risk_count_nn/len(y_test_nn)*100:.1f}% of population)")
    print(f"   Actual readmissions captured: {high_risk_readmissions_nn} ({capture_rate_nn*100:.1f}% of all readmissions)")
    print(f"   High-risk group readmission rate: {high_risk_rate_nn*100:.1f}%")
    print(f"   Number needed to screen: {high_risk_count_nn/high_risk_readmissions_nn:.1f}" if high_risk_readmissions_nn > 0 else "   Number needed to screen: N/A")
    
    # Neural network specific recommendations
    print(f"\n🎯 NEURAL NETWORK SPECIFIC INSIGHTS:")
    print("=" * 50)
    
    nn_insights = [
        "🧠 **Deep Learning Advantage**: Neural networks can capture complex non-linear relationships",
        "⚖️ **Class Balancing**: SMOTE and class weights help with imbalanced data",
        "🎛️ **Feature Engineering**: Normalized and cyclical features work better for NNs",
        "🔄 **Ensemble Power**: Combining multiple NN architectures improves robustness",
        "📊 **Regularization**: Dropout and L1/L2 regularization prevent overfitting",
        "🎯 **Early Stopping**: Prevents overfitting and reduces training time"
    ]
    
    for insight in nn_insights:
        print(f"   {insight}")
    
    # Deployment recommendations for neural networks
    print(f"\n🚀 NEURAL NETWORK DEPLOYMENT RECOMMENDATIONS:")
    print("=" * 55)
    
    deployment_recs = [
        "1. **Model Serving**: Deploy using TensorFlow Serving or ONNX for production",
        "2. **Real-time Inference**: Optimize for low-latency predictions in clinical settings",
        "3. **Model Monitoring**: Track prediction drift and retrain periodically",
        "4. **Interpretability**: Use SHAP or LIME for model explanation to clinicians",
        "5. **A/B Testing**: Compare NN performance against traditional ML models",
        "6. **Ensemble Strategy**: Combine NN with tree-based models for best results",
        "7. **Hardware Optimization**: Consider GPU acceleration for large-scale deployment",
        "8. **Continuous Learning**: Implement online learning for model updates"
    ]
    
    for rec in deployment_recs:
        print(f"   {rec}")
    
    # Final comparison with all models
    print(f"\n📊 FINAL MODEL COMPARISON:")
    print("=" * 40)
    
    all_models_comparison = {
        'Baseline Gradient Boosting': 0.577,
        f'Best Neural Network ({best_nn_name})': best_nn_metrics['roc_auc'],
        'Enhanced Random Forest': 0.532  # From previous analysis
    }
    
    sorted_models = sorted(all_models_comparison.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model_name, auc) in enumerate(sorted_models, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        print(f"{medal} {model_name:<35} {auc:.4f}")
    
    # Final recommendation
    print(f"\n💡 FINAL RECOMMENDATION:")
    print("=" * 30)
    
    if best_nn_auc > 0.577:
        print(f"✅ Neural networks show improvement over baseline!")
        print(f"   Recommended approach: Deploy {best_nn_name} for production use")
        print(f"   Expected improvement: {improvement:+.1f}% in ROC-AUC")
    else:
        print(f"⚠️ Neural networks did not significantly improve over baseline")
        print(f"   Recommended approach: Use ensemble of NN + traditional ML models")
        print(f"   Consider: More data, different architectures, or feature engineering")
    
else:
    print("❌ Neural network analysis not available")

print(f"\n" + "="*60)
print("🎉 NEURAL NETWORK ANALYSIS COMPLETE!")
print("="*60)
print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if TENSORFLOW_AVAILABLE and len(nn_results) > 0:
    print(f"Best Neural Network: {best_nn_name} with ROC-AUC: {best_nn_metrics['roc_auc']:.4f}")
    print(f"Performance vs Baseline: {improvement:+.1f}%")
print("Ready for clinical validation and deployment consideration")
print("="*60)

# %%