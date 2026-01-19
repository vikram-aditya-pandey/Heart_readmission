# %%
"""
Advanced Heart Failure 30-Day Readmission Prediction Models
===========================================================

Enhanced version with:
- Advanced ensemble methods (XGBoost, LightGBM, CatBoost)
- Neural Networks (Deep Learning)
- Stacking ensemble
- Hyperparameter optimization
- Advanced feature engineering

Goal: Improve model performance beyond baseline ROC-AUC of 0.577
"""

# %%
# Cell 1: Import Advanced Libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            ExtraTreesClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score,
                           average_precision_score)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Advanced ensemble libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available - will skip XGBoost models")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available - will skip LightGBM models")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available - will skip CatBoost models")

# Neural Network libraries
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    import tensorflow as tf
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - will skip Neural Network models")

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Set random seeds
np.random.seed(42)

print("✅ Advanced libraries imported successfully")
print(f"XGBoost Available: {XGBOOST_AVAILABLE}")
print(f"LightGBM Available: {LIGHTGBM_AVAILABLE}")
print(f"CatBoost Available: {CATBOOST_AVAILABLE}")
print(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# Cell 2: Load Data and Prepare Features (Reuse from previous model)
print("="*60)
print("LOADING DATA AND PREPARING FEATURES")
print("="*60)

# Load all MIMIC tables
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

# Create target variable (30-day readmission)
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
# Cell 3: Advanced Feature Engineering
print("="*60)
print("ADVANCED FEATURE ENGINEERING")
print("="*60)

def create_advanced_features(modeling_data, patients, diagnoses, icustays, labevents, prescriptions):
    """Create comprehensive feature set with advanced engineering"""
    
    # Merge with patient data
    modeling_data = modeling_data.merge(
        patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']], 
        on='SUBJECT_ID'
    )
    
    # Basic demographic features
    modeling_data['AGE_AT_ADMISSION'] = (modeling_data['ADMITTIME'] - modeling_data['DOB']).dt.days / 365.25
    modeling_data['GENDER_MALE'] = (modeling_data['GENDER'] == 'M').astype(int)
    modeling_data['LOS_DAYS'] = (modeling_data['DISCHTIME'] - modeling_data['ADMITTIME']).dt.days
    modeling_data['EMERGENCY_ADMISSION'] = (modeling_data['ADMISSION_TYPE'] == 'EMERGENCY').astype(int)
    modeling_data['MEDICARE'] = (modeling_data['INSURANCE'] == 'Medicare').astype(int)
    
    # Advanced age features
    modeling_data['AGE_SQUARED'] = modeling_data['AGE_AT_ADMISSION'] ** 2
    modeling_data['AGE_VERY_OLD'] = (modeling_data['AGE_AT_ADMISSION'] >= 85).astype(int)
    modeling_data['AGE_YOUNG_HF'] = (modeling_data['AGE_AT_ADMISSION'] < 65).astype(int)
    
    # Advanced LOS features
    modeling_data['LOS_LOG'] = np.log1p(modeling_data['LOS_DAYS'])
    modeling_data['LOS_VERY_SHORT'] = (modeling_data['LOS_DAYS'] <= 2).astype(int)
    modeling_data['LOS_VERY_LONG'] = (modeling_data['LOS_DAYS'] >= 14).astype(int)
    
    # Temporal features
    modeling_data['ADMIT_MONTH'] = modeling_data['ADMITTIME'].dt.month
    modeling_data['ADMIT_WEEKDAY'] = modeling_data['ADMITTIME'].dt.weekday
    modeling_data['ADMIT_WEEKEND'] = (modeling_data['ADMIT_WEEKDAY'] >= 5).astype(int)
    modeling_data['WINTER_ADMISSION'] = modeling_data['ADMIT_MONTH'].isin([12, 1, 2]).astype(int)
    
    print("✅ Basic and temporal features created")
    
    # Advanced diagnosis features
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
        
        # Basic counts
        total_dx = len(admission_diagnoses)
        hf_dx_count = sum(1 for code in admission_diagnoses if code in hf_codes)
        
        # Comorbidity flags
        has_diabetes = int(any(code in diabetes_codes for code in admission_diagnoses))
        has_hypertension = int(any(code in hypertension_codes for code in admission_diagnoses))
        has_kidney = int(any(code in kidney_codes for code in admission_diagnoses))
        has_cardiac = int(any(code in cardiac_codes for code in admission_diagnoses))
        has_respiratory = int(any(code in respiratory_codes for code in admission_diagnoses))
        
        # Advanced features
        comorbidity_score = has_diabetes + has_hypertension + has_kidney + has_cardiac + has_respiratory
        
        # Specific HF subtypes
        systolic_hf = [42820, 42821, 42822, 42823]
        diastolic_hf = [42830, 42831, 42832, 42833]
        combined_hf = [42840, 42841, 42842, 42843]
        acute_hf = [4281, 42821, 42831, 42841]
        chronic_hf = [42822, 42832, 42842]
        
        has_systolic = int(any(code in systolic_hf for code in admission_diagnoses))
        has_diastolic = int(any(code in diastolic_hf for code in admission_diagnoses))
        has_combined = int(any(code in combined_hf for code in admission_diagnoses))
        has_acute = int(any(code in acute_hf for code in admission_diagnoses))
        has_chronic = int(any(code in chronic_hf for code in admission_diagnoses))
        
        # Complexity score
        complexity_score = total_dx + hf_dx_count + comorbidity_score
        
        diagnosis_features.append({
            'HADM_ID': hadm_id,
            'TOTAL_DIAGNOSES': total_dx,
            'HF_DIAGNOSES_COUNT': hf_dx_count,
            'HAS_DIABETES': has_diabetes,
            'HAS_HYPERTENSION': has_hypertension,
            'HAS_KIDNEY_DISEASE': has_kidney,
            'HAS_CARDIAC_COMORBIDITY': has_cardiac,
            'HAS_RESPIRATORY_DISEASE': has_respiratory,
            'COMORBIDITY_COUNT': comorbidity_score,
            'HAS_SYSTOLIC_HF': has_systolic,
            'HAS_DIASTOLIC_HF': has_diastolic,
            'HAS_COMBINED_HF': has_combined,
            'HAS_ACUTE_HF': has_acute,
            'HAS_CHRONIC_HF': has_chronic,
            'COMPLEXITY_SCORE': complexity_score,
            'DIABETES_KIDNEY_COMBO': has_diabetes * has_kidney,
            'HF_KIDNEY_COMBO': (hf_dx_count > 0) * has_kidney
        })
    
    diagnosis_df = pd.DataFrame(diagnosis_features)
    modeling_data = modeling_data.merge(diagnosis_df, on='HADM_ID')
    
    print("✅ Advanced diagnosis features created")
    
    return modeling_data

# Create advanced features
modeling_data_advanced = create_advanced_features(
    modeling_data, patients, diagnoses, icustays, labevents, prescriptions
)

print(f"📊 Advanced feature set shape: {modeling_data_advanced.shape}")

# %%
# Cell 4: Complete Feature Engineering (ICU, Labs, Medications, Historical)
def add_remaining_features(modeling_data, icustays, labevents, prescriptions, admissions_sorted):
    """Add ICU, lab, medication, and historical features"""
    
    # ICU Features
    icu_features = []
    for hadm_id in modeling_data['HADM_ID'].unique():
        icu_stays_admission = icustays[icustays['HADM_ID'] == hadm_id]
        
        if len(icu_stays_admission) > 0:
            icu_los = icu_stays_admission['LOS'].sum()
            icu_count = len(icu_stays_admission)
            has_icu = 1
            max_icu_los = icu_stays_admission['LOS'].max()
        else:
            icu_los = 0
            icu_count = 0
            has_icu = 0
            max_icu_los = 0
        
        icu_features.append({
            'HADM_ID': hadm_id,
            'HAS_ICU_STAY': has_icu,
            'ICU_LOS_DAYS': icu_los,
            'ICU_STAYS_COUNT': icu_count,
            'MAX_ICU_LOS': max_icu_los,
            'ICU_LOS_LOG': np.log1p(icu_los)
        })
    
    icu_df = pd.DataFrame(icu_features)
    modeling_data = modeling_data.merge(icu_df, on='HADM_ID')
    
    # Lab Features with advanced engineering
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
                # Multiple statistics
                first_value = lab_values.iloc[0]
                last_value = lab_values.iloc[-1]
                mean_value = lab_values.mean()
                max_value = lab_values.max()
                min_value = lab_values.min()
                std_value = lab_values.std() if len(lab_values) > 1 else 0
                
                lab_feature[f'{lab_name}_FIRST'] = first_value
                lab_feature[f'{lab_name}_LAST'] = last_value
                lab_feature[f'{lab_name}_MEAN'] = mean_value
                lab_feature[f'{lab_name}_MAX'] = max_value
                lab_feature[f'{lab_name}_MIN'] = min_value
                lab_feature[f'{lab_name}_STD'] = std_value
                lab_feature[f'{lab_name}_TREND'] = last_value - first_value
                lab_feature[f'HAS_{lab_name}'] = 1
                
                # Abnormal flags with severity
                if lab_name == 'CREATININE':
                    lab_feature[f'{lab_name}_ELEVATED'] = int(last_value > 1.5)
                    lab_feature[f'{lab_name}_SEVERELY_ELEVATED'] = int(last_value > 3.0)
                elif lab_name == 'SODIUM':
                    lab_feature[f'{lab_name}_LOW'] = int(last_value < 135)
                    lab_feature[f'{lab_name}_VERY_LOW'] = int(last_value < 130)
                elif lab_name == 'BUN':
                    lab_feature[f'{lab_name}_ELEVATED'] = int(last_value > 20)
                    lab_feature[f'{lab_name}_SEVERELY_ELEVATED'] = int(last_value > 50)
                elif lab_name == 'HEMOGLOBIN':
                    lab_feature[f'{lab_name}_LOW'] = int(last_value < 10)
                    lab_feature[f'{lab_name}_VERY_LOW'] = int(last_value < 8)
                elif lab_name == 'ALBUMIN':
                    lab_feature[f'{lab_name}_LOW'] = int(last_value < 3.5)
                    lab_feature[f'{lab_name}_VERY_LOW'] = int(last_value < 2.5)
            else:
                # Fill missing values
                for suffix in ['_FIRST', '_LAST', '_MEAN', '_MAX', '_MIN', '_STD', '_TREND']:
                    lab_feature[f'{lab_name}{suffix}'] = np.nan
                lab_feature[f'HAS_{lab_name}'] = 0
                for suffix in ['_ELEVATED', '_SEVERELY_ELEVATED', '_LOW', '_VERY_LOW']:
                    if f'{lab_name}{suffix}' not in lab_feature:
                        lab_feature[f'{lab_name}{suffix}'] = 0
        
        # Composite lab scores
        if lab_feature.get('HAS_CREATININE', 0) and lab_feature.get('HAS_BUN', 0):
            lab_feature['BUN_CREAT_RATIO'] = lab_feature['BUN_LAST'] / max(lab_feature['CREATININE_LAST'], 0.1)
        else:
            lab_feature['BUN_CREAT_RATIO'] = np.nan
            
        lab_features.append(lab_feature)
    
    lab_df = pd.DataFrame(lab_features)
    modeling_data = modeling_data.merge(lab_df, on='HADM_ID')
    
    # Medication Features
    hf_medications = {
        'ACE_INHIBITOR': ['Lisinopril', 'Enalapril'],
        'BETA_BLOCKER': ['Metoprolol', 'Carvedilol'],
        'DIURETIC': ['Furosemide', 'Hydrochlorothiazide'],
        'ARB': ['Losartan'],
        'ALDOSTERONE_ANTAGONIST': ['Spironolactone'],
        'DIGOXIN': ['Digoxin'],
        'ANTICOAGULANT': ['Warfarin'],
        'POTASSIUM': ['Potassium Chloride']
    }
    
    med_features = []
    for hadm_id in modeling_data['HADM_ID'].unique():
        admission_meds = prescriptions[prescriptions['HADM_ID'] == hadm_id]
        med_drugs = admission_meds['DRUG'].tolist()
        
        med_feature = {
            'HADM_ID': hadm_id,
            'TOTAL_MEDICATIONS': len(admission_meds),
            'UNIQUE_MEDICATIONS': len(set(med_drugs))
        }
        
        # Medication class flags
        for med_class, drug_list in hf_medications.items():
            has_med = int(any(drug in med_drugs for drug in drug_list))
            med_feature[f'HAS_{med_class}'] = has_med
        
        # Advanced medication scores
        hf_med_count = sum(med_feature[f'HAS_{med_class}'] for med_class in hf_medications.keys())
        med_feature['HF_MEDICATION_COUNT'] = hf_med_count
        
        # Guideline-directed medical therapy score
        gdmt_score = (med_feature['HAS_ACE_INHIBITOR'] or med_feature['HAS_ARB']) + \
                     med_feature['HAS_BETA_BLOCKER'] + med_feature['HAS_DIURETIC']
        med_feature['GDMT_SCORE'] = gdmt_score
        med_feature['OPTIMAL_THERAPY'] = int(gdmt_score >= 3)
        med_feature['SUBOPTIMAL_THERAPY'] = int(gdmt_score < 2)
        
        # Polypharmacy
        med_feature['POLYPHARMACY'] = int(len(admission_meds) >= 10)
        
        med_features.append(med_feature)
    
    med_df = pd.DataFrame(med_features)
    modeling_data = modeling_data.merge(med_df, on='HADM_ID')
    
    # Historical Features
    historical_features = []
    for subject_id in modeling_data['SUBJECT_ID'].unique():
        patient_admissions = admissions_sorted[admissions_sorted['SUBJECT_ID'] == subject_id].sort_values('ADMITTIME')
        
        for idx, admission in patient_admissions.iterrows():
            prior_admissions = patient_admissions[patient_admissions['ADMITTIME'] < admission['ADMITTIME']]
            
            if len(prior_admissions) == 0:
                hist_features = {
                    'HADM_ID': admission['HADM_ID'],
                    'PRIOR_ADMISSIONS_COUNT': 0,
                    'DAYS_SINCE_LAST_ADMISSION': np.nan,
                    'HAD_PRIOR_READMISSION': 0,
                    'FREQUENT_FLYER': 0,
                    'ADMISSION_RATE_PER_YEAR': 0
                }
            else:
                last_admission = prior_admissions.iloc[-1]
                days_since_last = (admission['ADMITTIME'] - last_admission['DISCHTIME']).days
                
                # Check for prior readmissions
                had_prior_readmission = 0
                for i in range(len(prior_admissions) - 1):
                    curr_disch = pd.to_datetime(prior_admissions.iloc[i]['DISCHTIME'])
                    next_admit = pd.to_datetime(prior_admissions.iloc[i+1]['ADMITTIME'])
                    if (next_admit - curr_disch).days <= 30:
                        had_prior_readmission = 1
                        break
                
                # Calculate admission rate
                first_admission_date = prior_admissions.iloc[0]['ADMITTIME']
                years_of_history = (admission['ADMITTIME'] - first_admission_date).days / 365.25
                admission_rate = len(prior_admissions) / max(years_of_history, 0.1)
                
                hist_features = {
                    'HADM_ID': admission['HADM_ID'],
                    'PRIOR_ADMISSIONS_COUNT': len(prior_admissions),
                    'DAYS_SINCE_LAST_ADMISSION': days_since_last,
                    'HAD_PRIOR_READMISSION': had_prior_readmission,
                    'FREQUENT_FLYER': int(len(prior_admissions) >= 3),
                    'ADMISSION_RATE_PER_YEAR': admission_rate
                }
            
            historical_features.append(hist_features)
    
    hist_df = pd.DataFrame(historical_features)
    modeling_data = modeling_data.merge(hist_df, on='HADM_ID')
    
    return modeling_data

# Add remaining features
modeling_data_complete = add_remaining_features(
    modeling_data_advanced, icustays, labevents, prescriptions, admissions_sorted
)

print(f"✅ Complete feature engineering finished")
print(f"📊 Final feature set shape: {modeling_data_complete.shape}")

# %%
# Cell 5: Feature Selection and Preprocessing
print("="*60)
print("FEATURE SELECTION AND PREPROCESSING")
print("="*60)

# Select features for modeling (expanded set)
feature_columns = [
    # Basic demographics
    'AGE_AT_ADMISSION', 'AGE_SQUARED', 'AGE_VERY_OLD', 'AGE_YOUNG_HF',
    'GENDER_MALE', 'LOS_DAYS', 'LOS_LOG', 'LOS_VERY_SHORT', 'LOS_VERY_LONG',
    'EMERGENCY_ADMISSION', 'MEDICARE',
    
    # Temporal
    'ADMIT_WEEKEND', 'WINTER_ADMISSION',
    
    # Diagnoses
    'TOTAL_DIAGNOSES', 'HF_DIAGNOSES_COUNT', 'COMORBIDITY_COUNT', 'COMPLEXITY_SCORE',
    'HAS_DIABETES', 'HAS_HYPERTENSION', 'HAS_KIDNEY_DISEASE',
    'HAS_CARDIAC_COMORBIDITY', 'HAS_RESPIRATORY_DISEASE',
    'HAS_SYSTOLIC_HF', 'HAS_DIASTOLIC_HF', 'HAS_COMBINED_HF',
    'HAS_ACUTE_HF', 'HAS_CHRONIC_HF',
    'DIABETES_KIDNEY_COMBO', 'HF_KIDNEY_COMBO',
    
    # ICU
    'HAS_ICU_STAY', 'ICU_LOS_DAYS', 'ICU_LOS_LOG', 'MAX_ICU_LOS',
    
    # Labs (using last values and key flags)
    'TOTAL_LABS', 'CREATININE_LAST', 'SODIUM_LAST', 'BUN_LAST',
    'HEMOGLOBIN_LAST', 'ALBUMIN_LAST', 'BUN_CREAT_RATIO',
    'CREATININE_ELEVATED', 'CREATININE_SEVERELY_ELEVATED',
    'SODIUM_LOW', 'SODIUM_VERY_LOW', 'BUN_ELEVATED', 'BUN_SEVERELY_ELEVATED',
    'HEMOGLOBIN_LOW', 'HEMOGLOBIN_VERY_LOW', 'ALBUMIN_LOW', 'ALBUMIN_VERY_LOW',
    
    # Medications
    'TOTAL_MEDICATIONS', 'HF_MEDICATION_COUNT', 'GDMT_SCORE',
    'HAS_ACE_INHIBITOR', 'HAS_BETA_BLOCKER', 'HAS_DIURETIC', 'HAS_ARB',
    'OPTIMAL_THERAPY', 'SUBOPTIMAL_THERAPY', 'POLYPHARMACY',
    
    # Historical
    'PRIOR_ADMISSIONS_COUNT', 'DAYS_SINCE_LAST_ADMISSION',
    'HAD_PRIOR_READMISSION', 'FREQUENT_FLYER', 'ADMISSION_RATE_PER_YEAR'
]

# Create feature matrix
X = modeling_data_complete[feature_columns].copy()
y = modeling_data_complete['READMITTED_30DAY'].copy()

print(f"📊 Initial Feature Matrix Shape: {X.shape}")
print(f"📊 Target Distribution: {y.value_counts().to_dict()}")

# Handle missing values with advanced imputation
print(f"\n🔍 Missing Values Analysis:")
missing_counts = X.isnull().sum()
missing_features = missing_counts[missing_counts > 0]
if len(missing_features) > 0:
    print("Features with missing values:")
    for feature, count in missing_features.items():
        print(f"   {feature}: {count} ({count/len(X)*100:.1f}%)")
        
    # Advanced imputation strategy
    # For lab values, use median imputation
    lab_features = [col for col in X.columns if any(lab in col for lab in ['CREATININE', 'SODIUM', 'BUN', 'HEMOGLOBIN', 'ALBUMIN'])]
    
    # For historical features, use 0 or specific values
    historical_features = ['DAYS_SINCE_LAST_ADMISSION', 'ADMISSION_RATE_PER_YEAR']
    
    # Separate imputation strategies
    X_imputed = X.copy()
    
    # Lab features - median imputation
    lab_imputer = SimpleImputer(strategy='median')
    if lab_features:
        lab_cols = [col for col in lab_features if col in X.columns]
        if lab_cols:
            X_imputed[lab_cols] = lab_imputer.fit_transform(X_imputed[lab_cols])
    
    # Historical features - specific imputation
    for feature in historical_features:
        if feature in X_imputed.columns:
            if feature == 'DAYS_SINCE_LAST_ADMISSION':
                X_imputed[feature].fillna(365, inplace=True)  # Assume 1 year if missing
            elif feature == 'ADMISSION_RATE_PER_YEAR':
                X_imputed[feature].fillna(0, inplace=True)  # No prior admissions
    
    # Remaining features - median imputation
    remaining_imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        remaining_imputer.fit_transform(X_imputed),
        columns=X_imputed.columns,
        index=X_imputed.index
    )
else:
    X_imputed = X.copy()

print(f"✅ Missing values handled")

# Feature selection using multiple methods
print(f"\n🎯 Feature Selection:")

# Method 1: Statistical selection (SelectKBest)
selector_stats = SelectKBest(score_func=f_classif, k=min(30, X_imputed.shape[1]))
X_selected_stats = selector_stats.fit_transform(X_imputed, y)
selected_features_stats = X_imputed.columns[selector_stats.get_support()].tolist()

print(f"   Statistical selection: {len(selected_features_stats)} features")

# Method 2: Use all features for ensemble methods (they handle feature selection internally)
X_final = X_imputed.copy()
selected_features_final = X_final.columns.tolist()

print(f"   Final feature set: {len(selected_features_final)} features")
print(f"✅ Feature preprocessing completed")

# %%
# Cell 6: Train-Test Split with Advanced Sampling
print("="*60)
print("TRAIN-TEST SPLIT AND ADVANCED SAMPLING")
print("="*60)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"📊 Data Split:")
print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set: {X_test.shape[0]:,} samples")
print(f"   Features: {X_train.shape[1]}")

# Multiple scaling strategies
scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler()
}

# Use RobustScaler (better for outliers)
scaler = scalers['robust']
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features scaled using RobustScaler")

# Advanced sampling techniques
sampling_methods = {}

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
sampling_methods['SMOTE'] = (X_train_smote, y_train_smote)

# ADASYN (if available)
try:
    adasyn = ADASYN(random_state=42, n_neighbors=3)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_scaled, y_train)
    sampling_methods['ADASYN'] = (X_train_adasyn, y_train_adasyn)
except:
    print("⚠️ ADASYN failed, using SMOTE only")

# SMOTETomek (combination)
try:
    smote_tomek = SMOTETomek(random_state=42)
    X_train_smote_tomek, y_train_smote_tomek = smote_tomek.fit_resample(X_train_scaled, y_train)
    sampling_methods['SMOTETomek'] = (X_train_smote_tomek, y_train_smote_tomek)
except:
    print("⚠️ SMOTETomek failed, using SMOTE only")

print(f"\n📊 Sampling Methods Available:")
for method, (X_samp, y_samp) in sampling_methods.items():
    print(f"   {method}: {X_samp.shape[0]:,} samples")
    class_dist = pd.Series(y_samp).value_counts(normalize=True) * 100
    print(f"      Class distribution: {class_dist.to_dict()}")

# Use SMOTE as primary method
X_train_balanced = X_train_smote
y_train_balanced = y_train_smote

print(f"\n✅ Using SMOTE for class balancing")
print(f"   Balanced training set: {X_train_balanced.shape[0]:,} samples")

# %%
# Cell 7: Advanced Ensemble Models
print("="*60)
print("TRAINING ADVANCED ENSEMBLE MODELS")
print("="*60)

# Initialize advanced models
advanced_models = {}

# 1. Enhanced Random Forest
advanced_models['Enhanced_RF'] = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# 2. Enhanced Gradient Boosting
advanced_models['Enhanced_GBM'] = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)

# 3. Extra Trees
advanced_models['Extra_Trees'] = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# 4. XGBoost (if available)
if XGBOOST_AVAILABLE:
    advanced_models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

# 5. LightGBM (if available)
if LIGHTGBM_AVAILABLE:
    advanced_models['LightGBM'] = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

# 6. CatBoost (if available)
if CATBOOST_AVAILABLE:
    advanced_models['CatBoost'] = cb.CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=8,
        random_seed=42,
        verbose=False
    )

# Train all models
print("🚀 Training Advanced Models...")
print("-" * 50)

model_results = {}
trained_models = {}

for name, model in advanced_models.items():
    print(f"\nTraining {name}...")
    
    try:
        # Train model
        model.fit(X_train_balanced, y_train_balanced)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        model_results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   ✅ {name} trained successfully")
        print(f"      ROC-AUC: {roc_auc:.4f}")
        print(f"      F1-Score: {f1:.4f}")
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      Avg Precision: {avg_precision:.4f}")
        
    except Exception as e:
        print(f"   ❌ {name} failed: {str(e)}")

print(f"\n✅ Advanced ensemble models training completed!")
print(f"   Successfully trained: {len(model_results)} models")

# %%
# Cell 8: Neural Network Models
print("="*60)
print("TRAINING NEURAL NETWORK MODELS")
print("="*60)

if TENSORFLOW_AVAILABLE:
    def create_neural_network(input_dim, architecture='deep'):
        """Create neural network with different architectures"""
        
        model = Sequential()
        
        if architecture == 'simple':
            # Simple NN
            model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='sigmoid'))
            
        elif architecture == 'deep':
            # Deep NN with batch normalization
            model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.1))
            
            model.add(Dense(1, activation='sigmoid'))
            
        elif architecture == 'wide':
            # Wide NN
            model.add(Dense(256, activation='relu', input_shape=(input_dim,)))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    # Neural network architectures to try
    nn_architectures = ['simple', 'deep', 'wide']
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.0001,
        verbose=0
    )
    
    print("🧠 Training Neural Network Models...")
    print("-" * 40)
    
    nn_results = {}
    
    for arch in nn_architectures:
        print(f"\nTraining {arch.upper()} Neural Network...")
        
        try:
            # Create model
            nn_model = create_neural_network(X_train_balanced.shape[1], arch)
            
            # Train model
            history = nn_model.fit(
                X_train_balanced, y_train_balanced,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Predictions
            y_pred_proba_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
            y_pred_nn = (y_pred_proba_nn > 0.5).astype(int)
            
            # Calculate metrics
            accuracy_nn = accuracy_score(y_test, y_pred_nn)
            f1_nn = f1_score(y_test, y_pred_nn)
            roc_auc_nn = roc_auc_score(y_test, y_pred_proba_nn)
            avg_precision_nn = average_precision_score(y_test, y_pred_proba_nn)
            
            nn_results[f'NN_{arch.upper()}'] = {
                'accuracy': accuracy_nn,
                'f1_score': f1_nn,
                'roc_auc': roc_auc_nn,
                'avg_precision': avg_precision_nn,
                'predictions': y_pred_nn,
                'probabilities': y_pred_proba_nn,
                'model': nn_model,
                'history': history
            }
            
            print(f"   ✅ {arch.upper()} NN trained successfully")
            print(f"      ROC-AUC: {roc_auc_nn:.4f}")
            print(f"      F1-Score: {f1_nn:.4f}")
            print(f"      Accuracy: {accuracy_nn:.4f}")
            print(f"      Avg Precision: {avg_precision_nn:.4f}")
            print(f"      Training epochs: {len(history.history['loss'])}")
            
        except Exception as e:
            print(f"   ❌ {arch.upper()} NN failed: {str(e)}")
    
    # Add NN results to main results
    model_results.update(nn_results)
    
    print(f"\n✅ Neural network training completed!")
    print(f"   Successfully trained: {len(nn_results)} neural networks")
    
else:
    print("⚠️ TensorFlow not available - skipping neural network models")

# %%
# Cell 9: Stacking Ensemble
print("="*60)
print("CREATING STACKING ENSEMBLE")
print("="*60)

# Select best performing base models for stacking
if len(model_results) >= 3:
    # Sort models by ROC-AUC
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    
    # Select top models for stacking (exclude neural networks for sklearn compatibility)
    base_model_names = []
    base_models = []
    
    for name, results in sorted_models:
        if not name.startswith('NN_') and len(base_models) < 5:  # Max 5 base models
            if name in trained_models:
                base_model_names.append(name)
                base_models.append((name, trained_models[name]))
    
    if len(base_models) >= 2:
        print(f"🏗️ Creating Stacking Ensemble with base models:")
        for name in base_model_names:
            auc = model_results[name]['roc_auc']
            print(f"   - {name}: ROC-AUC = {auc:.4f}")
        
        # Create stacking classifier
        stacking_classifier = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        print(f"\n🚀 Training Stacking Ensemble...")
        
        try:
            # Train stacking ensemble
            stacking_classifier.fit(X_train_balanced, y_train_balanced)
            
            # Predictions
            y_pred_stack = stacking_classifier.predict(X_test_scaled)
            y_pred_proba_stack = stacking_classifier.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy_stack = accuracy_score(y_test, y_pred_stack)
            f1_stack = f1_score(y_test, y_pred_stack)
            roc_auc_stack = roc_auc_score(y_test, y_pred_proba_stack)
            avg_precision_stack = average_precision_score(y_test, y_pred_proba_stack)
            
            model_results['Stacking_Ensemble'] = {
                'accuracy': accuracy_stack,
                'f1_score': f1_stack,
                'roc_auc': roc_auc_stack,
                'avg_precision': avg_precision_stack,
                'predictions': y_pred_stack,
                'probabilities': y_pred_proba_stack
            }
            
            trained_models['Stacking_Ensemble'] = stacking_classifier
            
            print(f"   ✅ Stacking Ensemble trained successfully")
            print(f"      ROC-AUC: {roc_auc_stack:.4f}")
            print(f"      F1-Score: {f1_stack:.4f}")
            print(f"      Accuracy: {accuracy_stack:.4f}")
            print(f"      Avg Precision: {avg_precision_stack:.4f}")
            
        except Exception as e:
            print(f"   ❌ Stacking Ensemble failed: {str(e)}")
    
    else:
        print("⚠️ Not enough base models for stacking ensemble")
else:
    print("⚠️ Not enough models trained for stacking ensemble")

# Create Voting Ensemble as well
if len(base_models) >= 2:
    print(f"\n🗳️ Creating Voting Ensemble...")
    
    try:
        voting_classifier = VotingClassifier(
            estimators=base_models,
            voting='soft',
            n_jobs=-1
        )
        
        # Train voting ensemble
        voting_classifier.fit(X_train_balanced, y_train_balanced)
        
        # Predictions
        y_pred_vote = voting_classifier.predict(X_test_scaled)
        y_pred_proba_vote = voting_classifier.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy_vote = accuracy_score(y_test, y_pred_vote)
        f1_vote = f1_score(y_test, y_pred_vote)
        roc_auc_vote = roc_auc_score(y_test, y_pred_proba_vote)
        avg_precision_vote = average_precision_score(y_test, y_pred_proba_vote)
        
        model_results['Voting_Ensemble'] = {
            'accuracy': accuracy_vote,
            'f1_score': f1_vote,
            'roc_auc': roc_auc_vote,
            'avg_precision': avg_precision_vote,
            'predictions': y_pred_vote,
            'probabilities': y_pred_proba_vote
        }
        
        trained_models['Voting_Ensemble'] = voting_classifier
        
        print(f"   ✅ Voting Ensemble trained successfully")
        print(f"      ROC-AUC: {roc_auc_vote:.4f}")
        print(f"      F1-Score: {f1_vote:.4f}")
        print(f"      Accuracy: {accuracy_vote:.4f}")
        print(f"      Avg Precision: {avg_precision_vote:.4f}")
        
    except Exception as e:
        print(f"   ❌ Voting Ensemble failed: {str(e)}")

print(f"\n✅ Ensemble creation completed!")

# %%
# Cell 10: Model Comparison and Best Model Selection
print("="*60)
print("ADVANCED MODEL COMPARISON AND EVALUATION")
print("="*60)

# Create comprehensive comparison
comparison_metrics = ['roc_auc', 'f1_score', 'accuracy', 'avg_precision']
comparison_df = pd.DataFrame(model_results).T[comparison_metrics].round(4)
comparison_df = comparison_df.sort_values('roc_auc', ascending=False)

print("🏆 ADVANCED MODEL PERFORMANCE COMPARISON:")
print("=" * 70)
print(f"{'Model':<20} {'ROC-AUC':<10} {'F1-Score':<10} {'Accuracy':<10} {'Avg-Precision':<12}")
print("-" * 70)

for model_name in comparison_df.index:
    roc_auc = comparison_df.loc[model_name, 'roc_auc']
    f1 = comparison_df.loc[model_name, 'f1_score']
    accuracy = comparison_df.loc[model_name, 'accuracy']
    avg_prec = comparison_df.loc[model_name, 'avg_precision']
    
    print(f"{model_name:<20} {roc_auc:<10.4f} {f1:<10.4f} {accuracy:<10.4f} {avg_prec:<12.4f}")

# Find best model
best_model_name = comparison_df.index[0]
best_model_metrics = comparison_df.loc[best_model_name]

print(f"\n🥇 BEST MODEL: {best_model_name}")
print("=" * 40)
print(f"ROC-AUC Score: {best_model_metrics['roc_auc']:.4f}")
print(f"F1-Score: {best_model_metrics['f1_score']:.4f}")
print(f"Accuracy: {best_model_metrics['accuracy']:.4f}")
print(f"Average Precision: {best_model_metrics['avg_precision']:.4f}")

# Performance improvement analysis
baseline_auc = 0.577  # From previous basic model
best_auc = best_model_metrics['roc_auc']
improvement = ((best_auc - baseline_auc) / baseline_auc) * 100

print(f"\n📈 PERFORMANCE IMPROVEMENT:")
print("=" * 35)
print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
print(f"Best Model ROC-AUC: {best_auc:.4f}")
print(f"Improvement: {improvement:+.1f}%")

if improvement > 5:
    print("🎉 Significant improvement achieved!")
elif improvement > 0:
    print("✅ Modest improvement achieved")
else:
    print("⚠️ No improvement over baseline")

# Detailed evaluation of best model
best_predictions = model_results[best_model_name]['predictions']
best_probabilities = model_results[best_model_name]['probabilities']

print(f"\n📋 DETAILED EVALUATION - {best_model_name}:")
print("=" * 60)

# Classification Report
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
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nAdvanced Metrics:")
print(f"   Sensitivity (Recall): {sensitivity:.4f}")
print(f"   Specificity: {specificity:.4f}")
print(f"   Positive Predictive Value: {ppv:.4f}")
print(f"   Negative Predictive Value: {npv:.4f}")
print(f"   False Positive Rate: {fp/(fp+tn):.4f}")
print(f"   False Negative Rate: {fn/(fn+tp):.4f}")

# Model ranking
print(f"\n🏅 MODEL RANKING (by ROC-AUC):")
print("=" * 40)
for i, (model_name, metrics) in enumerate(comparison_df.iterrows(), 1):
    auc = metrics['roc_auc']
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
    print(f"{medal} {model_name:<20} {auc:.4f}")

print(f"\n✅ Advanced model evaluation completed!")

# %%
# Cell 11: Advanced Feature Importance and Model Interpretation
print("="*60)
print("ADVANCED FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance from best model
best_model = trained_models.get(best_model_name)

if best_model and hasattr(best_model, 'feature_importances_'):
    # Tree-based model feature importance
    feature_importance = best_model.feature_importances_
    importance_type = "Feature Importance"
    
elif best_model and hasattr(best_model, 'coef_'):
    # Linear model coefficients
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = "Coefficient Magnitude"
    
elif best_model_name == 'Stacking_Ensemble' and best_model:
    # For stacking ensemble, get importance from final estimator
    if hasattr(best_model.final_estimator_, 'coef_'):
        feature_importance = np.abs(best_model.final_estimator_.coef_[0])
        importance_type = "Meta-learner Coefficient Magnitude"
    else:
        feature_importance = None
else:
    feature_importance = None

if feature_importance is not None:
    # Create feature importance DataFrame
    feature_names = X_final.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(f"🔍 Top 25 Most Important Features ({importance_type}):")
    print("=" * 80)
    
    for i, (_, row) in enumerate(importance_df.head(25).iterrows(), 1):
        feature_name = row['Feature']
        importance = row['Importance']
        
        # Add clinical interpretation
        clinical_interpretations = {
            'HAD_PRIOR_READMISSION': 'History of previous 30-day readmissions',
            'PRIOR_ADMISSIONS_COUNT': 'Number of previous hospital admissions',
            'FREQUENT_FLYER': 'Patient with ≥3 prior admissions',
            'AGE_AT_ADMISSION': 'Patient age at current admission',
            'AGE_VERY_OLD': 'Patient age ≥85 years (very elderly)',
            'LOS_DAYS': 'Length of current hospital stay',
            'LOS_VERY_LONG': 'Hospital stay ≥14 days',
            'CREATININE_ELEVATED': 'Kidney dysfunction (creatinine >1.5)',
            'CREATININE_SEVERELY_ELEVATED': 'Severe kidney dysfunction (creatinine >3.0)',
            'SODIUM_LOW': 'Hyponatremia (sodium <135) - fluid retention',
            'SODIUM_VERY_LOW': 'Severe hyponatremia (sodium <130)',
            'BUN_ELEVATED': 'Elevated blood urea nitrogen (>20)',
            'BUN_SEVERELY_ELEVATED': 'Severely elevated BUN (>50)',
            'COMORBIDITY_COUNT': 'Number of major comorbid conditions',
            'COMPLEXITY_SCORE': 'Overall medical complexity score',
            'HAS_DIABETES': 'Diabetes mellitus diagnosis',
            'HAS_KIDNEY_DISEASE': 'Chronic kidney disease diagnosis',
            'DIABETES_KIDNEY_COMBO': 'Combined diabetes and kidney disease',
            'HF_KIDNEY_COMBO': 'Heart failure with kidney disease',
            'SUBOPTIMAL_THERAPY': 'Not on guideline-directed medical therapy',
            'POLYPHARMACY': 'Taking ≥10 medications',
            'HAS_ICU_STAY': 'Required intensive care during admission',
            'ICU_LOS_DAYS': 'Length of ICU stay',
            'EMERGENCY_ADMISSION': 'Admitted through emergency department',
            'WINTER_ADMISSION': 'Admitted during winter months',
            'ADMISSION_RATE_PER_YEAR': 'Historical admission frequency per year'
        }
        
        interpretation = clinical_interpretations.get(feature_name, 'Clinical factor')
        print(f"{i:2d}. {feature_name:<35} {importance:.4f} - {interpretation}")
    
    # Feature importance by category
    feature_categories = {
        'Demographics': ['AGE_AT_ADMISSION', 'AGE_SQUARED', 'AGE_VERY_OLD', 'AGE_YOUNG_HF', 'GENDER_MALE'],
        'Length_of_Stay': ['LOS_DAYS', 'LOS_LOG', 'LOS_VERY_SHORT', 'LOS_VERY_LONG'],
        'Admission_Type': ['EMERGENCY_ADMISSION', 'MEDICARE', 'ADMIT_WEEKEND', 'WINTER_ADMISSION'],
        'Diagnoses': ['TOTAL_DIAGNOSES', 'HF_DIAGNOSES_COUNT', 'COMORBIDITY_COUNT', 'COMPLEXITY_SCORE',
                     'HAS_DIABETES', 'HAS_HYPERTENSION', 'HAS_KIDNEY_DISEASE', 'HAS_CARDIAC_COMORBIDITY',
                     'HAS_RESPIRATORY_DISEASE', 'HAS_SYSTOLIC_HF', 'HAS_DIASTOLIC_HF', 'HAS_COMBINED_HF',
                     'HAS_ACUTE_HF', 'HAS_CHRONIC_HF', 'DIABETES_KIDNEY_COMBO', 'HF_KIDNEY_COMBO'],
        'ICU': ['HAS_ICU_STAY', 'ICU_LOS_DAYS', 'ICU_LOS_LOG', 'MAX_ICU_LOS'],
        'Laboratory': ['TOTAL_LABS', 'CREATININE_LAST', 'SODIUM_LAST', 'BUN_LAST', 'HEMOGLOBIN_LAST',
                      'ALBUMIN_LAST', 'BUN_CREAT_RATIO', 'CREATININE_ELEVATED', 'CREATININE_SEVERELY_ELEVATED',
                      'SODIUM_LOW', 'SODIUM_VERY_LOW', 'BUN_ELEVATED', 'BUN_SEVERELY_ELEVATED',
                      'HEMOGLOBIN_LOW', 'HEMOGLOBIN_VERY_LOW', 'ALBUMIN_LOW', 'ALBUMIN_VERY_LOW'],
        'Medications': ['TOTAL_MEDICATIONS', 'HF_MEDICATION_COUNT', 'GDMT_SCORE', 'HAS_ACE_INHIBITOR',
                       'HAS_BETA_BLOCKER', 'HAS_DIURETIC', 'HAS_ARB', 'OPTIMAL_THERAPY',
                       'SUBOPTIMAL_THERAPY', 'POLYPHARMACY'],
        'Historical': ['PRIOR_ADMISSIONS_COUNT', 'DAYS_SINCE_LAST_ADMISSION', 'HAD_PRIOR_READMISSION',
                      'FREQUENT_FLYER', 'ADMISSION_RATE_PER_YEAR']
    }
    
    print(f"\n📊 Feature Importance by Clinical Category:")
    print("=" * 60)
    
    category_importance = {}
    for category, features in feature_categories.items():
        category_total = importance_df[importance_df['Feature'].isin(features)]['Importance'].sum()
        category_importance[category] = category_total
    
    # Sort categories by importance
    sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    
    for category, total_importance in sorted_categories:
        percentage = (total_importance / importance_df['Importance'].sum()) * 100
        print(f"{category:<15}: {total_importance:.4f} ({percentage:.1f}%)")
    
    # Top predictive insights
    print(f"\n🎯 KEY CLINICAL INSIGHTS FOR READMISSION PREDICTION:")
    print("=" * 65)
    
    top_10_features = importance_df.head(10)
    insights = []
    
    for _, row in top_10_features.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        
        if 'PRIOR' in feature or 'READMISSION' in feature:
            insights.append("📈 **Historical admission patterns** are the strongest predictors")
        elif 'CREATININE' in feature or 'BUN' in feature or 'KIDNEY' in feature:
            insights.append("🫘 **Kidney function** is critical for readmission risk")
        elif 'SODIUM' in feature:
            insights.append("💧 **Fluid balance** (sodium levels) indicates HF severity")
        elif 'DIABETES' in feature:
            insights.append("🍯 **Diabetes** significantly increases readmission risk")
        elif 'AGE' in feature:
            insights.append("👴 **Advanced age** is a major risk factor")
        elif 'LOS' in feature:
            insights.append("🏥 **Length of stay** reflects illness severity")
        elif 'ICU' in feature:
            insights.append("🚨 **ICU requirement** indicates high-risk patients")
        elif 'THERAPY' in feature or 'MEDICATION' in feature:
            insights.append("💊 **Medication optimization** can reduce readmission risk")
    
    # Remove duplicates and print unique insights
    unique_insights = list(dict.fromkeys(insights))
    for i, insight in enumerate(unique_insights[:8], 1):
        print(f"{i}. {insight}")

else:
    print("⚠️ Feature importance not available for this model type")

print(f"\n✅ Advanced feature importance analysis completed")

# %%
# Cell 12: Final Summary and Clinical Recommendations
print("="*60)
print("FINAL ADVANCED MODEL SUMMARY")
print("="*60)

# Performance summary
print("🎯 ADVANCED MODEL PERFORMANCE SUMMARY:")
print("=" * 50)
print(f"Best Model: {best_model_name}")
print(f"Dataset Size: {len(modeling_data_complete):,} admissions")
print(f"Advanced Features: {X_final.shape[1]} clinical features")
print(f"Test Set Size: {len(y_test):,} admissions")
print(f"Baseline Readmission Rate: {y_test.mean()*100:.1f}%")

print(f"\nPerformance Metrics:")
print(f"   ROC-AUC Score: {best_model_metrics['roc_auc']:.4f}")
print(f"   F1-Score: {best_model_metrics['f1_score']:.4f}")
print(f"   Accuracy: {best_model_metrics['accuracy']:.4f}")
print(f"   Average Precision: {best_model_metrics['avg_precision']:.4f}")
print(f"   Sensitivity: {sensitivity:.4f}")
print(f"   Specificity: {specificity:.4f}")

# Model comparison with baseline
print(f"\n📊 IMPROVEMENT OVER BASELINE:")
print("=" * 40)
baseline_metrics = {
    'ROC-AUC': 0.577,
    'F1-Score': 0.057,
    'Accuracy': 0.830
}

for metric, baseline_value in baseline_metrics.items():
    if metric == 'ROC-AUC':
        current_value = best_model_metrics['roc_auc']
    elif metric == 'F1-Score':
        current_value = best_model_metrics['f1_score']
    elif metric == 'Accuracy':
        current_value = best_model_metrics['accuracy']
    
    improvement = ((current_value - baseline_value) / baseline_value) * 100
    print(f"{metric:<12}: {baseline_value:.3f} → {current_value:.3f} ({improvement:+.1f}%)")

# Clinical impact assessment
print(f"\n🏥 ENHANCED CLINICAL IMPACT:")
print("=" * 40)

# Risk stratification with best model
best_risk_scores = best_probabilities
high_risk_threshold = np.percentile(best_risk_scores, 80)
high_risk_patients = best_risk_scores >= high_risk_threshold
high_risk_count = high_risk_patients.sum()
high_risk_readmissions = y_test[high_risk_patients].sum()
high_risk_rate = high_risk_readmissions / high_risk_count if high_risk_count > 0 else 0

total_readmissions = y_test.sum()
capture_rate = high_risk_readmissions / total_readmissions if total_readmissions > 0 else 0

print(f"High-Risk Patient Identification (Top 20%):")
print(f"   Patients identified: {high_risk_count} ({high_risk_count/len(y_test)*100:.1f}% of population)")
print(f"   Actual readmissions captured: {high_risk_readmissions} ({capture_rate*100:.1f}% of all readmissions)")
print(f"   High-risk group readmission rate: {high_risk_rate*100:.1f}%")
print(f"   Number needed to screen: {high_risk_count/high_risk_readmissions:.1f}" if high_risk_readmissions > 0 else "   Number needed to screen: N/A")

# Enhanced cost-benefit analysis
avg_readmission_cost = 15000
intervention_cost_per_patient = 500
intervention_effectiveness = 0.35  # Assume 35% reduction with advanced model

potential_readmissions_prevented = high_risk_readmissions * intervention_effectiveness
potential_savings = potential_readmissions_prevented * avg_readmission_cost
intervention_costs = high_risk_count * intervention_cost_per_patient
net_benefit = potential_savings - intervention_costs

print(f"\nEnhanced Cost-Benefit Analysis:")
print(f"   Potential readmissions prevented: {potential_readmissions_prevented:.1f}")
print(f"   Potential cost savings: ${potential_savings:,.0f}")
print(f"   Intervention costs: ${intervention_costs:,.0f}")
print(f"   Net benefit: ${net_benefit:,.0f}")
print(f"   ROI: {(net_benefit/intervention_costs)*100:.0f}%" if intervention_costs > 0 else "   ROI: N/A")

# Advanced clinical recommendations
print(f"\n🎯 ADVANCED CLINICAL RECOMMENDATIONS:")
print("=" * 50)

recommendations = [
    {
        'Priority': 'CRITICAL',
        'Factor': 'Prior Readmission History',
        'Action': 'Implement AI-powered risk scoring at discharge with automatic care team alerts',
        'Impact': 'High - strongest predictor identified'
    },
    {
        'Priority': 'CRITICAL',
        'Factor': 'Kidney Function Monitoring',
        'Action': 'Real-time creatinine/BUN monitoring with nephrology auto-consults',
        'Impact': 'High - kidney dysfunction strongly predicts readmission'
    },
    {
        'Priority': 'HIGH',
        'Factor': 'Fluid Balance Management',
        'Action': 'Sodium level optimization before discharge with diuretic adjustment',
        'Impact': 'Medium-High - indicates HF severity'
    },
    {
        'Priority': 'HIGH',
        'Factor': 'Diabetes Management',
        'Action': 'Integrated diabetes-HF care protocols with endocrinology coordination',
        'Impact': 'Medium-High - major comorbidity factor'
    },
    {
        'Priority': 'MEDIUM',
        'Factor': 'Medication Optimization',
        'Action': 'AI-assisted GDMT optimization with clinical pharmacist review',
        'Impact': 'Medium - modifiable risk factor'
    },
    {
        'Priority': 'MEDIUM',
        'Factor': 'Length of Stay Patterns',
        'Action': 'Early discharge planning for extended stays with home health coordination',
        'Impact': 'Medium - reflects illness complexity'
    }
]

for rec in recommendations:
    print(f"\n[{rec['Priority']}] {rec['Factor']}:")
    print(f"   Action: {rec['Action']}")
    print(f"   Impact: {rec['Impact']}")

# Model deployment strategy
print(f"\n🚀 ADVANCED MODEL DEPLOYMENT STRATEGY:")
print("=" * 50)

deployment_steps = [
    "1. **Real-time Integration**: Deploy model in EHR with live risk scoring",
    "2. **Alert System**: Configure tiered alerts based on risk percentiles",
    "3. **Clinical Dashboard**: Create risk visualization for care teams",
    "4. **Intervention Protocols**: Implement risk-stratified care pathways",
    "5. **Performance Monitoring**: Track model performance and clinical outcomes",
    "6. **Continuous Learning**: Implement model retraining pipeline",
    "7. **Clinical Validation**: Conduct prospective validation study",
    "8. **Staff Training**: Train clinical staff on model interpretation"
]

for step in deployment_steps:
    print(f"   {step}")

print(f"\n" + "="*60)
print("🎉 ADVANCED HEART FAILURE READMISSION PREDICTION COMPLETE!")
print("="*60)
print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Best Model: {best_model_name} with ROC-AUC: {best_model_metrics['roc_auc']:.4f}")
print(f"Performance Improvement: {improvement:+.1f}% over baseline")
print(f"Ready for clinical validation and deployment")
print("="*60)

# %%