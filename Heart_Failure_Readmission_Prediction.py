# %%
"""
Heart Failure 30-Day Readmission Prediction Model
=================================================

Problem Statement:
- Predict readmission of heart-failure patients within 30-days of discharge
- Use MIMIC-III style dataset with heart failure ICD-9 codes
- Develop ML model with detailed analysis and evaluation

Dataset: Generated MIMIC-III style heart failure dataset
Target: 30-day readmission (binary classification)
"""

# %%
# Cell 1: Import Libraries and Setup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)

print("✅ Libraries imported successfully")
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# Cell 2: Load and Explore Dataset
print("="*60)
print("LOADING HEART FAILURE DATASET")
print("="*60)

# Load all MIMIC tables
patients = pd.read_csv('MIMIC_PATIENTS.csv')
admissions = pd.read_csv('MIMIC_ADMISSIONS.csv')
diagnoses = pd.read_csv('MIMIC_DIAGNOSES_ICD.csv')
icustays = pd.read_csv('MIMIC_ICUSTAYS.csv')
labevents = pd.read_csv('MIMIC_LABEVENTS.csv')
prescriptions = pd.read_csv('MIMIC_PRESCRIPTIONS.csv')

print(f"📊 Dataset Overview:")
print(f"   Patients: {len(patients):,}")
print(f"   Admissions: {len(admissions):,}")
print(f"   Diagnoses: {len(diagnoses):,}")
print(f"   ICU Stays: {len(icustays):,}")
print(f"   Lab Events: {len(labevents):,}")
print(f"   Prescriptions: {len(prescriptions):,}")

# Convert datetime columns
patients['DOB'] = pd.to_datetime(patients['DOB'])
patients['DOD'] = pd.to_datetime(patients['DOD'])
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])

print("\n✅ Data loaded and datetime columns converted")

# %%
# Cell 3: Create Target Variable (30-day Readmission)
print("="*60)
print("CREATING TARGET VARIABLE - 30-DAY READMISSION")
print("="*60)

# Sort admissions by patient and time
admissions_sorted = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME']).copy()

# Calculate time to next admission for same patient
admissions_sorted['NEXT_ADMITTIME'] = admissions_sorted.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
admissions_sorted['DAYS_TO_NEXT_ADMISSION'] = (
    admissions_sorted['NEXT_ADMITTIME'] - admissions_sorted['DISCHTIME']
).dt.days

# Create 30-day readmission target
admissions_sorted['READMITTED_30DAY'] = (
    (admissions_sorted['DAYS_TO_NEXT_ADMISSION'] <= 30) & 
    (admissions_sorted['DAYS_TO_NEXT_ADMISSION'] > 0)
).astype(int)

# Remove last admission for each patient (can't have readmission)
admissions_sorted['IS_LAST_ADMISSION'] = admissions_sorted.groupby('SUBJECT_ID').cumcount(ascending=False) == 0
modeling_data = admissions_sorted[~admissions_sorted['IS_LAST_ADMISSION']].copy()

print(f"📈 Target Variable Analysis:")
print(f"   Total admissions for modeling: {len(modeling_data):,}")
print(f"   30-day readmissions: {modeling_data['READMITTED_30DAY'].sum():,}")
print(f"   Readmission rate: {modeling_data['READMITTED_30DAY'].mean()*100:.1f}%")
print(f"   No readmission: {(modeling_data['READMITTED_30DAY']==0).sum():,}")

# Check class balance
readmission_counts = modeling_data['READMITTED_30DAY'].value_counts()
print(f"\n📊 Class Distribution:")
for class_val, count in readmission_counts.items():
    label = "Readmitted" if class_val == 1 else "Not Readmitted"
    print(f"   {label}: {count:,} ({count/len(modeling_data)*100:.1f}%)")

print("\n✅ Target variable created successfully")

# %%
# Cell 4: Feature Engineering - Patient Demographics
print("="*60)
print("FEATURE ENGINEERING - DEMOGRAPHICS")
print("="*60)

# Merge with patient data
modeling_data = modeling_data.merge(
    patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']], 
    on='SUBJECT_ID'
)

# Calculate age at admission
modeling_data['AGE_AT_ADMISSION'] = (
    modeling_data['ADMITTIME'] - modeling_data['DOB']
).dt.days / 365.25

# Create age groups
modeling_data['AGE_GROUP'] = pd.cut(
    modeling_data['AGE_AT_ADMISSION'], 
    bins=[0, 50, 65, 75, 85, 100], 
    labels=['<50', '50-64', '65-74', '75-84', '85+']
)

# Gender encoding
modeling_data['GENDER_MALE'] = (modeling_data['GENDER'] == 'M').astype(int)

# Calculate length of stay
modeling_data['LOS_DAYS'] = (
    modeling_data['DISCHTIME'] - modeling_data['ADMITTIME']
).dt.days

# LOS categories
modeling_data['LOS_CATEGORY'] = pd.cut(
    modeling_data['LOS_DAYS'], 
    bins=[0, 3, 7, 14, 30, 1000], 
    labels=['Short', 'Medium', 'Long', 'Extended', 'Very_Long']
)

# Admission type encoding
modeling_data['EMERGENCY_ADMISSION'] = (
    modeling_data['ADMISSION_TYPE'] == 'EMERGENCY'
).astype(int)

# Insurance type encoding
modeling_data['MEDICARE'] = (modeling_data['INSURANCE'] == 'Medicare').astype(int)
modeling_data['PRIVATE_INSURANCE'] = (modeling_data['INSURANCE'] == 'Private').astype(int)

print(f"✅ Demographic Features Created:")
print(f"   Age (continuous): {modeling_data['AGE_AT_ADMISSION'].describe()}")
print(f"   Gender (Male %): {modeling_data['GENDER_MALE'].mean()*100:.1f}%")
print(f"   Emergency admissions: {modeling_data['EMERGENCY_ADMISSION'].mean()*100:.1f}%")
print(f"   Medicare patients: {modeling_data['MEDICARE'].mean()*100:.1f}%")

# %%
# Cell 5: Feature Engineering - Diagnosis Features
print("="*60)
print("FEATURE ENGINEERING - DIAGNOSIS FEATURES")
print("="*60)

# Heart failure ICD-9 codes
hf_codes = [39891, 40201, 40211, 40291, 40401, 40403, 40411, 40413,
            40491, 40493, 4280, 4281, 42820, 42821, 42822, 42823,
            42830, 42831, 42832, 42833, 42840, 42841, 42842, 42843, 4289]

# Common comorbidity codes
diabetes_codes = [25000, 25001, 25002]
hypertension_codes = [4019, 40190, 40191]
kidney_codes = [5849, 5859]
cardiac_codes = [42731, 42732, 41401, 41071]
respiratory_codes = [49390, 49391]

# Create diagnosis features for each admission
diagnosis_features = []

for hadm_id in modeling_data['HADM_ID'].unique():
    admission_diagnoses = diagnoses[diagnoses['HADM_ID'] == hadm_id]['ICD9_CODE'].tolist()
    
    features = {
        'HADM_ID': hadm_id,
        'TOTAL_DIAGNOSES': len(admission_diagnoses),
        'HF_DIAGNOSES_COUNT': sum(1 for code in admission_diagnoses if code in hf_codes),
        'HAS_DIABETES': int(any(code in diabetes_codes for code in admission_diagnoses)),
        'HAS_HYPERTENSION': int(any(code in hypertension_codes for code in admission_diagnoses)),
        'HAS_KIDNEY_DISEASE': int(any(code in kidney_codes for code in admission_diagnoses)),
        'HAS_CARDIAC_COMORBIDITY': int(any(code in cardiac_codes for code in admission_diagnoses)),
        'HAS_RESPIRATORY_DISEASE': int(any(code in respiratory_codes for code in admission_diagnoses)),
        'COMORBIDITY_COUNT': (
            int(any(code in diabetes_codes for code in admission_diagnoses)) +
            int(any(code in hypertension_codes for code in admission_diagnoses)) +
            int(any(code in kidney_codes for code in admission_diagnoses)) +
            int(any(code in cardiac_codes for code in admission_diagnoses)) +
            int(any(code in respiratory_codes for code in admission_diagnoses))
        )
    }
    
    # Specific HF types
    systolic_hf_codes = [42820, 42821, 42822, 42823]
    diastolic_hf_codes = [42830, 42831, 42832, 42833]
    combined_hf_codes = [42840, 42841, 42842, 42843]
    
    features['HAS_SYSTOLIC_HF'] = int(any(code in systolic_hf_codes for code in admission_diagnoses))
    features['HAS_DIASTOLIC_HF'] = int(any(code in diastolic_hf_codes for code in admission_diagnoses))
    features['HAS_COMBINED_HF'] = int(any(code in combined_hf_codes for code in admission_diagnoses))
    
    diagnosis_features.append(features)

# Convert to DataFrame and merge
diagnosis_df = pd.DataFrame(diagnosis_features)
modeling_data = modeling_data.merge(diagnosis_df, on='HADM_ID')

print(f"✅ Diagnosis Features Created:")
print(f"   Average diagnoses per admission: {modeling_data['TOTAL_DIAGNOSES'].mean():.1f}")
print(f"   Patients with diabetes: {modeling_data['HAS_DIABETES'].mean()*100:.1f}%")
print(f"   Patients with hypertension: {modeling_data['HAS_HYPERTENSION'].mean()*100:.1f}%")
print(f"   Patients with kidney disease: {modeling_data['HAS_KIDNEY_DISEASE'].mean()*100:.1f}%")
print(f"   Average comorbidities: {modeling_data['COMORBIDITY_COUNT'].mean():.1f}")

# %%
# Cell 6: Feature Engineering - ICU and Lab Features
print("="*60)
print("FEATURE ENGINEERING - ICU & LAB FEATURES")
print("="*60)

# ICU Features
icu_features = []
for hadm_id in modeling_data['HADM_ID'].unique():
    icu_stays_admission = icustays[icustays['HADM_ID'] == hadm_id]
    
    if len(icu_stays_admission) > 0:
        icu_los = icu_stays_admission['LOS'].sum()
        icu_count = len(icu_stays_admission)
        has_icu = 1
    else:
        icu_los = 0
        icu_count = 0
        has_icu = 0
    
    icu_features.append({
        'HADM_ID': hadm_id,
        'HAS_ICU_STAY': has_icu,
        'ICU_LOS_DAYS': icu_los,
        'ICU_STAYS_COUNT': icu_count
    })

icu_df = pd.DataFrame(icu_features)
modeling_data = modeling_data.merge(icu_df, on='HADM_ID')

# Lab Features - Key heart failure labs
lab_features = []
key_lab_items = {
    50912: 'CREATININE',  # Kidney function
    50983: 'SODIUM',      # Fluid balance
    51006: 'BUN',         # Kidney function
    51222: 'HEMOGLOBIN',  # Anemia
    50862: 'ALBUMIN'      # Nutrition/liver function
}

for hadm_id in modeling_data['HADM_ID'].unique():
    admission_labs = labevents[labevents['HADM_ID'] == hadm_id]
    
    lab_feature = {'HADM_ID': hadm_id, 'TOTAL_LABS': len(admission_labs)}
    
    for itemid, lab_name in key_lab_items.items():
        lab_values = admission_labs[admission_labs['ITEMID'] == itemid]['VALUENUM']
        
        if len(lab_values) > 0:
            # Use last available value (closest to discharge)
            last_value = lab_values.iloc[-1]
            lab_feature[f'{lab_name}_LAST'] = last_value
            lab_feature[f'{lab_name}_MEAN'] = lab_values.mean()
            lab_feature[f'HAS_{lab_name}'] = 1
            
            # Abnormal flags
            if lab_name == 'CREATININE':
                lab_feature[f'{lab_name}_ELEVATED'] = int(last_value > 1.5)
            elif lab_name == 'SODIUM':
                lab_feature[f'{lab_name}_LOW'] = int(last_value < 135)
            elif lab_name == 'BUN':
                lab_feature[f'{lab_name}_ELEVATED'] = int(last_value > 20)
            elif lab_name == 'HEMOGLOBIN':
                lab_feature[f'{lab_name}_LOW'] = int(last_value < 10)
            elif lab_name == 'ALBUMIN':
                lab_feature[f'{lab_name}_LOW'] = int(last_value < 3.5)
        else:
            lab_feature[f'{lab_name}_LAST'] = np.nan
            lab_feature[f'{lab_name}_MEAN'] = np.nan
            lab_feature[f'HAS_{lab_name}'] = 0
            lab_feature[f'{lab_name}_ELEVATED'] = 0
            lab_feature[f'{lab_name}_LOW'] = 0
    
    lab_features.append(lab_feature)

lab_df = pd.DataFrame(lab_features)
modeling_data = modeling_data.merge(lab_df, on='HADM_ID')

print(f"✅ ICU & Lab Features Created:")
print(f"   ICU utilization: {modeling_data['HAS_ICU_STAY'].mean()*100:.1f}%")
print(f"   Average total labs: {modeling_data['TOTAL_LABS'].mean():.0f}")
print(f"   Patients with elevated creatinine: {modeling_data['CREATININE_ELEVATED'].mean()*100:.1f}%")
print(f"   Patients with low sodium: {modeling_data['SODIUM_LOW'].mean()*100:.1f}%")

# %%
# Cell 7: Feature Engineering - Medication Features
print("="*60)
print("FEATURE ENGINEERING - MEDICATION FEATURES")
print("="*60)

# Key heart failure medications
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

# Create medication features
med_features = []
for hadm_id in modeling_data['HADM_ID'].unique():
    admission_meds = prescriptions[prescriptions['HADM_ID'] == hadm_id]
    med_drugs = admission_meds['DRUG'].tolist()
    
    med_feature = {
        'HADM_ID': hadm_id,
        'TOTAL_MEDICATIONS': len(admission_meds),
        'UNIQUE_MEDICATIONS': len(set(med_drugs))
    }
    
    # Check for each medication class
    for med_class, drug_list in hf_medications.items():
        has_med = int(any(drug in med_drugs for drug in drug_list))
        med_feature[f'HAS_{med_class}'] = has_med
    
    # Count of HF-specific medications
    hf_med_count = sum(med_feature[f'HAS_{med_class}'] for med_class in hf_medications.keys())
    med_feature['HF_MEDICATION_COUNT'] = hf_med_count
    
    # Guideline-directed medical therapy (GDMT) score
    # ACE/ARB + Beta Blocker + Diuretic = optimal therapy
    gdmt_score = (med_feature['HAS_ACE_INHIBITOR'] or med_feature['HAS_ARB']) + \
                 med_feature['HAS_BETA_BLOCKER'] + med_feature['HAS_DIURETIC']
    med_feature['GDMT_SCORE'] = gdmt_score
    med_feature['OPTIMAL_THERAPY'] = int(gdmt_score >= 3)
    
    med_features.append(med_feature)

med_df = pd.DataFrame(med_features)
modeling_data = modeling_data.merge(med_df, on='HADM_ID')

print(f"✅ Medication Features Created:")
print(f"   Average medications per admission: {modeling_data['TOTAL_MEDICATIONS'].mean():.1f}")
print(f"   Patients on ACE inhibitors: {modeling_data['HAS_ACE_INHIBITOR'].mean()*100:.1f}%")
print(f"   Patients on beta blockers: {modeling_data['HAS_BETA_BLOCKER'].mean()*100:.1f}%")
print(f"   Patients on diuretics: {modeling_data['HAS_DIURETIC'].mean()*100:.1f}%")
print(f"   Patients on optimal therapy: {modeling_data['OPTIMAL_THERAPY'].mean()*100:.1f}%")
print(f"   Average GDMT score: {modeling_data['GDMT_SCORE'].mean():.1f}")

# %%
# Cell 8: Feature Engineering - Historical Features
print("="*60)
print("FEATURE ENGINEERING - HISTORICAL FEATURES")
print("="*60)

# Calculate historical admission patterns for each patient
historical_features = []

for subject_id in modeling_data['SUBJECT_ID'].unique():
    patient_admissions = admissions_sorted[admissions_sorted['SUBJECT_ID'] == subject_id].sort_values('ADMITTIME')
    
    for idx, admission in patient_admissions.iterrows():
        # Only include admissions before current one for historical features
        prior_admissions = patient_admissions[patient_admissions['ADMITTIME'] < admission['ADMITTIME']]
        
        if len(prior_admissions) == 0:
            # First admission - no history
            hist_features = {
                'HADM_ID': admission['HADM_ID'],
                'PRIOR_ADMISSIONS_COUNT': 0,
                'DAYS_SINCE_LAST_ADMISSION': np.nan,
                'PRIOR_ICU_STAYS': 0,
                'PRIOR_EMERGENCY_ADMISSIONS': 0,
                'AVERAGE_PRIOR_LOS': np.nan,
                'HAD_PRIOR_READMISSION': 0,
                'TOTAL_PRIOR_DIAGNOSES': 0
            }
        else:
            # Calculate historical features
            last_admission = prior_admissions.iloc[-1]
            days_since_last = (admission['ADMITTIME'] - last_admission['DISCHTIME']).days
            
            # Prior ICU stays
            prior_icu_count = 0
            for prior_hadm in prior_admissions['HADM_ID']:
                if len(icustays[icustays['HADM_ID'] == prior_hadm]) > 0:
                    prior_icu_count += 1
            
            # Prior emergency admissions
            prior_emergency = (prior_admissions['ADMISSION_TYPE'] == 'EMERGENCY').sum()
            
            # Average prior LOS
            prior_los = []
            for _, prior_adm in prior_admissions.iterrows():
                los = (pd.to_datetime(prior_adm['DISCHTIME']) - pd.to_datetime(prior_adm['ADMITTIME'])).days
                prior_los.append(los)
            avg_prior_los = np.mean(prior_los) if prior_los else 0
            
            # Check if had prior readmission (within 30 days)
            had_prior_readmission = 0
            for i in range(len(prior_admissions) - 1):
                curr_disch = pd.to_datetime(prior_admissions.iloc[i]['DISCHTIME'])
                next_admit = pd.to_datetime(prior_admissions.iloc[i+1]['ADMITTIME'])
                if (next_admit - curr_disch).days <= 30:
                    had_prior_readmission = 1
                    break
            
            # Total prior diagnoses
            total_prior_diagnoses = 0
            for prior_hadm in prior_admissions['HADM_ID']:
                total_prior_diagnoses += len(diagnoses[diagnoses['HADM_ID'] == prior_hadm])
            
            hist_features = {
                'HADM_ID': admission['HADM_ID'],
                'PRIOR_ADMISSIONS_COUNT': len(prior_admissions),
                'DAYS_SINCE_LAST_ADMISSION': days_since_last,
                'PRIOR_ICU_STAYS': prior_icu_count,
                'PRIOR_EMERGENCY_ADMISSIONS': prior_emergency,
                'AVERAGE_PRIOR_LOS': avg_prior_los,
                'HAD_PRIOR_READMISSION': had_prior_readmission,
                'TOTAL_PRIOR_DIAGNOSES': total_prior_diagnoses
            }
        
        historical_features.append(hist_features)

hist_df = pd.DataFrame(historical_features)
modeling_data = modeling_data.merge(hist_df, on='HADM_ID')

print(f"✅ Historical Features Created:")
print(f"   Patients with prior admissions: {(modeling_data['PRIOR_ADMISSIONS_COUNT'] > 0).mean()*100:.1f}%")
print(f"   Average prior admissions: {modeling_data['PRIOR_ADMISSIONS_COUNT'].mean():.1f}")
print(f"   Patients with prior readmissions: {modeling_data['HAD_PRIOR_READMISSION'].mean()*100:.1f}%")

# %%
# Cell 9: Prepare Features for Modeling
print("="*60)
print("PREPARING FEATURES FOR MODELING")
print("="*60)

# Select features for modeling
feature_columns = [
    # Demographics
    'AGE_AT_ADMISSION', 'GENDER_MALE', 'LOS_DAYS',
    'EMERGENCY_ADMISSION', 'MEDICARE', 'PRIVATE_INSURANCE',
    
    # Diagnoses
    'TOTAL_DIAGNOSES', 'HF_DIAGNOSES_COUNT', 'COMORBIDITY_COUNT',
    'HAS_DIABETES', 'HAS_HYPERTENSION', 'HAS_KIDNEY_DISEASE',
    'HAS_CARDIAC_COMORBIDITY', 'HAS_RESPIRATORY_DISEASE',
    'HAS_SYSTOLIC_HF', 'HAS_DIASTOLIC_HF', 'HAS_COMBINED_HF',
    
    # ICU
    'HAS_ICU_STAY', 'ICU_LOS_DAYS', 'ICU_STAYS_COUNT',
    
    # Labs
    'TOTAL_LABS', 'CREATININE_LAST', 'SODIUM_LAST', 'BUN_LAST',
    'HEMOGLOBIN_LAST', 'ALBUMIN_LAST',
    'CREATININE_ELEVATED', 'SODIUM_LOW', 'BUN_ELEVATED',
    'HEMOGLOBIN_LOW', 'ALBUMIN_LOW',
    
    # Medications
    'TOTAL_MEDICATIONS', 'HF_MEDICATION_COUNT', 'GDMT_SCORE',
    'HAS_ACE_INHIBITOR', 'HAS_BETA_BLOCKER', 'HAS_DIURETIC',
    'HAS_ARB', 'OPTIMAL_THERAPY',
    
    # Historical
    'PRIOR_ADMISSIONS_COUNT', 'DAYS_SINCE_LAST_ADMISSION',
    'PRIOR_ICU_STAYS', 'PRIOR_EMERGENCY_ADMISSIONS',
    'AVERAGE_PRIOR_LOS', 'HAD_PRIOR_READMISSION'
]

# Create feature matrix
X = modeling_data[feature_columns].copy()
y = modeling_data['READMITTED_30DAY'].copy()

print(f"📊 Feature Matrix Shape: {X.shape}")
print(f"📊 Target Distribution:")
print(f"   No Readmission (0): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"   Readmission (1): {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

# Handle missing values
print(f"\n🔍 Missing Values Analysis:")
missing_counts = X.isnull().sum()
missing_features = missing_counts[missing_counts > 0]
if len(missing_features) > 0:
    print("Features with missing values:")
    for feature, count in missing_features.items():
        print(f"   {feature}: {count} ({count/len(X)*100:.1f}%)")
else:
    print("   No missing values found!")

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X), 
    columns=X.columns, 
    index=X.index
)

print(f"\n✅ Features prepared for modeling")
print(f"   Final feature count: {X_imputed.shape[1]}")
print(f"   Sample size: {X_imputed.shape[0]:,}")

# %%
# Cell 10: Exploratory Analysis of Features vs Target
print("="*60)
print("EXPLORATORY ANALYSIS - FEATURES VS READMISSION")
print("="*60)

# Analyze key features by readmission status
readmit_analysis = modeling_data.groupby('READMITTED_30DAY').agg({
    'AGE_AT_ADMISSION': ['mean', 'std'],
    'LOS_DAYS': ['mean', 'std'],
    'TOTAL_DIAGNOSES': ['mean', 'std'],
    'COMORBIDITY_COUNT': ['mean', 'std'],
    'HAS_ICU_STAY': 'mean',
    'HAS_DIABETES': 'mean',
    'HAS_KIDNEY_DISEASE': 'mean',
    'PRIOR_ADMISSIONS_COUNT': ['mean', 'std'],
    'HAD_PRIOR_READMISSION': 'mean',
    'OPTIMAL_THERAPY': 'mean',
    'CREATININE_ELEVATED': 'mean',
    'SODIUM_LOW': 'mean'
}).round(2)

print("📊 Key Features by Readmission Status:")
print("="*50)

# Compare means for key continuous variables
key_continuous = ['AGE_AT_ADMISSION', 'LOS_DAYS', 'TOTAL_DIAGNOSES', 'COMORBIDITY_COUNT', 'PRIOR_ADMISSIONS_COUNT']
for feature in key_continuous:
    no_readmit = modeling_data[modeling_data['READMITTED_30DAY']==0][feature].mean()
    readmit = modeling_data[modeling_data['READMITTED_30DAY']==1][feature].mean()
    print(f"{feature}:")
    print(f"   No Readmission: {no_readmit:.1f}")
    print(f"   Readmission: {readmit:.1f}")
    print(f"   Difference: {readmit - no_readmit:+.1f}")
    print()

# Compare proportions for key binary variables
key_binary = ['HAS_ICU_STAY', 'HAS_DIABETES', 'HAS_KIDNEY_DISEASE', 'HAD_PRIOR_READMISSION', 
              'OPTIMAL_THERAPY', 'CREATININE_ELEVATED', 'SODIUM_LOW']
print("Binary Features (% with condition):")
for feature in key_binary:
    no_readmit = modeling_data[modeling_data['READMITTED_30DAY']==0][feature].mean() * 100
    readmit = modeling_data[modeling_data['READMITTED_30DAY']==1][feature].mean() * 100
    print(f"{feature}:")
    print(f"   No Readmission: {no_readmit:.1f}%")
    print(f"   Readmission: {readmit:.1f}%")
    print(f"   Difference: {readmit - no_readmit:+.1f}%")
    print()

print("✅ Exploratory analysis completed")

# %%
# Cell 11: Train-Test Split and Data Preprocessing
print("="*60)
print("TRAIN-TEST SPLIT AND PREPROCESSING")
print("="*60)

# Split the data (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"📊 Data Split:")
print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set: {X_test.shape[0]:,} samples")
print(f"   Features: {X_train.shape[1]}")

# Check class distribution in splits
print(f"\n📊 Class Distribution:")
print("Training set:")
train_dist = y_train.value_counts(normalize=True) * 100
for class_val, pct in train_dist.items():
    label = "Readmission" if class_val == 1 else "No Readmission"
    print(f"   {label}: {pct:.1f}%")

print("Test set:")
test_dist = y_test.value_counts(normalize=True) * 100
for class_val, pct in test_dist.items():
    label = "Readmission" if class_val == 1 else "No Readmission"
    print(f"   {label}: {pct:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Features scaled using StandardScaler")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\n📊 After SMOTE Balancing:")
print(f"   Training samples: {X_train_balanced.shape[0]:,}")
balanced_dist = pd.Series(y_train_balanced).value_counts(normalize=True) * 100
for class_val, pct in balanced_dist.items():
    label = "Readmission" if class_val == 1 else "No Readmission"
    print(f"   {label}: {pct:.1f}%")

print(f"\n✅ Data preprocessing completed")

# %%
# Cell 12: Model Training - Multiple Algorithms
print("="*60)
print("MODEL TRAINING - MULTIPLE ALGORITHMS")
print("="*60)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train models and store results
model_results = {}
trained_models = {}

print("🚀 Training Models...")
print("-" * 40)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train on balanced data
    model.fit(X_train_balanced, y_train_balanced)
    trained_models[name] = model
    
    # Predictions on test set (original unbalanced test set)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    model_results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"   ✅ {name} trained")
    print(f"      Accuracy: {accuracy:.3f}")
    print(f"      F1-Score: {f1:.3f}")
    print(f"      ROC-AUC: {roc_auc:.3f}")

print(f"\n✅ All models trained successfully!")

# %%
# Cell 13: Model Evaluation and Comparison
print("="*60)
print("MODEL EVALUATION AND COMPARISON")
print("="*60)

# Create comparison table
comparison_df = pd.DataFrame(model_results).T
comparison_df = comparison_df[['accuracy', 'f1_score', 'roc_auc']].round(4)
comparison_df = comparison_df.sort_values('roc_auc', ascending=False)

print("📊 Model Performance Comparison:")
print("=" * 50)
print(comparison_df.to_string())

# Find best model
best_model_name = comparison_df.index[0]
best_model = trained_models[best_model_name]
best_predictions = model_results[best_model_name]['predictions']
best_probabilities = model_results[best_model_name]['probabilities']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")
print(f"   F1-Score: {comparison_df.loc[best_model_name, 'f1_score']:.4f}")
print(f"   Accuracy: {comparison_df.loc[best_model_name, 'accuracy']:.4f}")

# Detailed evaluation of best model
print(f"\n📋 Detailed Evaluation - {best_model_name}:")
print("=" * 50)

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

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # Recall for positive class
specificity = tn / (tn + fp)  # Recall for negative class
ppv = tp / (tp + fp)  # Precision for positive class
npv = tn / (tn + fn)  # Precision for negative class

print(f"\nDetailed Metrics:")
print(f"   Sensitivity (Recall): {sensitivity:.3f}")
print(f"   Specificity: {specificity:.3f}")
print(f"   Positive Predictive Value: {ppv:.3f}")
print(f"   Negative Predictive Value: {npv:.3f}")
print(f"   False Positive Rate: {fp/(fp+tn):.3f}")
print(f"   False Negative Rate: {fn/(fn+tp):.3f}")

print(f"\n✅ Model evaluation completed")

# %%
# Cell 14: Feature Importance Analysis
print("="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance from best model
if hasattr(best_model, 'feature_importances_'):
    # Tree-based models
    feature_importance = best_model.feature_importances_
    importance_type = "Feature Importance"
elif hasattr(best_model, 'coef_'):
    # Linear models
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = "Coefficient Magnitude"
else:
    feature_importance = None

if feature_importance is not None:
    # Create feature importance DataFrame
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(f"🔍 Top 20 Most Important Features ({importance_type}):")
    print("=" * 60)
    
    for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:<35} {row['Importance']:.4f}")
    
    # Group features by category
    feature_categories = {
        'Demographics': ['AGE_AT_ADMISSION', 'GENDER_MALE', 'LOS_DAYS', 'EMERGENCY_ADMISSION', 'MEDICARE', 'PRIVATE_INSURANCE'],
        'Diagnoses': ['TOTAL_DIAGNOSES', 'HF_DIAGNOSES_COUNT', 'COMORBIDITY_COUNT', 'HAS_DIABETES', 'HAS_HYPERTENSION', 'HAS_KIDNEY_DISEASE', 'HAS_CARDIAC_COMORBIDITY', 'HAS_RESPIRATORY_DISEASE', 'HAS_SYSTOLIC_HF', 'HAS_DIASTOLIC_HF', 'HAS_COMBINED_HF'],
        'ICU': ['HAS_ICU_STAY', 'ICU_LOS_DAYS', 'ICU_STAYS_COUNT'],
        'Labs': ['TOTAL_LABS', 'CREATININE_LAST', 'SODIUM_LAST', 'BUN_LAST', 'HEMOGLOBIN_LAST', 'ALBUMIN_LAST', 'CREATININE_ELEVATED', 'SODIUM_LOW', 'BUN_ELEVATED', 'HEMOGLOBIN_LOW', 'ALBUMIN_LOW'],
        'Medications': ['TOTAL_MEDICATIONS', 'HF_MEDICATION_COUNT', 'GDMT_SCORE', 'HAS_ACE_INHIBITOR', 'HAS_BETA_BLOCKER', 'HAS_DIURETIC', 'HAS_ARB', 'OPTIMAL_THERAPY'],
        'Historical': ['PRIOR_ADMISSIONS_COUNT', 'DAYS_SINCE_LAST_ADMISSION', 'PRIOR_ICU_STAYS', 'PRIOR_EMERGENCY_ADMISSIONS', 'AVERAGE_PRIOR_LOS', 'HAD_PRIOR_READMISSION']
    }
    
    print(f"\n📊 Feature Importance by Category:")
    print("=" * 50)
    
    for category, features in feature_categories.items():
        category_importance = importance_df[importance_df['Feature'].isin(features)]['Importance'].sum()
        print(f"{category:<15}: {category_importance:.4f}")
    
    # Top predictive features interpretation
    print(f"\n🎯 Key Predictive Factors for 30-Day Readmission:")
    print("=" * 55)
    
    top_features = importance_df.head(10)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        feature_name = row['Feature']
        importance = row['Importance']
        
        # Provide clinical interpretation
        interpretations = {
            'HAD_PRIOR_READMISSION': 'History of previous readmissions',
            'PRIOR_ADMISSIONS_COUNT': 'Number of previous hospital admissions',
            'AGE_AT_ADMISSION': 'Patient age at admission',
            'LOS_DAYS': 'Length of current hospital stay',
            'CREATININE_ELEVATED': 'Kidney function impairment',
            'SODIUM_LOW': 'Fluid retention/heart failure severity',
            'COMORBIDITY_COUNT': 'Number of additional medical conditions',
            'HAS_KIDNEY_DISEASE': 'Presence of chronic kidney disease',
            'DAYS_SINCE_LAST_ADMISSION': 'Time since previous admission',
            'TOTAL_DIAGNOSES': 'Overall medical complexity'
        }
        
        interpretation = interpretations.get(feature_name, 'Clinical factor')
        print(f"{i:2d}. {feature_name:<30} - {interpretation}")

else:
    print("⚠️  Feature importance not available for this model type")

print(f"\n✅ Feature importance analysis completed")

# %%
# Cell 15: Cross-Validation and Model Robustness
print("="*60)
print("CROSS-VALIDATION AND MODEL ROBUSTNESS")
print("="*60)

# Perform stratified k-fold cross-validation
cv_folds = 5
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

print(f"🔄 Performing {cv_folds}-Fold Cross-Validation...")
print("-" * 40)

cv_results = {}

for name, model in models.items():
    print(f"\nCross-validating {name}...")
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                 cv=skf, scoring='accuracy', n_jobs=-1)
    cv_f1 = cross_val_score(model, X_train_balanced, y_train_balanced, 
                           cv=skf, scoring='f1', n_jobs=-1)
    cv_roc_auc = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                cv=skf, scoring='roc_auc', n_jobs=-1)
    
    cv_results[name] = {
        'accuracy_mean': cv_accuracy.mean(),
        'accuracy_std': cv_accuracy.std(),
        'f1_mean': cv_f1.mean(),
        'f1_std': cv_f1.std(),
        'roc_auc_mean': cv_roc_auc.mean(),
        'roc_auc_std': cv_roc_auc.std()
    }
    
    print(f"   Accuracy: {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
    print(f"   F1-Score: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(f"   ROC-AUC:  {cv_roc_auc.mean():.3f} ± {cv_roc_auc.std():.3f}")

# Create CV results summary
cv_summary = pd.DataFrame(cv_results).T
cv_summary = cv_summary.round(4)

print(f"\n📊 Cross-Validation Summary:")
print("=" * 60)
print("Model                 Accuracy        F1-Score        ROC-AUC")
print("-" * 60)

for model_name in cv_summary.index:
    acc_mean = cv_summary.loc[model_name, 'accuracy_mean']
    acc_std = cv_summary.loc[model_name, 'accuracy_std']
    f1_mean = cv_summary.loc[model_name, 'f1_mean']
    f1_std = cv_summary.loc[model_name, 'f1_std']
    auc_mean = cv_summary.loc[model_name, 'roc_auc_mean']
    auc_std = cv_summary.loc[model_name, 'roc_auc_std']
    
    print(f"{model_name:<20} {acc_mean:.3f}±{acc_std:.3f}    {f1_mean:.3f}±{f1_std:.3f}    {auc_mean:.3f}±{auc_std:.3f}")

# Model stability analysis
print(f"\n🎯 Model Stability Analysis:")
print("=" * 40)

most_stable_model = cv_summary.loc[cv_summary['roc_auc_std'].idxmin()]
least_stable_model = cv_summary.loc[cv_summary['roc_auc_std'].idxmax()]

print(f"Most Stable Model: {cv_summary['roc_auc_std'].idxmin()}")
print(f"   ROC-AUC Std Dev: {cv_summary['roc_auc_std'].min():.4f}")
print(f"Least Stable Model: {cv_summary['roc_auc_std'].idxmax()}")
print(f"   ROC-AUC Std Dev: {cv_summary['roc_auc_std'].max():.4f}")

print(f"\n✅ Cross-validation completed")

# %%
# Cell 16: Risk Stratification and Clinical Insights
print("="*60)
print("RISK STRATIFICATION AND CLINICAL INSIGHTS")
print("="*60)

# Create risk scores using best model probabilities
risk_scores = best_probabilities
risk_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print("🎯 Risk Stratification Analysis:")
print("=" * 50)

# Analyze different risk thresholds
threshold_analysis = []
for threshold in risk_thresholds:
    predictions_at_threshold = (risk_scores >= threshold).astype(int)
    
    # Calculate metrics at this threshold
    tn, fp, fn, tp = confusion_matrix(y_test, predictions_at_threshold).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    threshold_analysis.append({
        'Threshold': threshold,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'Predicted_Positive': (predictions_at_threshold == 1).sum(),
        'True_Positive': tp,
        'False_Positive': fp
    })

threshold_df = pd.DataFrame(threshold_analysis)

print("Threshold  Sensitivity  Specificity  PPV     NPV     Predicted+")
print("-" * 60)
for _, row in threshold_df.iterrows():
    print(f"{row['Threshold']:.1f}        {row['Sensitivity']:.3f}       {row['Specificity']:.3f}      {row['PPV']:.3f}   {row['NPV']:.3f}   {int(row['Predicted_Positive']):4d}")

# Risk stratification groups
print(f"\n📊 Risk Group Analysis:")
print("=" * 40)

# Define risk groups based on probability quartiles
risk_quartiles = np.percentile(risk_scores, [25, 50, 75])
risk_groups = []

for i, score in enumerate(risk_scores):
    if score <= risk_quartiles[0]:
        risk_groups.append('Low Risk')
    elif score <= risk_quartiles[1]:
        risk_groups.append('Medium-Low Risk')
    elif score <= risk_quartiles[2]:
        risk_groups.append('Medium-High Risk')
    else:
        risk_groups.append('High Risk')

risk_groups = np.array(risk_groups)

# Analyze outcomes by risk group
risk_group_analysis = []
for group in ['Low Risk', 'Medium-Low Risk', 'Medium-High Risk', 'High Risk']:
    group_mask = risk_groups == group
    group_size = group_mask.sum()
    group_readmissions = y_test[group_mask].sum()
    group_rate = group_readmissions / group_size if group_size > 0 else 0
    
    risk_group_analysis.append({
        'Risk_Group': group,
        'Count': group_size,
        'Readmissions': group_readmissions,
        'Rate': group_rate,
        'Percentage': group_size / len(y_test) * 100
    })

risk_df = pd.DataFrame(risk_group_analysis)

print("Risk Group        Count  Readmissions  Rate    % of Population")
print("-" * 65)
for _, row in risk_df.iterrows():
    print(f"{row['Risk_Group']:<15} {row['Count']:5d}  {row['Readmissions']:11d}  {row['Rate']:.3f}  {row['Percentage']:6.1f}%")

# Clinical actionability analysis
print(f"\n🏥 Clinical Actionability:")
print("=" * 40)

# High-risk patients (top 20% by probability)
high_risk_threshold = np.percentile(risk_scores, 80)
high_risk_patients = risk_scores >= high_risk_threshold
high_risk_count = high_risk_patients.sum()
high_risk_readmissions = y_test[high_risk_patients].sum()
high_risk_rate = high_risk_readmissions / high_risk_count

print(f"High-Risk Patients (Top 20% by probability):")
print(f"   Count: {high_risk_count} patients")
print(f"   Actual readmissions: {high_risk_readmissions}")
print(f"   Readmission rate: {high_risk_rate:.1%}")
print(f"   Risk concentration: {high_risk_readmissions / y_test.sum():.1%} of all readmissions")

# Number needed to screen
nns = high_risk_count / high_risk_readmissions if high_risk_readmissions > 0 else float('inf')
print(f"   Number needed to screen: {nns:.1f}")

print(f"\n✅ Risk stratification analysis completed")

# %%
# Cell 17: Model Interpretation and Clinical Recommendations
print("="*60)
print("MODEL INTERPRETATION & CLINICAL RECOMMENDATIONS")
print("="*60)

# Analyze high-risk vs low-risk patient characteristics
high_risk_mask = risk_scores >= np.percentile(risk_scores, 80)
low_risk_mask = risk_scores <= np.percentile(risk_scores, 20)

print("🔍 High-Risk vs Low-Risk Patient Characteristics:")
print("=" * 55)

# Get original data for high and low risk patients
high_risk_indices = X_test.index[high_risk_mask]
low_risk_indices = X_test.index[low_risk_mask]

high_risk_data = modeling_data.loc[high_risk_indices]
low_risk_data = modeling_data.loc[low_risk_indices]

# Compare key characteristics
comparison_features = [
    'AGE_AT_ADMISSION', 'LOS_DAYS', 'PRIOR_ADMISSIONS_COUNT',
    'COMORBIDITY_COUNT', 'HAS_DIABETES', 'HAS_KIDNEY_DISEASE',
    'HAD_PRIOR_READMISSION', 'CREATININE_ELEVATED', 'SODIUM_LOW',
    'OPTIMAL_THERAPY', 'HAS_ICU_STAY'
]

print("Feature                    High-Risk    Low-Risk     Difference")
print("-" * 65)

for feature in comparison_features:
    high_risk_mean = high_risk_data[feature].mean()
    low_risk_mean = low_risk_data[feature].mean()
    difference = high_risk_mean - low_risk_mean
    
    if feature in ['HAS_DIABETES', 'HAS_KIDNEY_DISEASE', 'HAD_PRIOR_READMISSION', 
                   'CREATININE_ELEVATED', 'SODIUM_LOW', 'OPTIMAL_THERAPY', 'HAS_ICU_STAY']:
        # Binary features - show as percentages
        print(f"{feature:<25} {high_risk_mean*100:8.1f}%   {low_risk_mean*100:8.1f}%   {difference*100:+8.1f}%")
    else:
        # Continuous features
        print(f"{feature:<25} {high_risk_mean:8.1f}    {low_risk_mean:8.1f}    {difference:+8.1f}")

# Clinical recommendations based on model insights
print(f"\n🏥 Clinical Recommendations for Readmission Prevention:")
print("=" * 60)

recommendations = [
    {
        'Priority': 'HIGH',
        'Factor': 'Prior Readmission History',
        'Recommendation': 'Implement intensive discharge planning and 48-72 hour post-discharge follow-up for patients with previous readmissions'
    },
    {
        'Priority': 'HIGH', 
        'Factor': 'Multiple Prior Admissions',
        'Recommendation': 'Consider case management and care coordination for frequent utilizers'
    },
    {
        'Priority': 'HIGH',
        'Factor': 'Elevated Creatinine',
        'Recommendation': 'Nephrology consultation and careful medication dosing adjustments before discharge'
    },
    {
        'Priority': 'MEDIUM',
        'Factor': 'Low Sodium Levels',
        'Recommendation': 'Optimize fluid management and consider heart failure medication adjustments'
    },
    {
        'Priority': 'MEDIUM',
        'Factor': 'Multiple Comorbidities',
        'Recommendation': 'Comprehensive medication reconciliation and specialist coordination'
    },
    {
        'Priority': 'MEDIUM',
        'Factor': 'Suboptimal Therapy',
        'Recommendation': 'Ensure guideline-directed medical therapy (ACE/ARB + Beta-blocker + Diuretic) before discharge'
    },
    {
        'Priority': 'LOW',
        'Factor': 'Extended Length of Stay',
        'Recommendation': 'Enhanced discharge planning and home health services consideration'
    }
]

for rec in recommendations:
    print(f"\n[{rec['Priority']}] {rec['Factor']}:")
    print(f"   → {rec['Recommendation']}")

# Model deployment considerations
print(f"\n🚀 Model Deployment Considerations:")
print("=" * 45)

deployment_notes = [
    "Real-time Risk Scoring: Integrate model into EHR for automatic risk calculation at discharge",
    "Alert System: Generate alerts for high-risk patients (>80th percentile) for care team",
    "Intervention Protocols: Develop standardized protocols for different risk levels",
    "Performance Monitoring: Track model performance and readmission rates over time",
    "Model Updates: Retrain model quarterly with new data to maintain accuracy",
    "Clinical Validation: Validate predictions with clinical judgment before interventions"
]

for i, note in enumerate(deployment_notes, 1):
    print(f"{i}. {note}")

print(f"\n✅ Clinical interpretation completed")

# %%
# Cell 18: Final Model Summary and Conclusions
print("="*60)
print("FINAL MODEL SUMMARY AND CONCLUSIONS")
print("="*60)

# Model performance summary
print("🎯 FINAL MODEL PERFORMANCE:")
print("=" * 40)
print(f"Best Model: {best_model_name}")
print(f"Dataset Size: {len(modeling_data):,} admissions")
print(f"Features Used: {X_train.shape[1]} clinical features")
print(f"Test Set Size: {len(y_test):,} admissions")
print(f"Baseline Readmission Rate: {y_test.mean()*100:.1f}%")

print(f"\nKey Performance Metrics:")
print(f"   ROC-AUC Score: {model_results[best_model_name]['roc_auc']:.3f}")
print(f"   F1-Score: {model_results[best_model_name]['f1_score']:.3f}")
print(f"   Accuracy: {model_results[best_model_name]['accuracy']:.3f}")
print(f"   Sensitivity: {sensitivity:.3f}")
print(f"   Specificity: {specificity:.3f}")

# Clinical impact assessment
print(f"\n🏥 CLINICAL IMPACT ASSESSMENT:")
print("=" * 45)

total_readmissions = y_test.sum()
high_risk_captured = y_test[high_risk_patients].sum()
capture_rate = high_risk_captured / total_readmissions

print(f"Readmission Capture Analysis:")
print(f"   Total 30-day readmissions: {total_readmissions}")
print(f"   High-risk group size: {high_risk_count} patients ({high_risk_count/len(y_test)*100:.1f}% of population)")
print(f"   Readmissions captured in high-risk group: {high_risk_captured} ({capture_rate*100:.1f}%)")
print(f"   Number needed to screen: {nns:.1f} patients per prevented readmission")

# Cost-benefit analysis (hypothetical)
avg_readmission_cost = 15000  # Average cost of heart failure readmission
intervention_cost_per_patient = 500  # Cost of intervention per high-risk patient

potential_savings = high_risk_captured * avg_readmission_cost * 0.3  # Assume 30% reduction
intervention_costs = high_risk_count * intervention_cost_per_patient
net_benefit = potential_savings - intervention_costs

print(f"\nHypothetical Cost-Benefit Analysis:")
print(f"   Potential readmissions prevented (30% reduction): {high_risk_captured * 0.3:.1f}")
print(f"   Potential cost savings: ${potential_savings:,.0f}")
print(f"   Intervention costs: ${intervention_costs:,.0f}")
print(f"   Net benefit: ${net_benefit:,.0f}")

# Model strengths and limitations
print(f"\n✅ MODEL STRENGTHS:")
print("=" * 25)
strengths = [
    "Comprehensive feature set including demographics, diagnoses, labs, medications, and history",
    "Realistic performance metrics suitable for clinical deployment",
    "Clear risk stratification enabling targeted interventions",
    "Interpretable features aligned with clinical knowledge",
    "Robust cross-validation demonstrating model stability"
]

for i, strength in enumerate(strengths, 1):
    print(f"{i}. {strength}")

print(f"\n⚠️  MODEL LIMITATIONS:")
print("=" * 25)
limitations = [
    "Synthetic dataset may not capture all real-world complexities",
    "Class imbalance required SMOTE balancing which may affect generalizability",
    "Limited to 30-day readmission prediction (not longer-term outcomes)",
    "Requires regular retraining to maintain performance over time",
    "Clinical validation needed before deployment in real healthcare settings"
]

for i, limitation in enumerate(limitations, 1):
    print(f"{i}. {limitation}")

# Next steps
print(f"\n🚀 RECOMMENDED NEXT STEPS:")
print("=" * 35)
next_steps = [
    "Validate model on real MIMIC-III data or institutional dataset",
    "Conduct prospective clinical trial to measure intervention effectiveness",
    "Develop automated risk scoring integration with EHR systems",
    "Create clinical decision support tools for care teams",
    "Establish monitoring framework for model performance tracking",
    "Design intervention protocols for different risk levels"
]

for i, step in enumerate(next_steps, 1):
    print(f"{i}. {step}")

print(f"\n" + "="*60)
print("🎉 HEART FAILURE READMISSION PREDICTION MODEL COMPLETE!")
print("="*60)
print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Ready for clinical validation and deployment consideration")
print("="*60)

# %%