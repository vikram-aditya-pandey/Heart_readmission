# MIMIC-III Style Heart Failure Dataset

## Overview
This synthetic dataset mimics the structure and characteristics of the MIMIC-III database, specifically focused on heart failure patients. It includes realistic patient demographics, admission patterns, diagnoses, lab values, and medications commonly seen in heart failure care.

## Heart Failure ICD-9 Codes Used
The dataset includes patients with the following heart failure diagnosis codes:
- 39891, 40201, 40211, 40291, 40401, 40403, 40411, 40413
- 40491, 40493, 4280, 4281, 42820, 42821, 42822, 42823
- 42830, 42831, 42832, 42833, 42840, 42841, 42842, 42843, 4289

## Generated Tables

### 1. MIMIC_PATIENTS.csv
- **Records**: 1,000 patients
- **Key Fields**: SUBJECT_ID, GENDER, DOB, DOD, EXPIRE_FLAG
- **Characteristics**: 
  - Age distribution: Mean ~72 years (typical for HF patients)
  - Gender: Slightly more males (52% vs 48%)
  - Mortality rate: ~11.4%

### 2. MIMIC_ADMISSIONS.csv
- **Records**: 1,897 admissions
- **Key Fields**: HADM_ID, SUBJECT_ID, ADMITTIME, DISCHTIME, DIAGNOSIS
- **Characteristics**:
  - Readmission rate: 53.9% (realistic for HF)
  - Average length of stay: 3-7 days
  - 70% emergency admissions
  - Primary diagnosis: "HEART FAILURE"

### 3. MIMIC_DIAGNOSES_ICD.csv
- **Records**: 10,476 diagnosis codes
- **Key Fields**: HADM_ID, SUBJECT_ID, SEQ_NUM, ICD9_CODE
- **Characteristics**:
  - Primary diagnosis always from HF code list
  - Secondary diagnoses include common comorbidities:
    - Diabetes (25000-25002)
    - Hypertension (4019, 40190, 40191)
    - Atrial fibrillation (42731, 42732)
    - Chronic kidney disease (5849, 5859)
    - COPD/Asthma (49390, 49391)

### 4. MIMIC_ICUSTAYS.csv
- **Records**: 569 ICU stays (~30% of admissions)
- **Key Fields**: ICUSTAY_ID, HADM_ID, INTIME, OUTTIME, FIRST_CAREUNIT
- **Characteristics**:
  - Care units: MICU (50%), CCU (30%), CVICU (20%)
  - Average ICU length of stay: ~2 days

### 5. MIMIC_LABEVENTS.csv
- **Records**: 110,856 lab results
- **Key Fields**: SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE
- **Important Labs for HF**:
  - Creatinine (50912): Often elevated
  - Sodium (50983): Often low in HF
  - BUN (51006): Often elevated
  - Potassium (50971): Critical for HF management
  - Albumin (50862): Often low

### 6. MIMIC_PRESCRIPTIONS.csv
- **Records**: 12,326 prescriptions
- **Key Fields**: SUBJECT_ID, HADM_ID, DRUG, DOSE_VAL_RX, ROUTE
- **Common HF Medications**:
  - Furosemide (Lasix) - Diuretic
  - Lisinopril (Prinivil) - ACE inhibitor
  - Metoprolol (Lopressor) - Beta blocker
  - Spironolactone (Aldactone) - Aldosterone antagonist
  - Digoxin (Lanoxin) - Cardiac glycoside

## Dataset Characteristics

### Realistic Clinical Patterns
- **Readmission Patterns**: 53.9% of patients have multiple admissions
- **Mortality**: 11.4% mortality rate during study period
- **Comorbidities**: Multiple diagnosis codes per admission (average 5-6)
- **Medication Complexity**: 4-10 medications per admission
- **Lab Monitoring**: Regular lab draws with HF-specific abnormalities

### Time Periods
- **Study Period**: 2008-2012 (5 years)
- **Readmission Window**: 7-180 days between admissions
- **Lab Frequency**: Daily during admission for most patients

## Use Cases
This dataset can be used for:
1. **Readmission Prediction Models**: Identify patients at risk for 30/60/90-day readmission
2. **Mortality Risk Assessment**: Predict in-hospital and post-discharge mortality
3. **Length of Stay Prediction**: Estimate hospital and ICU length of stay
4. **Medication Adherence Studies**: Analyze prescription patterns
5. **Comorbidity Analysis**: Study disease progression and complications
6. **Healthcare Utilization**: Analyze admission patterns and resource usage

## Data Quality Notes
- All timestamps are realistic and consistent
- Lab values reflect typical HF patient abnormalities
- Medication dosages are clinically appropriate
- ICD-9 codes follow proper sequencing (primary diagnosis first)
- Patient demographics reflect real-world HF population

## Files Generated
- `MIMIC_PATIENTS.csv` - Patient demographics
- `MIMIC_ADMISSIONS.csv` - Hospital admissions
- `MIMIC_DIAGNOSES_ICD.csv` - Diagnosis codes
- `MIMIC_ICUSTAYS.csv` - ICU stays
- `MIMIC_LABEVENTS.csv` - Laboratory results
- `MIMIC_PRESCRIPTIONS.csv` - Medications