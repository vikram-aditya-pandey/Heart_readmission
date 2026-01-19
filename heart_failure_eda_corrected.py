import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def corrected_diagnosis_analysis():
    """Corrected diagnosis analysis with proper data types"""
    
    print("\n" + "="*60)
    print("CORRECTED HEART FAILURE DIAGNOSIS ANALYSIS")
    print("="*60)
    
    # Load data
    diagnoses = pd.read_csv('MIMIC_DIAGNOSES_ICD.csv')
    
    # Heart failure codes as integers (not strings!)
    hf_codes_int = [39891, 40201, 40211, 40291, 40401, 40403, 40411, 40413,
                    40491, 40493, 4280, 4281, 42820, 42821, 42822, 42823,
                    42830, 42831, 42832, 42833, 42840, 42841, 42842, 42843, 4289]
    
    # Separate HF and non-HF diagnoses
    hf_diagnoses = diagnoses[diagnoses['ICD9_CODE'].isin(hf_codes_int)]
    other_diagnoses = diagnoses[~diagnoses['ICD9_CODE'].isin(hf_codes_int)]
    
    print(f"✓ Total diagnoses: {len(diagnoses):,}")
    print(f"✓ Heart failure diagnoses: {len(hf_diagnoses):,} ({len(hf_diagnoses)/len(diagnoses)*100:.1f}%)")
    print(f"✓ Other diagnoses (comorbidities): {len(other_diagnoses):,} ({len(other_diagnoses)/len(diagnoses)*100:.1f}%)")
    
    # Most common HF codes
    print(f"\nMost Common Heart Failure ICD-9 Codes:")
    hf_code_counts = hf_diagnoses['ICD9_CODE'].value_counts().head(15)
    
    # HF code descriptions
    hf_descriptions = {
        39891: "Rheumatic heart failure",
        40201: "Malignant hypertensive heart disease with heart failure",
        40211: "Benign hypertensive heart disease with heart failure", 
        40291: "Unspecified hypertensive heart disease with heart failure",
        40401: "Malignant hypertensive heart and chronic kidney disease with heart failure",
        40403: "Malignant hypertensive heart and chronic kidney disease with heart failure and chronic kidney disease",
        40411: "Benign hypertensive heart and chronic kidney disease with heart failure",
        40413: "Benign hypertensive heart and chronic kidney disease with heart failure and chronic kidney disease",
        40491: "Unspecified hypertensive heart and chronic kidney disease with heart failure",
        40493: "Unspecified hypertensive heart and chronic kidney disease with heart failure and chronic kidney disease",
        4280: "Heart failure, unspecified",
        4281: "Acute heart failure",
        42820: "Systolic heart failure, unspecified",
        42821: "Acute systolic heart failure",
        42822: "Chronic systolic heart failure",
        42823: "Acute on chronic systolic heart failure",
        42830: "Diastolic heart failure, unspecified",
        42831: "Acute diastolic heart failure",
        42832: "Chronic diastolic heart failure",
        42833: "Acute on chronic diastolic heart failure",
        42840: "Combined systolic and diastolic heart failure, unspecified",
        42841: "Acute combined systolic and diastolic heart failure",
        42842: "Chronic combined systolic and diastolic heart failure",
        42843: "Acute on chronic combined systolic and diastolic heart failure",
        4289: "Heart failure, unspecified"
    }
    
    for code, count in hf_code_counts.items():
        description = hf_descriptions.get(code, "Unknown HF code")
        print(f"  {code}: {count:,} occurrences - {description}")
    
    # HF diagnosis patterns
    print(f"\nHeart Failure Diagnosis Patterns:")
    
    # Primary vs secondary HF diagnoses
    primary_hf = hf_diagnoses[hf_diagnoses['SEQ_NUM'] == 1]
    secondary_hf = hf_diagnoses[hf_diagnoses['SEQ_NUM'] > 1]
    
    print(f"  Primary HF diagnoses (SEQ_NUM=1): {len(primary_hf):,} ({len(primary_hf)/len(hf_diagnoses)*100:.1f}%)")
    print(f"  Secondary HF diagnoses (SEQ_NUM>1): {len(secondary_hf):,} ({len(secondary_hf)/len(hf_diagnoses)*100:.1f}%)")
    
    # HF diagnoses per admission
    hf_per_admission = hf_diagnoses.groupby('HADM_ID').size()
    print(f"  Average HF codes per admission: {hf_per_admission.mean():.1f}")
    print(f"  Admissions with multiple HF codes: {(hf_per_admission > 1).sum():,}")
    
    # Most common comorbidities
    print(f"\nMost Common Comorbidities (Non-HF Codes):")
    comorbidity_counts = other_diagnoses['ICD9_CODE'].value_counts().head(10)
    
    # Common comorbidity descriptions
    comorbidity_descriptions = {
        25000: "Diabetes mellitus without mention of complication",
        25001: "Diabetes mellitus without mention of complication, type I",
        25002: "Diabetes mellitus without mention of complication, type II",
        4019: "Unspecified essential hypertension",
        40190: "Hypertensive chronic kidney disease, unspecified",
        40191: "Hypertensive chronic kidney disease, benign",
        42731: "Atrial fibrillation",
        42732: "Atrial flutter",
        5849: "Acute kidney failure, unspecified",
        5859: "Chronic kidney disease, unspecified",
        49390: "Asthma, unspecified type, unspecified",
        49391: "Asthma, unspecified type, with status asthmaticus",
        2724: "Other and unspecified hyperlipidemia",
        2720: "Pure hypercholesterolemia",
        41401: "Coronary atherosclerosis of native coronary artery",
        41071: "Subendocardial infarction, initial episode of care"
    }
    
    for code, count in comorbidity_counts.items():
        description = comorbidity_descriptions.get(code, "Unknown condition")
        print(f"  {code}: {count:,} occurrences - {description}")
    
    return hf_diagnoses, other_diagnoses

def comprehensive_summary():
    """Generate comprehensive corrected summary"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE HEART FAILURE DATASET SUMMARY")
    print("="*60)
    
    # Load all data
    patients = pd.read_csv('MIMIC_PATIENTS.csv')
    admissions = pd.read_csv('MIMIC_ADMISSIONS.csv')
    diagnoses = pd.read_csv('MIMIC_DIAGNOSES_ICD.csv')
    icustays = pd.read_csv('MIMIC_ICUSTAYS.csv')
    labevents = pd.read_csv('MIMIC_LABEVENTS.csv')
    prescriptions = pd.read_csv('MIMIC_PRESCRIPTIONS.csv')
    
    # Convert dates
    patients['DOB'] = pd.to_datetime(patients['DOB'])
    patients['DOD'] = pd.to_datetime(patients['DOD'])
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
    
    # Calculate key metrics
    admissions = admissions.merge(patients[['SUBJECT_ID', 'DOB', 'GENDER']], on='SUBJECT_ID')
    admissions['AGE_AT_ADMISSION'] = (admissions['ADMITTIME'] - admissions['DOB']).dt.days / 365.25
    admissions['LOS_DAYS'] = (admissions['DISCHTIME'] - admissions['ADMITTIME']).dt.days
    
    # Heart failure codes
    hf_codes_int = [39891, 40201, 40211, 40291, 40401, 40403, 40411, 40413,
                    40491, 40493, 4280, 4281, 42820, 42821, 42822, 42823,
                    42830, 42831, 42832, 42833, 42840, 42841, 42842, 42843, 4289]
    
    hf_diagnoses = diagnoses[diagnoses['ICD9_CODE'].isin(hf_codes_int)]
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Study Period: {admissions['ADMITTIME'].min().strftime('%Y-%m-%d')} to {admissions['ADMITTIME'].max().strftime('%Y-%m-%d')}")
    print(f"   Total Patients: {len(patients):,}")
    print(f"   Total Admissions: {len(admissions):,}")
    print(f"   Total Diagnoses: {len(diagnoses):,}")
    print(f"   Heart Failure Diagnoses: {len(hf_diagnoses):,} ({len(hf_diagnoses)/len(diagnoses)*100:.1f}%)")
    print(f"   ICU Stays: {len(icustays):,}")
    print(f"   Lab Events: {len(labevents):,}")
    print(f"   Prescriptions: {len(prescriptions):,}")
    
    print(f"\n👥 PATIENT DEMOGRAPHICS:")
    print(f"   Average Age: {admissions['AGE_AT_ADMISSION'].mean():.1f} years")
    print(f"   Age Range: {admissions['AGE_AT_ADMISSION'].min():.1f} - {admissions['AGE_AT_ADMISSION'].max():.1f} years")
    print(f"   Male Patients: {(admissions['GENDER'] == 'M').sum():,} ({(admissions['GENDER'] == 'M').mean()*100:.1f}%)")
    print(f"   Female Patients: {(admissions['GENDER'] == 'F').sum():,} ({(admissions['GENDER'] == 'F').mean()*100:.1f}%)")
    
    print(f"\n🏥 CLINICAL OUTCOMES:")
    print(f"   Average Length of Stay: {admissions['LOS_DAYS'].mean():.1f} days")
    print(f"   Median Length of Stay: {admissions['LOS_DAYS'].median():.1f} days")
    print(f"   ICU Utilization Rate: {len(icustays)/len(admissions)*100:.1f}%")
    print(f"   Overall Mortality Rate: {patients['EXPIRE_FLAG'].mean()*100:.1f}%")
    print(f"   In-Hospital Mortality: {admissions['HOSPITAL_EXPIRE_FLAG'].mean()*100:.1f}%")
    
    # Readmission analysis
    admissions_sorted = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME'])
    admissions_sorted['PREV_DISCHTIME'] = admissions_sorted.groupby('SUBJECT_ID')['DISCHTIME'].shift(1)
    admissions_sorted['DAYS_TO_READMISSION'] = (admissions_sorted['ADMITTIME'] - admissions_sorted['PREV_DISCHTIME']).dt.days
    
    readmissions_30 = ((admissions_sorted['DAYS_TO_READMISSION'] <= 30) & (admissions_sorted['DAYS_TO_READMISSION'] > 0)).sum()
    readmissions_60 = ((admissions_sorted['DAYS_TO_READMISSION'] <= 60) & (admissions_sorted['DAYS_TO_READMISSION'] > 0)).sum()
    
    print(f"\n🔄 READMISSION PATTERNS:")
    print(f"   30-day Readmissions: {readmissions_30:,} ({readmissions_30/len(admissions)*100:.1f}%)")
    print(f"   60-day Readmissions: {readmissions_60:,} ({readmissions_60/len(admissions)*100:.1f}%)")
    
    patient_admission_counts = admissions.groupby('SUBJECT_ID').size()
    patients_with_multiple = (patient_admission_counts > 1).sum()
    print(f"   Patients with Multiple Admissions: {patients_with_multiple:,} ({patients_with_multiple/len(patients)*100:.1f}%)")
    
    print(f"\n💊 MEDICATION PATTERNS:")
    med_counts = prescriptions['DRUG'].value_counts()
    print(f"   Total Medications Prescribed: {len(prescriptions):,}")
    print(f"   Unique Medications: {prescriptions['DRUG'].nunique()}")
    print(f"   Average Medications per Admission: {len(prescriptions)/len(admissions):.1f}")
    
    print(f"\n🧪 LABORATORY PATTERNS:")
    print(f"   Total Lab Tests: {len(labevents):,}")
    print(f"   Unique Lab Types: {labevents['ITEMID'].nunique()}")
    print(f"   Average Labs per Admission: {len(labevents)/len(admissions):.1f}")
    
    # Key lab abnormalities
    creatinine = labevents[labevents['ITEMID'] == 50912]['VALUENUM']
    sodium = labevents[labevents['ITEMID'] == 50983]['VALUENUM']
    bun = labevents[labevents['ITEMID'] == 51006]['VALUENUM']
    
    if len(creatinine) > 0:
        elevated_creat = (creatinine > 1.5).mean() * 100
        print(f"   Elevated Creatinine (>1.5): {elevated_creat:.1f}% of tests")
    
    if len(sodium) > 0:
        low_sodium = (sodium < 135).mean() * 100
        print(f"   Low Sodium (<135): {low_sodium:.1f}% of tests")
    
    print(f"\n✅ DATA QUALITY ASSESSMENT:")
    print(f"   Missing Patient DOB: {patients['DOB'].isna().sum()}")
    print(f"   Missing Admission Times: {admissions['ADMITTIME'].isna().sum()}")
    print(f"   Missing Discharge Times: {admissions['DISCHTIME'].isna().sum()}")
    print(f"   Invalid Length of Stay: {(admissions['LOS_DAYS'] < 0).sum()}")
    print(f"   Duplicate Patient IDs: {patients['SUBJECT_ID'].duplicated().sum()}")
    print(f"   Duplicate Admission IDs: {admissions['HADM_ID'].duplicated().sum()}")
    
    print(f"\n" + "="*60)
    print("✅ DATASET READY FOR MACHINE LEARNING & ANALYSIS!")
    print("="*60)

if __name__ == "__main__":
    corrected_diagnosis_analysis()
    comprehensive_summary()