import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

# Initialize faker for generating realistic data
fake = Faker()
np.random.seed(42)
random.seed(42)

# Heart failure ICD-9 codes
HEART_FAILURE_ICD9_CODES = [
    '39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', 
    '40491', '40493', '4280', '4281', '42820', '42821', '42822', '42823', 
    '42830', '42831', '42832', '42833', '42840', '42841', '42842', '42843', '4289'
]

# Additional common comorbidity codes for heart failure patients
COMORBIDITY_CODES = [
    '25000', '25001', '25002',  # Diabetes
    '4019', '40190', '40191',   # Hypertension
    '42731', '42732',           # Atrial fibrillation
    '5849', '5859',             # Chronic kidney disease
    '49390', '49391',           # Asthma/COPD
    '2724', '2720',             # Hyperlipidemia
    '41401', '41071',           # Coronary artery disease
]

# Generate base parameters
NUM_PATIENTS = 1000
NUM_ADMISSIONS = 1500
START_DATE = datetime(2008, 1, 1)
END_DATE = datetime(2012, 12, 31)

print("Generating MIMIC-III style heart failure dataset...")
print(f"Target: {NUM_PATIENTS} patients, {NUM_ADMISSIONS} admissions")

# 1. PATIENTS table
def generate_patients():
    patients = []
    for i in range(NUM_PATIENTS):
        subject_id = 10000 + i
        gender = np.random.choice(['M', 'F'], p=[0.52, 0.48])  # Slightly more males in HF
        
        # Age distribution for heart failure (typically older patients)
        age = int(np.random.normal(72, 12))
        age = max(40, min(95, age))  # Constrain age between 40-95
        
        # Calculate DOB based on admission year and age
        dob = START_DATE - timedelta(days=age*365.25)
        
        # DOD - some patients die (mortality rate ~10-15% for HF patients)
        dod = None
        if np.random.random() < 0.12:
            # Death occurs within study period
            death_date = START_DATE + timedelta(days=np.random.randint(0, (END_DATE - START_DATE).days))
            dod = death_date
        
        patients.append({
            'SUBJECT_ID': subject_id,
            'GENDER': gender,
            'DOB': dob.strftime('%Y-%m-%d %H:%M:%S'),
            'DOD': dod.strftime('%Y-%m-%d %H:%M:%S') if dod else None,
            'DOD_HOSP': dod.strftime('%Y-%m-%d %H:%M:%S') if dod and np.random.random() < 0.6 else None,
            'DOD_SSN': dod.strftime('%Y-%m-%d %H:%M:%S') if dod and np.random.random() < 0.8 else None,
            'EXPIRE_FLAG': 1 if dod else 0
        })
    
    return pd.DataFrame(patients)

# 2. ADMISSIONS table
def generate_admissions(patients_df):
    admissions = []
    hadm_id = 20000
    
    for _, patient in patients_df.iterrows():
        subject_id = patient['SUBJECT_ID']
        patient_dod = pd.to_datetime(patient['DOD']) if patient['DOD'] else None
        
        # Number of admissions per patient (HF patients often have multiple admissions)
        num_admissions = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
        
        for admission_num in range(num_admissions):
            # Admission time
            if admission_num == 0:
                admittime = START_DATE + timedelta(days=np.random.randint(0, (END_DATE - START_DATE).days))
            else:
                # Subsequent admissions are typically readmissions within 30-180 days
                days_since_last = np.random.randint(7, 180)
                admittime = last_dischtime + timedelta(days=days_since_last)
            
            # Don't admit after death
            if patient_dod and admittime > patient_dod:
                break
                
            # Length of stay (HF patients typically 3-7 days)
            los_days = max(1, int(np.random.exponential(4.5)))
            dischtime = admittime + timedelta(days=los_days)
            
            # If patient dies, some chance death occurs during this admission
            hospital_expire_flag = 0
            if patient_dod and np.random.random() < 0.3:
                dischtime = patient_dod
                hospital_expire_flag = 1
            
            # Admission type and location
            admission_type = np.random.choice(['EMERGENCY', 'URGENT', 'ELECTIVE'], p=[0.7, 0.2, 0.1])
            admission_location = np.random.choice(['EMERGENCY ROOM ADMIT', 'CLINIC REFERRAL/PREMATURE', 'PHYSICIAN REFERRAL'], p=[0.7, 0.15, 0.15])
            
            # Discharge location
            if hospital_expire_flag:
                discharge_location = 'DEAD/EXPIRED'
            else:
                discharge_location = np.random.choice([
                    'HOME', 'HOME HEALTH CARE', 'REHAB/DISTINCT PART HOSP', 
                    'SNF', 'SHORT TERM HOSPITAL'
                ], p=[0.5, 0.2, 0.15, 0.1, 0.05])
            
            # Insurance
            insurance = np.random.choice(['Medicare', 'Private', 'Medicaid', 'Government'], p=[0.6, 0.25, 0.1, 0.05])
            
            # Language and religion
            language = np.random.choice(['ENGL', 'SPAN', 'FREN', 'ITAL'], p=[0.85, 0.08, 0.04, 0.03])
            religion = np.random.choice(['CATHOLIC', 'PROTESTANT QUAKER', 'JEWISH', 'UNOBTAINABLE', 'OTHER'], p=[0.3, 0.25, 0.1, 0.2, 0.15])
            
            admissions.append({
                'HADM_ID': hadm_id,
                'SUBJECT_ID': subject_id,
                'ADMITTIME': admittime.strftime('%Y-%m-%d %H:%M:%S'),
                'DISCHTIME': dischtime.strftime('%Y-%m-%d %H:%M:%S'),
                'DEATHTIME': dischtime.strftime('%Y-%m-%d %H:%M:%S') if hospital_expire_flag else None,
                'ADMISSION_TYPE': admission_type,
                'ADMISSION_LOCATION': admission_location,
                'DISCHARGE_LOCATION': discharge_location,
                'INSURANCE': insurance,
                'LANGUAGE': language,
                'RELIGION': religion,
                'MARITAL_STATUS': np.random.choice(['MARRIED', 'SINGLE', 'WIDOWED', 'DIVORCED'], p=[0.5, 0.2, 0.2, 0.1]),
                'ETHNICITY': np.random.choice(['WHITE', 'BLACK/AFRICAN AMERICAN', 'HISPANIC/LATINO', 'ASIAN', 'OTHER'], p=[0.7, 0.15, 0.08, 0.05, 0.02]),
                'EDREGTIME': (admittime - timedelta(hours=np.random.randint(1, 6))).strftime('%Y-%m-%d %H:%M:%S') if admission_type == 'EMERGENCY' else None,
                'EDOUTTIME': admittime.strftime('%Y-%m-%d %H:%M:%S') if admission_type == 'EMERGENCY' else None,
                'DIAGNOSIS': 'HEART FAILURE',
                'HOSPITAL_EXPIRE_FLAG': hospital_expire_flag,
                'HAS_CHARTEVENTS_DATA': 1
            })
            
            hadm_id += 1
            last_dischtime = dischtime
    
    return pd.DataFrame(admissions)
# 3. DIAGNOSES_ICD table
def generate_diagnoses_icd(admissions_df):
    diagnoses = []
    
    for _, admission in admissions_df.iterrows():
        hadm_id = admission['HADM_ID']
        subject_id = admission['SUBJECT_ID']
        
        # Primary diagnosis - always a heart failure code
        primary_hf_code = np.random.choice(HEART_FAILURE_ICD9_CODES)
        diagnoses.append({
            'HADM_ID': hadm_id,
            'SUBJECT_ID': subject_id,
            'SEQ_NUM': 1,
            'ICD9_CODE': primary_hf_code
        })
        
        # Secondary diagnoses - mix of HF codes and comorbidities
        num_secondary = np.random.randint(2, 8)  # HF patients typically have multiple comorbidities
        seq_num = 2
        
        # Add additional HF codes (some patients have multiple HF diagnoses)
        if np.random.random() < 0.3:
            additional_hf_code = np.random.choice([code for code in HEART_FAILURE_ICD9_CODES if code != primary_hf_code])
            diagnoses.append({
                'HADM_ID': hadm_id,
                'SUBJECT_ID': subject_id,
                'SEQ_NUM': seq_num,
                'ICD9_CODE': additional_hf_code
            })
            seq_num += 1
            num_secondary -= 1
        
        # Add comorbidity codes
        selected_comorbidities = np.random.choice(COMORBIDITY_CODES, size=min(num_secondary, len(COMORBIDITY_CODES)), replace=False)
        for comorbidity_code in selected_comorbidities:
            diagnoses.append({
                'HADM_ID': hadm_id,
                'SUBJECT_ID': subject_id,
                'SEQ_NUM': seq_num,
                'ICD9_CODE': comorbidity_code
            })
            seq_num += 1
    
    return pd.DataFrame(diagnoses)

# 4. ICUSTAYS table (subset of admissions go to ICU)
def generate_icustays(admissions_df):
    icustays = []
    icustay_id = 30000
    
    # About 30% of HF admissions require ICU care
    icu_admissions = admissions_df.sample(frac=0.3)
    
    for _, admission in icu_admissions.iterrows():
        hadm_id = admission['HADM_ID']
        subject_id = admission['SUBJECT_ID']
        
        admittime = pd.to_datetime(admission['ADMITTIME'])
        dischtime = pd.to_datetime(admission['DISCHTIME'])
        
        # ICU stay typically starts within first day of admission
        icu_intime = admittime + timedelta(hours=np.random.randint(0, 24))
        
        # ICU length of stay (typically 1-5 days for HF)
        icu_los_hours = max(12, int(np.random.exponential(48)))  # Average 2 days
        icu_outtime = min(icu_intime + timedelta(hours=icu_los_hours), dischtime)
        
        # ICU type
        first_careunit = np.random.choice(['MICU', 'CCU', 'CVICU'], p=[0.5, 0.3, 0.2])
        last_careunit = first_careunit  # Most patients stay in same unit
        
        icustays.append({
            'ICUSTAY_ID': icustay_id,
            'HADM_ID': hadm_id,
            'SUBJECT_ID': subject_id,
            'FIRST_CAREUNIT': first_careunit,
            'LAST_CAREUNIT': last_careunit,
            'FIRST_WARDID': np.random.randint(1, 50),
            'LAST_WARDID': np.random.randint(1, 50),
            'INTIME': icu_intime.strftime('%Y-%m-%d %H:%M:%S'),
            'OUTTIME': icu_outtime.strftime('%Y-%m-%d %H:%M:%S'),
            'LOS': round((icu_outtime - icu_intime).total_seconds() / 86400, 2)  # LOS in days
        })
        
        icustay_id += 1
    
    return pd.DataFrame(icustays)
# 5. LABEVENTS table (key lab values for HF patients)
def generate_labevents(admissions_df):
    labevents = []
    
    # Common lab items for heart failure patients
    lab_items = {
        50862: 'Albumin',           # Low in HF
        50863: 'Alkaline Phosphatase',
        50868: 'Anion Gap',
        50882: 'Bicarbonate',
        50885: 'Bilirubin, Total',
        50912: 'Creatinine',        # Often elevated in HF
        50931: 'Glucose',
        50960: 'Magnesium',
        50970: 'Phosphate',
        50971: 'Potassium',         # Important for HF management
        50983: 'Sodium',            # Often low in HF
        51006: 'Urea Nitrogen',     # BUN, often elevated
        51221: 'Hematocrit',
        51222: 'Hemoglobin',
        51265: 'Platelet Count',
        51279: 'Red Blood Cells',
        51301: 'White Blood Cells',
        50809: 'Glucose',
        50820: 'pH',
        50821: 'pO2',
        50818: 'pCO2'
    }
    
    row_id = 1
    
    for _, admission in admissions_df.iterrows():
        hadm_id = admission['HADM_ID']
        subject_id = admission['SUBJECT_ID']
        
        admittime = pd.to_datetime(admission['ADMITTIME'])
        dischtime = pd.to_datetime(admission['DISCHTIME'])
        
        # Generate labs for this admission
        num_lab_days = max(1, int((dischtime - admittime).days))
        
        for day in range(num_lab_days):
            lab_time = admittime + timedelta(days=day, hours=np.random.randint(6, 10))
            
            # Not all labs drawn every day
            if np.random.random() < 0.7:  # 70% chance of labs on any given day
                
                # Select subset of labs to draw
                num_labs = np.random.randint(5, 15)
                selected_items = np.random.choice(list(lab_items.keys()), size=num_labs, replace=False)
                
                for itemid in selected_items:
                    # Generate realistic values based on lab type
                    if itemid == 50912:  # Creatinine (elevated in HF)
                        value = max(0.5, np.random.normal(1.8, 0.8))
                    elif itemid == 50983:  # Sodium (often low in HF)
                        value = max(125, np.random.normal(135, 5))
                    elif itemid == 50971:  # Potassium
                        value = max(2.5, min(6.0, np.random.normal(4.0, 0.5)))
                    elif itemid == 51006:  # BUN (often elevated)
                        value = max(5, np.random.normal(35, 15))
                    elif itemid == 50862:  # Albumin (often low)
                        value = max(1.5, np.random.normal(3.2, 0.6))
                    elif itemid == 51222:  # Hemoglobin
                        value = max(6, np.random.normal(11.5, 2.0))
                    elif itemid == 51221:  # Hematocrit
                        value = max(18, np.random.normal(34, 6))
                    elif itemid == 50931:  # Glucose
                        value = max(60, np.random.normal(140, 40))
                    else:
                        # Generic normal distribution for other labs
                        value = max(0.1, np.random.normal(10, 5))
                    
                    labevents.append({
                        'ROW_ID': row_id,
                        'SUBJECT_ID': subject_id,
                        'HADM_ID': hadm_id,
                        'ITEMID': itemid,
                        'CHARTTIME': lab_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'VALUE': round(value, 2),
                        'VALUENUM': round(value, 2),
                        'VALUEUOM': 'mg/dL' if itemid in [50912, 50931, 51006] else 'mEq/L' if itemid in [50971, 50983] else 'g/dL',
                        'FLAG': 'abnormal' if np.random.random() < 0.3 else None
                    })
                    
                    row_id += 1
    
    return pd.DataFrame(labevents)
# 6. PRESCRIPTIONS table (medications common in HF)
def generate_prescriptions(admissions_df):
    prescriptions = []
    
    # Common heart failure medications
    hf_medications = [
        ('Furosemide', 'Lasix'),           # Diuretic
        ('Lisinopril', 'Prinivil'),        # ACE inhibitor
        ('Metoprolol', 'Lopressor'),       # Beta blocker
        ('Spironolactone', 'Aldactone'),   # Aldosterone antagonist
        ('Digoxin', 'Lanoxin'),            # Cardiac glycoside
        ('Carvedilol', 'Coreg'),           # Beta blocker
        ('Enalapril', 'Vasotec'),          # ACE inhibitor
        ('Losartan', 'Cozaar'),            # ARB
        ('Hydrochlorothiazide', 'HCTZ'),   # Diuretic
        ('Warfarin', 'Coumadin'),          # Anticoagulant
        ('Aspirin', 'ASA'),                # Antiplatelet
        ('Potassium Chloride', 'K-Dur'),   # Electrolyte replacement
    ]
    
    row_id = 1
    
    for _, admission in admissions_df.iterrows():
        hadm_id = admission['HADM_ID']
        subject_id = admission['SUBJECT_ID']
        
        admittime = pd.to_datetime(admission['ADMITTIME'])
        dischtime = pd.to_datetime(admission['DISCHTIME'])
        
        # HF patients typically on multiple medications
        num_medications = np.random.randint(4, 10)
        selected_meds = np.random.choice(len(hf_medications), size=num_medications, replace=False)
        
        for med_idx in selected_meds:
            drug, brand = hf_medications[med_idx]
            
            # Start time (usually within first day)
            startdate = admittime + timedelta(hours=np.random.randint(0, 24))
            
            # End time (usually continues through discharge)
            enddate = dischtime - timedelta(hours=np.random.randint(0, 12))
            
            # Dosage varies by medication
            if 'Furosemide' in drug:
                dose_val_rx = f"{np.random.choice([20, 40, 80])} mg"
                route = 'PO'
            elif 'Lisinopril' in drug or 'Enalapril' in drug:
                dose_val_rx = f"{np.random.choice([2.5, 5, 10, 20])} mg"
                route = 'PO'
            elif 'Metoprolol' in drug or 'Carvedilol' in drug:
                dose_val_rx = f"{np.random.choice([12.5, 25, 50])} mg"
                route = 'PO'
            elif 'Digoxin' in drug:
                dose_val_rx = f"{np.random.choice([0.125, 0.25])} mg"
                route = 'PO'
            else:
                dose_val_rx = f"{np.random.randint(5, 100)} mg"
                route = np.random.choice(['PO', 'IV'])
            
            prescriptions.append({
                'ROW_ID': row_id,
                'SUBJECT_ID': subject_id,
                'HADM_ID': hadm_id,
                'ICUSTAY_ID': None,  # Most prescriptions not ICU-specific
                'STARTDATE': startdate.strftime('%Y-%m-%d %H:%M:%S'),
                'ENDDATE': enddate.strftime('%Y-%m-%d %H:%M:%S'),
                'DRUG_TYPE': 'MAIN',
                'DRUG': drug,
                'DRUG_NAME_POE': brand,
                'DRUG_NAME_GENERIC': drug,
                'FORMULARY_DRUG_CD': f"DRUG{row_id:06d}",
                'GSN': f"{100000 + row_id}",
                'NDC': f"{row_id:010d}",
                'PROD_STRENGTH': dose_val_rx,
                'DOSE_VAL_RX': dose_val_rx,
                'DOSE_UNIT_RX': dose_val_rx.split()[-1],
                'FORM_VAL_DISP': 'TAB' if route == 'PO' else 'INJ',
                'FORM_UNIT_DISP': 'TAB' if route == 'PO' else 'VIAL',
                'ROUTE': route
            })
            
            row_id += 1
    
    return pd.DataFrame(prescriptions)

# 7. Generate all tables and save
def main():
    print("Generating PATIENTS table...")
    patients_df = generate_patients()
    
    print("Generating ADMISSIONS table...")
    admissions_df = generate_admissions(patients_df)
    
    print("Generating DIAGNOSES_ICD table...")
    diagnoses_df = generate_diagnoses_icd(admissions_df)
    
    print("Generating ICUSTAYS table...")
    icustays_df = generate_icustays(admissions_df)
    
    print("Generating LABEVENTS table...")
    labevents_df = generate_labevents(admissions_df)
    
    print("Generating PRESCRIPTIONS table...")
    prescriptions_df = generate_prescriptions(admissions_df)
    
    # Save all tables
    print("\nSaving tables to CSV files...")
    patients_df.to_csv('MIMIC_PATIENTS.csv', index=False)
    admissions_df.to_csv('MIMIC_ADMISSIONS.csv', index=False)
    diagnoses_df.to_csv('MIMIC_DIAGNOSES_ICD.csv', index=False)
    icustays_df.to_csv('MIMIC_ICUSTAYS.csv', index=False)
    labevents_df.to_csv('MIMIC_LABEVENTS.csv', index=False)
    prescriptions_df.to_csv('MIMIC_PRESCRIPTIONS.csv', index=False)
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"Patients: {len(patients_df)}")
    print(f"Admissions: {len(admissions_df)}")
    print(f"Diagnoses: {len(diagnoses_df)}")
    print(f"ICU Stays: {len(icustays_df)}")
    print(f"Lab Events: {len(labevents_df)}")
    print(f"Prescriptions: {len(prescriptions_df)}")
    
    # Show readmission analysis
    readmissions = admissions_df.groupby('SUBJECT_ID').size()
    readmission_rate = (readmissions > 1).sum() / len(patients_df) * 100
    print(f"Readmission rate: {readmission_rate:.1f}%")
    
    # Show mortality rate
    mortality_rate = patients_df['EXPIRE_FLAG'].sum() / len(patients_df) * 100
    print(f"Mortality rate: {mortality_rate:.1f}%")

if __name__ == "__main__":
    main()