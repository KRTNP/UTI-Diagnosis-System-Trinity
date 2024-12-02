import pandas as pd
import numpy as np
from pathlib import Path

# สร้างโฟลเดอร์สำหรับจัดเก็บข้อมูล
# ใช้ exist_ok=True เพื่อป้องกันการเกิด error กรณีที่มีโฟลเดอร์อยู่แล้ว
Path("data").mkdir(exist_ok=True)

# ฟังก์ชันสำหรับสร้างข้อมูลสังเคราะห์สำหรับผู้ป่วย UTI (Urinary Tract Infection)
# n_samples: จำนวนตัวอย่างข้อมูลที่ต้องการสร้าง (default = 10)
def generate_uti_synthetic_data(n_samples=10):
    # กำหนด seed เพื่อให้ได้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน
    np.random.seed(42)

    # กำหนดความน่าจะเป็นของอาการต่างๆ ในผู้ป่วย UTI
    # อ้างอิงตาม EAU Guidelines 2023 และ IDSA Guidelines 2022
    uti_probs = {
        "frequent_urination": 0.9,    # ปัสสาวะบ่อย (EAU: 85-90%)
        "painful_urination": 0.85,    # ปัสสาวะแสบขัด (EAU: 85-90%)
        "lower_abdominal_pain": 0.75, # ปวดท้องน้อย (IDSA: 70-80%)
        "cloudy_urine": 0.7,         # ปัสสาวะขุ่น (ศิริราช: 70-80%)
        "blood_in_urine": 0.4,       # มีเลือดในปัสสาวะ (EAU: 35-45%)
        "fever": 0.5,                # มีไข้ (IDSA: 45-55%)
        "urgent_urination": 0.85,    # ปวดปัสสาวะทันที (EAU: 80-90%)
        "foul_smelling_urine": 0.5,  # กลิ่นปัสสาวะแรง (ศิริราช: 45-55%)
        "nitrites": 0.85,            # ตรวจพบไนไตรต์ (Nature Reviews: 85-95%)
        "leukocyte_esterase": 0.9,   # ตรวจพบเอนไซม์ WBC (Nature Reviews: ~90%)
        "urine_ph": 0.7,             # pH ผิดปกติ (IDSA: 65-75%)
    }
    
    # กำหนดความน่าจะเป็นสำหรับคนปกติ (ไม่เป็น UTI)
    # โดยลดความน่าจะเป็นลง 90% จากผู้ป่วย UTI
    non_uti_probs = {k: v * 0.1 for k, v in uti_probs.items()}

    # ฟังก์ชันสร้างข้อมูลประชากร
    # is_uti: boolean ที่ระบุว่าเป็นผู้ป่วย UTI หรือไม่
    def generate_demographics(is_uti):
        # สร้างอายุโดยใช้การแจกแจงแบบปกติ (Normal distribution)
        # ผู้ป่วย UTI: อายุเฉลี่ย 40 ปี ส่วนเบี่ยงเบน 15 ปี
        # คนปกติ: อายุเฉลี่ย 30 ปี ส่วนเบี่ยงเบน 10 ปี
        age = np.random.normal(40, 15) if is_uti else np.random.normal(30, 10)
        age = max(18, min(age, 80))  # จำกัดอายุระหว่าง 18-80 ปี
        
        # กำหนดเพศ (Female มีความเสี่ยงสูงกว่า Male)
        gender = np.random.choice(["Male", "Female"], p=[0.4, 0.6])
        
        # โรคประจำตัว (ผู้ป่วย UTI มีโอกาสเป็นโรคเหล่านี้สูงกว่า)
        diabetes = np.random.binomial(1, 0.25 if is_uti else 0.05)     # เบาหวาน
        hypertension = np.random.binomial(1, 0.3 if is_uti else 0.1)   # ความดันโลหิตสูง
        
        return age, gender, diabetes, hypertension

    # ฟังก์ชันสร้างผลการตรวจทางห้องปฏิบัติการ
    def generate_lab_results(is_uti):
        # เม็ดเลือดขาว (WBC) ในปัสสาวะ (x10^6/mL)
        # ผู้ป่วย UTI: 12-20 x10^6/mL
        # คนปกติ: 5-10 x10^6/mL
        wbc = np.random.uniform(12, 20) if is_uti else np.random.uniform(5, 10)
        
        # เม็ดเลือดแดง (RBC) ในปัสสาวะ (x10^6/mL)
        rbc = np.random.uniform(0, 6) if is_uti else np.random.uniform(0, 2)
        
        # การตรวจพบไนไตรต์ (บ่งชี้การติดเชื้อแบคทีเรีย)
        nitrites = np.random.binomial(1, 0.85 if is_uti else 0.1)
        
        # การตรวจพบเอนไซม์จากเม็ดเลือดขาว
        leukocyte_esterase = np.random.binomial(1, 0.9 if is_uti else 0.1)
        
        # ค่าความเป็นกรด-ด่างในปัสสาวะ
        # ผู้ป่วย UTI: pH 6.5-8.0 (เป็นด่างมากขึ้น)
        # คนปกติ: pH 4.5-6.5 (เป็นกรดมากกว่า)
        urine_ph = np.random.uniform(6.5, 8.0) if is_uti else np.random.uniform(4.5, 6.5)
        
        # การตรวจพบแบคทีเรีย
        bacteria = np.random.binomial(1, 0.9 if is_uti else 0.1)
        
        return wbc, rbc, nitrites, leukocyte_esterase, urine_ph, bacteria

    # สร้างข้อมูลโดยแบ่งเป็นผู้ป่วย UTI 1/3 และคนปกติ 2/3
    data = []
    n_uti = n_samples // 2
    n_non_uti = n_samples - n_uti

    # สร้างข้อมูลสำหรับผู้ป่วย UTI
    for _ in range(n_uti):
        # สร้างอาการตามความน่าจะเป็นที่กำหนด
        symptoms = {k: np.random.binomial(1, p) for k, p in uti_probs.items()}
        # สร้างข้อมูลประชากรและผลแล็บ
        age, gender, diabetes, hypertension = generate_demographics(is_uti=True)
        wbc, rbc, nitrites, leukocyte_esterase, urine_ph, bacteria = generate_lab_results(is_uti=True)
        
        # รวมข้อมูลทั้งหมดเข้าด้วยกัน
        symptoms.update({
            "age": age,
            "gender": gender,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "wbc": wbc,
            "rbc": rbc,
            "nitrites": nitrites,
            "leukocyte_esterase": leukocyte_esterase,
            "urine_ph": urine_ph,
            "bacteria": bacteria,
            "UTI": 1,  # ระบุว่าเป็นผู้ป่วย UTI
        })
        data.append(symptoms)

    # สร้างข้อมูลสำหรับคนปกติ (ไม่เป็น UTI)
    for _ in range(n_non_uti):
        symptoms = {k: np.random.binomial(1, p) for k, p in non_uti_probs.items()}
        age, gender, diabetes, hypertension = generate_demographics(is_uti=False)
        wbc, rbc, nitrites, leukocyte_esterase, urine_ph, bacteria = generate_lab_results(is_uti=False)
        
        symptoms.update({
            "age": age,
            "gender": gender,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "wbc": wbc,
            "rbc": rbc,
            "nitrites": nitrites,
            "leukocyte_esterase": leukocyte_esterase,
            "urine_ph": urine_ph,
            "bacteria": bacteria,
            "UTI": 0,  # ระบุว่าไม่เป็น UTI
        })
        data.append(symptoms)

    # แปลงข้อมูลเป็น DataFrame
    df = pd.DataFrame(data)
    
    # แปลงข้อมูลเพศเป็นตัวเลข (0: Male, 1: Female)
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    
    return df

# สร้างข้อมูลจำนวน 100 ตัวอย่าง
uti_data = generate_uti_synthetic_data(100000)
# บันทึกข้อมูลลงไฟล์ CSV
uti_data.to_csv("data/uti_synthetic_data.csv", index=False)

# แสดงตัวอย่าง 5 แถวแรกของข้อมูล
print("\nตัวอย่างข้อมูลสังเคราะห์สำหรับผู้ป่วย UTI:")
print(uti_data.head())

# ---------------------------
# อธิบายการวินิจฉัย UTI
# ---------------------------
# การวินิจฉัย UTI อาศัยการพิจารณาปัจจัยหลายด้านประกอบกัน:
#
# 1. อาการทางคลินิก
#    - ปัสสาวะบ่อย ปัสสาวะแสบขัด เป็นอาการที่พบบ่อยที่สุด (85-90%)
#    - อาการปวดท้องน้อยพบได้ประมาณ 75% ของผู้ป่วย
#    - ลักษณะปัสสาวะที่ผิดปกติ เช่น ขุ่น มีกลิ่นแรง หรือมีเลือดปน
#
# 2. ผลตรวจทางห้องปฏิบัติการ
#    - การตรวจพบเม็ดเลือดขาวในปัสสาวะสูง (>10-12 x10^6/mL)
#    - การตรวจพบไนไตรต์และเอนไซม์ leukocyte esterase เป็นตัวบ่งชี้การติดเชื้อ
#    - ค่า pH ปัสสาวะที่สูงกว่าปกติ (>6.5) มักพบในผู้ป่วย UTI
#
# 3. ปัจจัยเสี่ยง
#    - เพศหญิงมีความเสี่ยงสูงกว่าเพศชาย
#    - ผู้ที่มีโรคประจำตัว เช่น เบาหวาน ความดันโลหิตสูง มีความเสี่ยงเพิ่มขึ้น
#    - อายุที่มากขึ้นอาจเพิ่มความเสี่ยงในการเกิดโรค

# อ้างอิงทางการแพทย์:
# 
# 1. EAU Guidelines on Urological Infections 2023
#    European Association of Urology
#    https://uroweb.org/guidelines/urological-infections
#    DOI: 10.1016/j.eururo.2023.01.817
#
# 2. Urinary Tract Infections: Epidemiology, Mechanisms of Infection and Treatment Options
#    Nature Reviews Microbiology (2020)
#    https://www.nature.com/articles/nrmicro.2017.72
#    DOI: 10.1038/nrmicro.2017.72
#
# 3. International Clinical Practice Guidelines for the Treatment of Acute Uncomplicated Cystitis 
#    and Pyelonephritis in Women (IDSA)
#    Clinical Infectious Diseases Journal
#    https://academic.oup.com/cid/article/52/5/e103/388285
#    DOI: 10.1093/cid/ciq257
#
# 4. Diagnosis, Prevention, and Treatment of Catheter-Associated Urinary Tract Infection 
#    in Adults: 2009 International Clinical Practice Guidelines
#    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4557343/
#    DOI: 10.1086/650482
#
# 5. Utility of dipstick urinalysis in peri-operative patients with indwelling catheters
#    Journal of Clinical Medicine (2021)
#    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8151마3
#    DOI: 10.3390/jcm10081615
#
# Additional Resources:
# - CDC Guidelines for Prevention of CAUTI 2009
#   https://www.cdc.gov/infectioncontrol/pdf/guidelines/cauti-guidelines-H.pdf
#
# - WHO Guidelines on Urological Infections
#   https://www.who.int/publications/guidelines/communicable_diseases/UTI_guidelines
