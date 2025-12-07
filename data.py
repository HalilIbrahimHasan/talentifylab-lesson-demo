import pandas as pd

# E-Commerce Dataset (Amazon-style)
amazon_data = pd.DataFrame([
    [1001, "2024-01-02", "Electronics", 129.99, 1, "Prime",   "Organic Search", "USA",     "Mobile"],
    [1002, "2024-01-02", "Books",       19.50,  2, "Regular", "Email",          "USA",     "Desktop"],
    [1003, "2024-01-03", "Fashion",     49.90,  1, "Prime",   "Social Ads",     "UK",      "Mobile"],
    [1004, "2024-01-03", "Home",        89.00,  1, "Regular", "Direct",         "Germany", "Desktop"],
    [1005, "2024-01-04", "Electronics", 249.00, 1, "Prime",   "Organic Search", "USA",     "Tablet"],
    [1006, "2024-01-04", "Beauty",      29.99,  3, "Regular", "Social Ads",     "France",  "Mobile"],
    [1007, "2024-01-05", "Books",       12.00,  1, "Prime",   "Direct",         "USA",     "Mobile"],
    [1008, "2024-01-05", "Fashion",     79.00,  2, "Regular", "Email",          "Canada",  "Desktop"],
], columns=[
    "order_id","order_date","category","price","quantity",
    "customer_segment","traffic_source","country","device"
])

# Convert order_date to datetime
amazon_data['order_date'] = pd.to_datetime(amazon_data['order_date'])
amazon_data['revenue'] = amazon_data['price'] * amazon_data['quantity']

# Health Dataset (Labcorp/BCBS-style)
health_data = pd.DataFrame([
    ["P001", 45, "F", 27.5, 130, "Type 2 Diabetes", "Metformin", 7.8, "Labcorp East", 78],
    ["P002", 60, "M", 31.2, 145, "Hypertension",    "ACE Inhibitor", 5.5, "Labcorp West", 70],
    ["P003", 35, "F", 22.0, 118, "Healthy",         "Lifestyle", 5.0, "BCBS Clinic", 90],
    ["P004", 52, "M", 29.8, 150, "Type 2 Diabetes", "Insulin", 8.2, "BCBS Clinic", 72],
    ["P005", 28, "F", 24.4, 110, "Healthy",         "Lifestyle", 4.9, "Labcorp East", 95],
], columns=[
    "patient_id","age","gender","bmi","systolic_bp",
    "diagnosis","treatment","lab_result","hospital","outcome_score"
])

# Education Dataset (School/Exam Scores)
education_data = pd.DataFrame([
    ["S001", 9,  "Math",    88, 5.0, 0.96, "Ms. Green", "Regular"],
    ["S002", 9,  "English", 92, 4.0, 0.98, "Mr. Brown", "Honors"],
    ["S003", 10, "Science", 75, 2.5, 0.90, "Ms. White", "Regular"],
    ["S004", 10, "Math",    64, 1.0, 0.82, "Ms. Green", "Support"],
    ["S005", 11, "History", 81, 3.0, 0.95, "Mr. Black", "Regular"],
    ["S006", 11, "Math",    93, 6.0, 0.99, "Ms. Green", "Honors"],
], columns=[
    "student_id","grade_level","subject",
    "exam_score","homework_hours","attendance_rate",
    "teacher","program"
])

# Finance Dataset (Freddie Mac/Bank of America Mortgage)
finance_data = pd.DataFrame([
    ["L001", "Bank of America", 4.2, 30, 350000, 1710, 95000, 0.28, 740, "East",    0],
    ["L002", "Freddie Mac",     3.8, 15, 220000, 1600, 120000,0.22, 780, "West",    0],
    ["L003", "Bank of America", 5.1, 30, 180000, 980,  65000, 0.35, 690, "South",   1],
    ["L004", "Freddie Mac",     4.5, 20, 260000, 1650, 85000, 0.31, 710, "Midwest", 0],
    ["L005", "Bank of America", 3.9, 25, 300000, 1580,105000,0.26, 760, "East",    0],
], columns=[
    "loan_id","bank","interest_rate","term_years","principal",
    "monthly_payment","income","dti_ratio","credit_score",
    "region","defaulted"
])

