import os
import pandas as pd

# Sample job data
jobs_data = {
    'job_id': range(1, 6),
    'title': ['Software Engineer', 'Data Scientist', 'ML Engineer', 'Frontend Developer', 'Backend Developer'],
    'company': ['Acme Corp', 'DataWorks', 'ModelOps', 'Webify', 'ServeIT'],
    'skills': [
        'python, pandas, sql, aws',
        'python, pandas, sklearn, ml',
        'python, pytorch, kubernetes',
        'javascript, react, css',
        'python, django, postgresql'
    ],
    'location': ['San Francisco, CA', 'New York, NY', 'Seattle, WA', 'Remote', 'Austin, TX'],
    'seniority': ['Mid-Level', 'Senior', 'Senior', 'Junior', 'Mid-Level'],
    'salary_range': ['120000-150000', '140000-180000', '150000-200000', '80000-100000', '110000-140000'],
    'applied': [False] * 5,
    'viewed': [False] * 5,
    'clicked': [False] * 5
}

# Create DataFrame
df = pd.DataFrame(jobs_data)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV with UTF-8 encoding
df.to_csv('data/jobs.csv', index=False, encoding='utf-8')
print("✅ Created jobs.csv successfully")