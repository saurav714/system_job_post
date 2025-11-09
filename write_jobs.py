# write_jobs.py
import csv

jobs = [
    ['job_id', 'title', 'company', 'skills', 'location', 'seniority', 'salary_range', 'applied', 'viewed', 'clicked'],
    ['1', 'Software Engineer', 'Acme Corp', 'python, pandas, sql, aws', 'San Francisco, CA', 'Mid-Level', '120000-150000', 'False', 'False', 'False'],
    ['2', 'Data Scientist', 'DataWorks', 'python, pandas, sklearn, ml', 'New York, NY', 'Senior', '140000-180000', 'False', 'False', 'False'],
    ['3', 'ML Engineer', 'ModelOps', 'python, pytorch, kubernetes', 'Seattle, WA', 'Senior', '150000-200000', 'False', 'False', 'False'],
    ['4', 'Frontend Developer', 'Webify', 'javascript, react, css', 'Remote', 'Junior', '80000-100000', 'False', 'False', 'False'],
    ['5', 'Backend Developer', 'ServeIT', 'python, django, postgresql', 'Austin, TX', 'Mid-Level', '110000-140000', 'False', 'False', 'False'],
    ['6', 'DevOps Engineer', 'InfraPros', 'docker, kubernetes, aws', 'Remote', 'Senior', '130000-170000', 'False', 'False', 'False'],
    ['7', 'Product Manager', 'Prodify', 'roadmapping, communication, analytics', 'Boston, MA', 'Senior', '140000-180000', 'False', 'False', 'False'],
    ['8', 'Data Analyst', 'Insightful', 'sql, tableau, excel', 'Chicago, IL', 'Junior', '70000-90000', 'False', 'False', 'False'],
    ['9', 'AI Researcher', 'DeepMindSim', 'python, tensorflow, research', 'San Francisco, CA', 'Senior', '160000-200000', 'False', 'False', 'False'],
    ['10', 'QA Engineer', 'Testify', 'testing, selenium, python', 'Remote', 'Mid-Level', '90000-120000', 'False', 'False', 'False'],
    ['11', 'Full Stack Developer', 'TechStack', 'javascript, python, react, node', 'New York, NY', 'Mid-Level', '130000-160000', 'False', 'False', 'False'],
    ['12', 'Cloud Architect', 'CloudNine', 'aws, azure, terraform', 'Remote', 'Senior', '160000-200000', 'False', 'False', 'False'],
    ['13', 'Data Engineer', 'DataFlow', 'python, spark, kafka', 'Seattle, WA', 'Mid-Level', '140000-170000', 'False', 'False', 'False'],
    ['14', 'UI/UX Designer', 'DesignCo', 'figma, sketch, user research', 'San Francisco, CA', 'Mid-Level', '110000-140000', 'False', 'False', 'False'],
    ['15', 'Security Engineer', 'SecureIT', 'security, penetration testing, python', 'Boston, MA', 'Senior', '150000-190000', 'False', 'False', 'False'],
    ['16', 'Mobile Developer', 'AppWorks', 'react native, flutter, mobile', 'Austin, TX', 'Mid-Level', '120000-150000', 'False', 'False', 'False'],
    ['17', 'Technical Writer', 'DocuTech', 'documentation, api, markdown', 'Remote', 'Junior', '75000-95000', 'False', 'False', 'False'],
    ['18', 'System Administrator', 'SysOps', 'linux, ansible, networking', 'Chicago, IL', 'Mid-Level', '100000-130000', 'False', 'False', 'False'],
    ['19', 'Deep Learning Engineer', 'AILabs', 'pytorch, tensorflow, computer vision', 'San Francisco, CA', 'Senior', '170000-210000', 'False', 'False', 'False'],
    ['20', 'Blockchain Developer', 'ChainTech', 'solidity, web3, ethereum', 'Remote', 'Senior', '140000-180000', 'False', 'False', 'False']
]

with open('data/jobs.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(jobs)

print("✅ Created jobs.csv with UTF-8 encoding")