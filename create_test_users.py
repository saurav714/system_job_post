# Initialize some test users
import os
import numpy as np
from user_profiles import UserProfile

# Create test users with different interests
def create_test_users():
    # Python Developer profile
    python_dev = UserProfile('python_dev')
    python_dev.set_filters({
        'locations': ['San Francisco, CA', 'Remote'],
        'seniority': 'Mid-Level',
        'min_salary': 120000,
        'max_salary': 180000
    })
    python_dev.embedding = np.array([0.8, 0.9, 0.7, 0.2, 0.1, 0.3, 0.8, 0.9])  # Python-heavy
    python_dev.save()

    # Data Scientist profile
    data_scientist = UserProfile('data_scientist')
    data_scientist.set_filters({
        'locations': ['New York, NY', 'Boston, MA'],
        'seniority': 'Senior',
        'min_salary': 140000,
        'max_salary': 200000
    })
    data_scientist.embedding = np.array([0.7, 0.9, 0.9, 0.1, 0.2, 0.8, 0.7, 0.6])  # ML/Data-heavy
    data_scientist.save()

    # Frontend Developer profile
    frontend_dev = UserProfile('frontend_dev')
    frontend_dev.set_filters({
        'locations': ['Remote'],
        'seniority': 'Mid-Level',
        'min_salary': 100000,
        'max_salary': 160000
    })
    frontend_dev.embedding = np.array([0.2, 0.3, 0.1, 0.9, 0.8, 0.2, 0.3, 0.1])  # Frontend-heavy
    frontend_dev.save()

if __name__ == '__main__':
    create_test_users()
    print("✅ Created test users: python_dev, data_scientist, frontend_dev")
    print("You can now select these users in the app's login dropdown.")