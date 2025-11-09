"""User profile and auth management for the recommender system."""
import os
import json
import numpy as np
from pathlib import Path

USER_DIR = os.path.join(os.path.dirname(__file__), "data", "users")
os.makedirs(USER_DIR, exist_ok=True)

class UserProfile:
    def __init__(self, username):
        self.username = username
        self.profile_path = os.path.join(USER_DIR, f"{username}.npy")
        self.metadata_path = os.path.join(USER_DIR, f"{username}.json")
        self._profile = None
        self._metadata = None
        self.load()

    def load(self):
        """Load profile embedding and metadata from disk."""
        try:
            if os.path.exists(self.profile_path):
                self._profile = np.load(self.profile_path)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {'click_counts': {}, 'ema_alpha': 0.7}
        except Exception as e:
            print(f"Error loading profile for {self.username}: {e}")
            self._profile = None
            self._metadata = {'click_counts': {}, 'ema_alpha': 0.7}

    def save(self):
        """Save profile embedding and metadata to disk."""
        try:
            if self._profile is not None:
                np.save(self.profile_path, self._profile)
            with open(self.metadata_path, 'w') as f:
                json.dump(self._metadata, f)
        except Exception as e:
            print(f"Error saving profile for {self.username}: {e}")

    @property
    def embedding(self):
        """Get the user's profile embedding."""
        return self._profile

    @embedding.setter
    def embedding(self, value):
        """Set the user's profile embedding."""
        self._profile = value
        self.save()

    def update_with_click(self, job_id, job_embedding):
        """Update profile using exponential moving average of job embeddings."""
        # Update click count
        counts = self._metadata['click_counts']
        counts[str(job_id)] = counts.get(str(job_id), 0) + 1
        
        # EMA update
        alpha = self._metadata['ema_alpha']
        if self._profile is None or np.linalg.norm(self._profile) == 0:
            self._profile = job_embedding
        else:
            self._profile = alpha * job_embedding + (1 - alpha) * self._profile
        
        self.save()

    @property
    def click_counts(self):
        """Get job click counts."""
        return self._metadata['click_counts']

    def get_filters(self):
        """Get user's saved job filters."""
        return self._metadata.get('filters', {})

    def set_filters(self, filters):
        """Save user's job filters."""
        self._metadata['filters'] = filters
        self.save()


def get_user_profile(username):
    """Get or create a UserProfile instance."""
    return UserProfile(username)


def list_users():
    """List all users with saved profiles."""
    users = []
    if os.path.exists(USER_DIR):
        for f in os.listdir(USER_DIR):
            if f.endswith('.npy'):
                users.append(os.path.splitext(f)[0])
    return sorted(users)