import json
import os
import numpy as np
from typing import List, Tuple, Dict

ENROLLMENT_FILE = os.path.join(os.path.dirname(__file__), "enrolled_speakers.json")

class SpeakerIdentifier:
    def __init__(self, enrollment_file: str = ENROLLMENT_FILE):
        self.enrollment_file = enrollment_file
        self.enrolled_profiles: Dict[str, List[float]] = {}
        self.load_profiles()

    def load_profiles(self):
        if os.path.exists(self.enrollment_file):
            with open(self.enrollment_file, 'r') as f:
                self.enrolled_profiles = json.load(f)
        else:
            self.enrolled_profiles = {}

    def save_profiles(self):
        with open(self.enrollment_file, 'w') as f:
            json.dump(self.enrolled_profiles, f, indent=4)

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def identify(self, vector: List[float], threshold: float = 0.5) -> Tuple[str, float]:
        if not vector or not self.enrolled_profiles:
            return "UNKNOWN", 0.0

        best_match = "UNKNOWN"
        highest_sim = -1.0

        for speaker, profile_vector in self.enrolled_profiles.items():
            sim = self.cosine_similarity(vector, profile_vector)
            if sim > highest_sim:
                highest_sim = sim
                best_match = speaker

        if highest_sim >= threshold:
            return best_match, highest_sim
        else:
            return "UNKNOWN", highest_sim
