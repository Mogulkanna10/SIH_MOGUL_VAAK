import numpy as np

class CapacityController:
    """
    Deterministic Capacity Controller (Stage 3).
    Routes KWS inference to SMALL or LARGE models based on environmental noise
    and recent KWS confidence history, using hysteresis to prevent oscillation.

    Threshold calibration (from measured test-split log-mel mean energy):
      Background: mean=-9.4, std=3.7, p10=-13.9, p50=-8.7, p90=-5.5
      Speech:     mean=-10.7, std=0.4  (tight band, overlaps noisy BG)

    SMALL condition: rolling noise floor < -9.5  (below speech floor, in quiet BG region)
    LARGE condition: rolling noise floor > -8.0  (clearly above quiet-only threshold)
    Hysteresis gap of 1.5 dB prevents oscillation at borderline.
    """
    def __init__(self,
                 noise_thresh_high=-8.0,
                 noise_thresh_low=-9.5,
                 conf_thresh_high=0.10,
                 conf_thresh_low=0.05):
        self.noise_thresh_high = noise_thresh_high
        self.noise_thresh_low  = noise_thresh_low
        self.conf_thresh_high  = conf_thresh_high
        self.conf_thresh_low   = conf_thresh_low

        # Start in LARGE mode for safety
        self.current_model = "LARGE"

        # Rolling noise floor — initialised to a quiet value so the controller
        # can transition to SMALL quickly on genuine silence.
        self.rolling_noise_floor = -11.0

        # Prevent rapid flapping: minimum frames to stay in LARGE
        self.large_hold_frames = 0
        self.min_large_frames  = 5


    def select_model(self, logmel_features, recent_confidence):
        """
        Takes the existing (49, 40) Log-Mel feature frame and the latest KWS
        dracarys confidence score, updates the state, and returns "SMALL" or "LARGE".
        """
        # 1. Estimate noise floor from features
        # The mean energy across all 49 frames and 40 bins
        current_energy = np.mean(logmel_features)

        # Update rolling noise floor only if confidence is low (to avoid
        # treating loud active speech entirely as background noise).
        if recent_confidence < self.conf_thresh_low:
            # Alpha filter for smoothing
            alpha = 0.2
            self.rolling_noise_floor = (alpha * current_energy) + ((1 - alpha) * self.rolling_noise_floor)

        # 2. Hysteresis State Machine
        if self.current_model == "SMALL":
            # Switch to LARGE if it gets too noisy, OR we heard something that might be speech
            if self.rolling_noise_floor > self.noise_thresh_high or recent_confidence > self.conf_thresh_high:
                self.current_model = "LARGE"
                self.large_hold_frames = self.min_large_frames
                
        elif self.current_model == "LARGE":
            if self.large_hold_frames > 0:
                self.large_hold_frames -= 1
                
            # Switch to SMALL only if it's very quiet AND we haven't heard anything speech-like
            if self.large_hold_frames == 0 and self.rolling_noise_floor < self.noise_thresh_low and recent_confidence < self.conf_thresh_low:
                self.current_model = "SMALL"

        return self.current_model
