import numpy as np

# Audio parameters matching shared KWS architecture
SAMPLE_RATE = 16000
FRAME_LENGTH = 480    # 30 ms window
FRAME_STEP = 320      # 20 ms hop
FFT_SIZE = 512
N_MELS = 40
FMIN = 80.0
FMAX = 7500.0

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def create_mel_filterbank():
    mel_min = hz_to_mel(FMIN)
    mel_max = hz_to_mel(FMAX)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((FFT_SIZE + 1) * hz_points / SAMPLE_RATE).astype(int)
    
    filters = np.zeros((N_MELS, FFT_SIZE // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, filters.shape[1])):
            filters[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, filters.shape[1])):
            filters[m - 1, k] = (right - k) / (right - center)
    return filters

MEL_FILTERS = create_mel_filterbank()

def extract_logmel_features(audio_data):
    """
    Extracts 40-bin log-mel filterbank features from a 1.0 second audio array (16000 samples).
    Returns array of shape (49, 40) corresponding to 49 temporal frames and 40 mel frequency bins.
    """
    if len(audio_data) < SAMPLE_RATE:
        # Pad with zeros if shorter than 1 sec
        audio_data = np.pad(audio_data, (0, SAMPLE_RATE - len(audio_data)))
    elif len(audio_data) > SAMPLE_RATE:
        # Truncate if longer than 1 sec
        audio_data = audio_data[:SAMPLE_RATE]
        
    hann = np.hanning(FRAME_LENGTH).astype(np.float32)
    frames = []
    for start in range(0, len(audio_data) - FRAME_LENGTH + 1, FRAME_STEP):
        frame = audio_data[start:start + FRAME_LENGTH] * hann
        frames.append(frame)
        
    frames = np.asarray(frames, dtype=np.float32)
    spectrum = np.fft.rfft(frames, n=FFT_SIZE, axis=1)
    power = (np.abs(spectrum) ** 2) / FFT_SIZE
    mel_energy = np.dot(power, MEL_FILTERS.T)
    mel_energy = np.maximum(mel_energy, 1e-10)
    return np.log(mel_energy).astype(np.float32)
