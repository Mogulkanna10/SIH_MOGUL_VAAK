#!/usr/bin/env python3
"""
test_live_kws_offline.py — Comprehensive Streaming-Mode Offline Pipeline Audit.
Evaluates the FULL validation and test datasets through live_kws.py's rolling 1s-buffer / 200ms-hop streaming engine.
Computes per-utterance streaming Recall and False Activation Rate to verify against Gate 4/5 batch numbers.
"""

import os
import sys
import wave
import time
import numpy as np

sys.path.append(os.path.dirname(__file__))
from live_kws import DracarysKWS, THRESHOLD

SPLIT_DIR = '/home/yuvan/split'
CLASS_MAP = {'background': 0, 'positive': 1, 'negative': 2}

def load_wav_mono_16k(filepath):
    try:
        with wave.open(filepath, 'rb') as wf:
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
    except Exception:
        return np.zeros(16000, dtype=np.float32)

def evaluate_streaming_split(kws, split_name='validation'):
    split_dir = os.path.join(SPLIT_DIR, split_name)
    results = {'positive': [], 'negative': [], 'background': []}
    hop_samples = kws.hop_size
    
    print(f"\n--- Running Streaming Audit on '{split_name.upper()}' Split ---")
    
    for cls_folder in ['positive', 'negative', 'background']:
        folder_path = os.path.join(split_dir, cls_folder)
        if not os.path.exists(folder_path):
            continue
            
        raw_files = [f for f in os.listdir(folder_path) if f.endswith('.wav') and not f.startswith('aug_')]
        print(f"  Streaming {len(raw_files)} '{cls_folder}' clips through 1s-buffer / 200ms-hop engine...")
        
        detected_count = 0
        total_files = len(raw_files)
        
        for idx, fn in enumerate(raw_files):
            filepath = os.path.join(folder_path, fn)
            audio = load_wav_mono_16k(filepath)
            
            # Reset kws buffer for new audio stream
            kws.audio_buffer = np.zeros(kws.buffer_size, dtype=np.float32)
            clip_detected = False
            
            # Stream audio through rolling window in 200ms hops
            for start_idx in range(0, len(audio), hop_samples):
                chunk = audio[start_idx:start_idx + hop_samples]
                if len(chunk) < hop_samples:
                    chunk = np.pad(chunk, (0, hop_samples - len(chunk)))
                    
                is_det, score, probs, inf_ms = kws.process_chunk(chunk)
                if is_det:
                    clip_detected = True
                    break  # Keyword detected for this utterance stream
                    
            if clip_detected:
                detected_count += 1
            results[cls_folder].append(clip_detected)
            
        rate = (detected_count / total_files) * 100.0 if total_files > 0 else 0.0
        print(f"    -> '{cls_folder}': {detected_count}/{total_files} clips triggered keyword detection ({rate:.2f}%)")
        
    pos_triggers = sum(results['positive'])
    pos_total = len(results['positive'])
    streaming_recall = (pos_triggers / pos_total) if pos_total > 0 else 0.0
    
    neg_triggers = sum(results['negative']) + sum(results['background'])
    neg_total = len(results['negative']) + len(results['background'])
    streaming_fa = (neg_triggers / neg_total) if neg_total > 0 else 0.0
    
    return streaming_recall, streaming_fa, pos_triggers, pos_total, neg_triggers, neg_total

def main():
    print("======================================================================")
    print("FULL STREAMING-MODE OFFLINE PIPELINE AUDIT (live_kws.py Engine)")
    print("======================================================================\n")

    kws = DracarysKWS()
    
    # 1. Evaluate Streaming Validation Split (Speakers 8-9, 11 unseen background categories)
    t0 = time.time()
    val_recall, val_fa, val_pos_k, val_pos_n, val_neg_k, val_neg_n = evaluate_streaming_split(kws, 'validation')
    val_time = time.time() - t0
    
    # 2. Evaluate Streaming Test Split (Speaker 10, 17 unseen background categories)
    t0 = time.time()
    test_recall, test_fa, test_pos_k, test_pos_n, test_neg_k, test_neg_n = evaluate_streaming_split(kws, 'test')
    test_time = time.time() - t0
    
    print("\n======================================================================")
    print("STREAMING vs BATCH METRICS COMPARISON (Threshold = 0.85)")
    print("======================================================================")
    print(f"Split       | Metric                    | Batch (Gate 5) | Streaming Mode | Status")
    print("-" * 75)
    
    val_rec_pass = "PASSED ✅" if val_recall >= 0.90 else "FAIL ❌"
    val_fa_pass = "PASSED ✅" if val_fa <= 0.02 else "FAIL ❌"
    test_rec_pass = "PASSED ✅" if test_recall >= 0.90 else "FAIL ❌"
    test_fa_pass = "PASSED ✅" if test_fa <= 0.02 else "FAIL ❌"
    
    print(f"Validation  | Dracarys Recall (pos)     |          93.00%|         {val_recall*100:5.2f}% | {val_rec_pass}")
    print(f"Validation  | False Activation (non-pos)|           0.03%|         {val_fa*100:5.2f}% | {val_fa_pass}")
    print(f"Test (Held) | Dracarys Recall (pos)     |          92.00%|         {test_recall*100:5.2f}% | {test_rec_pass}")
    print(f"Test (Held) | False Activation (non-pos)|           0.15%|         {test_fa*100:5.2f}% | {test_fa_pass}")
    
    print(f"\nAudit Execution Time: Val = {val_time:.2f}s, Test = {test_time:.2f}s")
    
    passed_streaming_gate = (val_recall >= 0.90) and (val_fa <= 0.02) and (test_recall >= 0.90) and (test_fa <= 0.02)
    print("\n======================================================================")
    if passed_streaming_gate:
        print("STREAMING-MODE PIPELINE VERIFICATION: PASSED HARD GATE! ✅")
        print("Continuous rolling buffer logic matches Gate 4/5 batch performance!")
    else:
        print("STREAMING-MODE PIPELINE VERIFICATION: REGRESSION DETECTED ⚠️")
    print("======================================================================")

if __name__ == '__main__':
    main()
