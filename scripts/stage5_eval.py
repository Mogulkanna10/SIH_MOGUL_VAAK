#!/usr/bin/env python3
import glob
import os
import queue
import sys
import threading
import time
import wave

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pi_deploy_rm.asr_client import VoskConnection, stream_utterance, SAMPLE_RATE, CHUNK_FRAMES
from pi_deploy_rm.speaker_id import SpeakerIdentifier

SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "split")

def run_test(connection, identifier, wav_file):
    q = queue.Queue()
    t1 = time.time()
    
    def _feed():
        with wave.open(wav_file, "rb") as wf:
            while True:
                frames = wf.readframes(CHUNK_FRAMES)
                if not frames:
                    break
                q.put(frames)
                time.sleep(CHUNK_FRAMES / SAMPLE_RATE)
        q.put(None)
        
    feeder = threading.Thread(target=_feed, daemon=True)
    feeder.start()
    
    rec = stream_utterance(q, connection, t1_keyword_end=t1)
    feeder.join()
    
    similarities = {}
    if rec.success and rec.spk_vector:
        for enrolled_spk, profile in identifier.enrolled_profiles.items():
            sim = identifier.cosine_similarity(rec.spk_vector, profile)
            similarities[enrolled_spk] = sim
            
    best_match, highest_sim = identifier.identify(rec.spk_vector, threshold=0.45)
    return rec.transcript, similarities, best_match, rec.raw_final_json, rec.t4_final_transcript

def main():
    conn = VoskConnection("ws://localhost:2700")
    identifier = SpeakerIdentifier()
    
    if not identifier.enrolled_profiles:
        print("Error: No enrolled profiles found. Run enroll_speakers.py first.")
        sys.exit(1)
        
    test_speakers = ["speaker_01", "speaker_02", "speaker_03", "speaker_10"]
    
    print("=" * 80)
    print("STAGE 5: SERVER-SIDE ASR + SPEAKER ID - RAW SIMILARITY MATRIX")
    print("=" * 80)
    
    # Header
    enrolled_names = list(identifier.enrolled_profiles.keys())
    concurrency_proof_shown = False
    header = f"{'Test File':<30} | {'Transcript':<15} | "
    for name in enrolled_names:
        header += f"{name:>12} | "
    header += f"{'Best Match':<12}"
    print(header)
    print("-" * 80)
    
    for spk in test_speakers:
        # Use files 11 to 20
        files = []
        for i in range(11, 21):
            pattern = os.path.join(SPLIT_DIR, "*", "positive", f"{spk}_vaak_{i:04d}.wav")
            matches = glob.glob(pattern)
            if matches:
                files.extend(matches)
                
        for wav_file in sorted(files):
            transcript, similarities, best_match, raw_json, t4 = run_test(conn, identifier, wav_file)
            filename = os.path.basename(wav_file)
            
            if not concurrency_proof_shown and similarities:
                print("=" * 80)
                print("CONCURRENCY PROOF (First successful detection)")
                print("=" * 80)
                print(f"File: {filename}")
                print(f"Timestamp of final response (T4): {t4}")
                try:
                    import json
                    parsed = json.loads(raw_json)
                    print(f"Raw JSON Keys present: {list(parsed.keys())}")
                    print(f"Transcript in response: {parsed.get('text', '')}")
                    print(f"Speaker vector length: {len(parsed.get('spk', []))}")
                    assert "text" in parsed and "spk" in parsed
                    print("ASSERTION PASSED: Both 'text' and 'spk' were received in the single final WebSocket response payload.")
                except Exception as e:
                    print(f"Proof failed: {e}")
                print("=" * 80 + "\n")
                concurrency_proof_shown = True
            
            if not similarities:
                print(f"{filename:<30} | {'FAIL':<15} | {'No spk vector'}")
                continue
                
            # Format row
            # Limit transcript length
            trunc_transcript = transcript[:13] + ".." if len(transcript) > 15 else transcript
            row = f"{filename:<30} | {trunc_transcript:<15} | "
            for name in enrolled_names:
                sim = similarities.get(name, 0.0)
                row += f"{sim:12.4f} | "
            row += f"{best_match:<12}"
            print(row)
            
        print("-" * 80)
        
    conn.close()

if __name__ == "__main__":
    main()
