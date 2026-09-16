#!/usr/bin/env python3
"""
gate6_measure.py — Gate 6 CPU & RAM Measurement Script (Stage 2: Dual Interpreters).
Run this in a SEPARATE terminal while live_kws.py is running.
Samples CPU and RSS every second for 30 seconds, then reports.
From Stage 2 onwards live_kws.py allocates BOTH interpreters (large + small) at startup,
so this measurement captures the combined resident set of both tensor arenas.

Usage:
  1. Start live_kws.py in Terminal 1
  2. In Terminal 2: python3 gate6_measure.py
"""

import os
import sys
import time
import subprocess

def find_live_kws_pid():
    """Find the PID of the running live_kws.py process."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'live_kws.py'],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split('\n')
        # Filter out our own PID
        pids = [p for p in pids if p and int(p) != os.getpid()]
        if pids:
            return int(pids[0])
    except Exception:
        pass
    return None

def get_rss_kb(pid):
    """Get RSS memory in KB for a given PID."""
    try:
        result = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(pid)],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except Exception:
        return 0

def main():
    print("=" * 60)
    print("GATE 6 — CPU & RAM MEASUREMENT")
    print("=" * 60)
    
    pid = find_live_kws_pid()
    if pid is None:
        print("\nERROR: live_kws.py is not running!")
        print("Start it first in another terminal:")
        print("  python3 live_kws.py")
        return
        
    print(f"\nFound live_kws.py process: PID {pid}")
    print("Sampling CPU and RSS every 1 second for 30 seconds...")
    print("Keep the room QUIET (idle listening mode).\n")
    
    try:
        import psutil
        proc = psutil.Process(pid)
    except ImportError:
        print("ERROR: psutil not installed. Run: pip install psutil")
        return
    except psutil.NoSuchProcess:
        print(f"ERROR: PID {pid} no longer exists.")
        return
    
    cpu_samples = []
    rss_samples = []
    
    for i in range(30):
        try:
            cpu = proc.cpu_percent(interval=1.0)
            rss_kb = get_rss_kb(pid)
            cpu_samples.append(cpu)
            rss_samples.append(rss_kb)
            print(f"  Sample {i+1:2d}/30: CPU={cpu:5.1f}%  RSS={rss_kb} KB ({rss_kb/1024:.1f} MB)")
        except Exception as e:
            print(f"  Sample {i+1:2d}/30: Error — {e}")
    
    print("\n" + "=" * 60)
    print("GATE 6 RESULTS — CPU & RAM (Idle Listening)")
    print("=" * 60)
    
    if cpu_samples:
        mean_cpu = sum(cpu_samples) / len(cpu_samples)
        peak_cpu = max(cpu_samples)
        print(f"  CPU Mean:  {mean_cpu:.1f}% (Target: <10%)")
        print(f"  CPU Peak:  {peak_cpu:.1f}%")
        print(f"  CPU Status: {'PASSED ✅' if mean_cpu < 10.0 else 'FAILED ❌'}")
    
    if rss_samples:
        rss_val = rss_samples[-1]  # Most stable reading
        print(f"\n  RSS Memory: {rss_val} KB ({rss_val/1024:.1f} MB)")
        print(f"  RAM Status: INFORMATIONAL ONLY")
        print(f"  Note: RSS includes Python runtime + numpy + tflite-runtime overhead,")
        print(f"        not just the 64KB tensor arena. This is expected and normal.")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
