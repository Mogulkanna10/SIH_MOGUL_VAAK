import json
import hashlib
import os
import time

LOG_FILE = "qa_audit_log.jsonl"

def get_last_entry():
    if not os.path.exists(LOG_FILE):
        return None
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
        if not lines:
            return None
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return None

def compute_hash(entry: dict) -> str:
    # Hash components: timestamp, speaker, command, value, result, prev_hash
    # Ensure stable formatting
    val_str = str(entry.get("value")) if entry.get("value") is not None else "None"
    
    payload = (
        f"{entry['timestamp']}|"
        f"{entry['speaker']}|"
        f"{entry['command']}|"
        f"{val_str}|"
        f"{entry['result']}|"
        f"{entry['prev_hash']}"
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdixigest() # wait typo: hexdigest()

def append_log(speaker: str, command: str, value: float = None, result: str = "SUCCESS"):
    timestamp = str(time.time())
    
    last_entry = get_last_entry()
    prev_hash = last_entry["entry_hash"] if last_entry else "0000000000000000000000000000000000000000000000000000000000000000"
    
    entry = {
        "timestamp": timestamp,
        "speaker": speaker,
        "command": command,
        "value": value,
        "result": result,
        "prev_hash": prev_hash
    }
    
    # Compute the entry hash
    payload = (
        f"{entry['timestamp']}|"
        f"{entry['speaker']}|"
        f"{entry['command']}|"
        f"{entry['value']}|"
        f"{entry['result']}|"
        f"{entry['prev_hash']}"
    )
    entry["entry_hash"] = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry) + "\n")
        
    return entry

def verify_chain(log_file=LOG_FILE) -> bool:
    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist.")
        return True # Empty is valid
        
    prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Line {line_num+1} is not valid JSON.")
                return False
                
            if entry["prev_hash"] != prev_hash:
                print(f"Chain broken at line {line_num+1}: prev_hash mismatch.")
                return False
                
            payload = (
                f"{entry['timestamp']}|"
                f"{entry['speaker']}|"
                f"{entry['command']}|"
                f"{entry['value']}|"
                f"{entry['result']}|"
                f"{entry['prev_hash']}"
            )
            expected_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
            
            if entry["entry_hash"] != expected_hash:
                print(f"Chain broken at line {line_num+1}: entry_hash mismatch.")
                return False
                
            prev_hash = entry["entry_hash"]
            
    print("Hash chain verification PASS.")
    return True
