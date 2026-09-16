import re
from typing import Dict, Any, Optional, Tuple

COMMAND_WHITELIST = {
    "NEXT_STEP": r"(?i).*\b(next|advance|forward|move on|continue)\b.*\b(step|instruction|procedure)?\b.*",
    "PREV_STEP": r"(?i).*\b(previous|back|go back)\b.*\b(step|instruction)?\b.*",
    "REPEAT_STEP": r"(?i).*\b(repeat|say again)\b.*\b(step|instruction)?\b.*",
    "RECORD_VALUE": r"(?i).*\b(record|log|enter|save)\b.*\b(value)?\b\s+(?P<val>\d+(\.\d+)?).*",
    "CONFIRM": r"(?i)^\s*(yes|confirm|affirmative|ok|okay)\s*$",
    "CANCEL": r"(?i)^\s*(no|cancel|stop|abort|negative)\s*$",
    "START_PROCEDURE": r"(?i).*\b(start|begin|engine)\b.*",
    "END_PROCEDURE": r"(?i).*\b(end|finish|stop|stop engine)\b.*",
    "PAUSE_PROCEDURE": r"(?i).*\b(pause|hold)\b.*\b(procedure|process)?\b.*",
    "RESUME_PROCEDURE": r"(?i).*\b(resume|continue)\b.*\b(procedure|process)?\b.*",
    "REPORT_ISSUE": r"(?i).*\b(report|flag)\b.*\b(issue|problem|error)\b.*",
    "REQUEST_HELP": r"(?i).*\b(help|assist|assistance)\b.*",
    "OPEN_VALVE": r"(?i).*\b(open)\b.*\b(valve)\b.*",
    "CLOSE_VALVE": r"(?i).*\b(close|shut)\b.*\b(valve)\b.*",
    "SET_TEMPERATURE": r"(?i).*\b(set)\b.*\b(temperature|temp)\b.*\b(to)?\s*(?P<val>\d+(\.\d+)?).*",
    "SET_PRESSURE": r"(?i).*\b(set)\b.*\b(pressure)\b.*\b(to)?\s*(?P<val>\d+(\.\d+)?).*",
    "CHECK_STATUS": r"(?i).*\b(check|what is)\b.*\b(status|state)\b.*",
    "LOCK_SYSTEM": r"(?i).*\b(lock)\b.*\b(system|terminal|screen)\b.*",
    "UNLOCK_SYSTEM": r"(?i).*\b(unlock)\b.*\b(system|terminal|screen)\b.*",
    "EMERGENCY_STOP": r"(?i).*\b(emergency|e-stop|fire|halt)\b.*"
}

def parse_command(transcript: str, speaker_identity: str) -> Tuple[Optional[str], Optional[float], bool]:
    """
    Parses a transcript to find a matching whitelisted command.
    Returns (intent, slot_value, is_authorized).
    """
    # Authorization logic: UNKNOWN is strictly unauthorized.
    is_authorized = speaker_identity != "UNKNOWN"
    
    transcript = transcript.strip()
    
    # Priority check for exact/strict matches
    for intent in ["CONFIRM", "CANCEL"]:
        match = re.match(COMMAND_WHITELIST[intent], transcript)
        if match:
            return (intent, None, is_authorized)
            
    # Substring / pattern matching
    best_intent = None
    slot_val = None
    
    for intent, pattern in COMMAND_WHITELIST.items():
        if intent in ["CONFIRM", "CANCEL"]: continue
        
        match = re.match(pattern, transcript)
        if match:
            best_intent = intent
            if "val" in match.groupdict():
                try:
                    slot_val = float(match.group("val"))
                except (ValueError, TypeError):
                    pass
            break
            
    if best_intent:
        return (best_intent, slot_val, is_authorized)
        
    return (None, None, False)
