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

WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90"
}

def words_to_numbers(text: str) -> str:
    """Converts spoken number words in a string into digit strings."""
    words = text.lower().split()
    result = []
    i = 0
    while i < len(words):
        w = words[i]
        # Handle compound numbers like "forty five"
        if w in ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"] and i + 1 < len(words) and words[i+1] in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]:
            val = int(WORD_TO_NUM[w]) + int(WORD_TO_NUM[words[i+1]])
            result.append(str(val))
            i += 2
        elif w in WORD_TO_NUM:
            result.append(WORD_TO_NUM[w])
            i += 1
        else:
            result.append(w)
            i += 1
    return " ".join(result)

def parse_command(transcript: str, speaker_identity: str) -> Tuple[Optional[str], Optional[float], bool]:
    """
    Parses a transcript to find a matching whitelisted command.
    Returns (intent, slot_value, is_authorized).
    """
    # Authorization logic: UNKNOWN is strictly unauthorized.
    is_authorized = speaker_identity != "UNKNOWN"
    
    transcript = transcript.strip()
    transcript_normalized = words_to_numbers(transcript)
    
    # Priority check for exact/strict matches
    for intent in ["CONFIRM", "CANCEL"]:
        match = re.match(COMMAND_WHITELIST[intent], transcript_normalized)
        if match:
            return (intent, None, is_authorized)
            
    # Substring / pattern matching
    best_intent = None
    slot_val = None
    
    for intent, pattern in COMMAND_WHITELIST.items():
        if intent in ["CONFIRM", "CANCEL"]: continue
        
        match = re.match(pattern, transcript_normalized)
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
