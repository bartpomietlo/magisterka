"""Decision policies and thresholds."""

from typing import Dict, Optional


# Default policies (can be overridden by calibration)
POLICIES: Dict[str, Dict[str, float]] = {
    "high_precision": {
        "fake_min": 70.0,
        "real_max": 25.0,
    },
    "balanced": {
        "fake_min": 60.0,
        "real_max": 30.0,
    },
    "high_recall": {
        "fake_min": 50.0,
        "real_max": 35.0,
    },
}


def get_verdict(
    score: Optional[float],
    policy_name: str = "high_precision",
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Determine verdict from score and policy.
    
    Args:
        score: Detection score (0-100)
        policy_name: One of "high_precision", "balanced", "high_recall"
        custom_thresholds: Optional dict with 'fake_min' and 'real_max' keys
        
    Returns:
        Verdict string
    """
    if score is None:
        return "NIEPEWNE / BRAK DANYCH"
    
    if custom_thresholds:
        thresholds = custom_thresholds
    else:
        thresholds = POLICIES.get(policy_name, POLICIES["high_precision"])
    
    fake_min = thresholds.get("fake_min", 70.0)
    real_max = thresholds.get("real_max", 25.0)
    
    if score >= fake_min:
        return "FAKE (PRAWDOPODOBNE)"
    elif score <= real_max:
        return "REAL (PRAWDOPODOBNE)"
    else:
        return "NIEPEWNE / GREY ZONE"


def combine_verdicts(
    ai_verdict: str,
    deepfake_verdict: str,
) -> str:
    """Combine AI detector and deepfake detector verdicts.
    
    Args:
        ai_verdict: Verdict from AI detector
        deepfake_verdict: Verdict from deepfake detector
        
    Returns:
        Combined verdict
    """
    av = ai_verdict.upper()
    dv = deepfake_verdict.upper()
    
    # If either says FAKE, combined is FAKE
    if "FAKE" in av or "FAKE" in dv:
        return "FAKE (PRAWDOPODOBNE)"
    
    # If both say REAL, combined is REAL
    if "REAL" in av and "REAL" in dv:
        return "REAL (PRAWDOPODOBNE)"
    
    # Otherwise uncertain
    return "NIEPEWNE / GREY ZONE"
