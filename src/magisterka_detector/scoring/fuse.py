"""Signal fusion - combine multiple detector scores."""

from typing import Dict, List, Optional


def fuse_scores(
    face_score: Optional[float],
    scene_score: Optional[float],
    video_score: Optional[float],
    weights: Optional[Dict[str, float]] = None,
    suppress_scene: bool = False,
) -> Optional[float]:
    """Fuse AI detector scores with configurable weights.
    
    Args:
        face_score: HF face/subject score (0-100)
        scene_score: HF scene score (0-100)
        video_score: VideoMAE score (0-100)
        weights: Dict with 'face', 'scene', 'video' weights (default: 0.35, 0.15, 0.50)
        suppress_scene: If True, ignore scene score
        
    Returns:
        Weighted average score (0-100), or None if all inputs None
    """
    if weights is None:
        weights = {"face": 0.35, "scene": 0.15, "video": 0.50}
    
    items = []
    if face_score is not None:
        items.append((face_score, weights.get("face", 0.35)))
    if scene_score is not None and not suppress_scene:
        items.append((scene_score, weights.get("scene", 0.15)))
    if video_score is not None:
        items.append((video_score, weights.get("video", 0.50)))
    
    if not items:
        return None
    
    weighted_sum = sum(score * weight for score, weight in items)
    total_weight = sum(weight for _, weight in items)
    
    if total_weight <= 0:
        return None
    
    return max(0.0, min(100.0, weighted_sum / total_weight))


def fuse_forensic(
    jitter: Optional[float],
    ela: Optional[float],
    fft: Optional[float],
    border: Optional[float],
    sharpness: Optional[float],
) -> Optional[float]:
    """Fuse forensic signals into single score.
    
    All inputs are normalized to 0-100 scale before calling this.
    
    Returns:
        Average forensic score (0-100), or None if all inputs None
    """
    signals = [s for s in [jitter, ela, fft, border, sharpness] if s is not None]
    if not signals:
        return None
    return sum(signals) / len(signals)


def compute_deepfake_score(
    ai_face: Optional[float],
    ai_video: Optional[float],
    forensic_score: Optional[float],
    gate_threshold: float = 60.0,
) -> Optional[float]:
    """Compute deepfake-specific score combining AI + forensic.
    
    Args:
        ai_face: Face AI score
        ai_video: Video AI score
        forensic_score: Fused forensic score
        gate_threshold: Threshold for forensic weight mixing
        
    Returns:
        Combined deepfake score (0-100)
    """
    # Base: average of face + video
    base_signals = [s for s in [ai_face, ai_video] if s is not None]
    if not base_signals:
        return None
    
    base_score = sum(base_signals) / len(base_signals)
    
    # If no forensic, return base
    if forensic_score is None:
        return base_score
    
    # Below gate: forensic gets 15% weight
    # Above gate: forensic gets 30% weight
    if base_score < gate_threshold:
        return base_score * 0.85 + forensic_score * 0.15
    else:
        return base_score * 0.70 + forensic_score * 0.30
