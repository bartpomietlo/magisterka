"""Report writers: TXT and JSON formats."""

import json
import os
from datetime import datetime
from typing import Optional

from ..types import AnalysisResult


def write_txt_report(
    result: AnalysisResult,
    run_dir: str,
) -> str:
    """Write TXT report for analysis result.
    
    Args:
        result: AnalysisResult to write
        run_dir: Run directory path
        
    Returns:
        Path to written TXT file
    """
    filename = os.path.basename(result.features.path)
    safe_name = "".join(c if c.isalnum() or c in "_- " else "_" for c in filename)
    txt_path = os.path.join(run_dir, f"{safe_name}_report.txt")
    
    lines = []
    lines.append(f"Plik: {filename}")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"WERDYKT: {result.verdict}")
    lines.append(f"Score: {result.final_score:.2f}%")
    lines.append("")
    
    feat = result.features
    lines.append("--- DETALE AI ---")
    lines.append(f"AI Face Score: {_fmt(feat.ai_face_score)}")
    lines.append(f"AI Scene Score: {_fmt(feat.ai_scene_score)}")
    lines.append(f"AI Video Score: {_fmt(feat.ai_video_score)}")
    lines.append("")
    
    lines.append("--- DETALE FORENSIC ---")
    lines.append(f"Jitter: {_fmt(feat.jitter_px, 'px')}")
    lines.append(f"ELA Score: {_fmt(feat.ela_score)}")
    lines.append(f"FFT Score: {_fmt(feat.fft_score)}")
    lines.append(f"Border Artifacts: {_fmt(feat.border_artifacts)}")
    lines.append(f"Face Sharpness: {_fmt(feat.face_sharpness)}")
    lines.append("")
    
    if result.flags:
        lines.append("--- FLAGS ---")
        for flag in result.flags:
            lines.append(f"- {flag}")
        lines.append("")
    
    content = "\n".join(lines)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return txt_path


def write_json_report(
    result: AnalysisResult,
    run_dir: str,
) -> str:
    """Write JSON report for analysis result.
    
    Args:
        result: AnalysisResult to write
        run_dir: Run directory path
        
    Returns:
        Path to written JSON file
    """
    filename = os.path.basename(result.features.path)
    safe_name = "".join(c if c.isalnum() or c in "_- " else "_" for c in filename)
    json_path = os.path.join(run_dir, f"{safe_name}_report.json")
    
    data = {
        "verdict": result.verdict,
        "final_score": result.final_score,
        "features": {
            "media_kind": result.features.media_kind,
            "path": result.features.path,
            "ai_face_score": result.features.ai_face_score,
            "ai_scene_score": result.features.ai_scene_score,
            "ai_video_score": result.features.ai_video_score,
            "jitter_px": result.features.jitter_px,
            "ela_score": result.features.ela_score,
            "fft_score": result.features.fft_score,
            "border_artifacts": result.features.border_artifacts,
            "face_sharpness": result.features.face_sharpness,
            "face_ratio": result.features.face_ratio,
            "face_frames": result.features.face_frames,
            "total_frames": result.features.total_frames,
        },
        "flags": result.flags,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return json_path


def _fmt(value: Optional[float], unit: str = "%") -> str:
    """Format optional float value."""
    if value is None:
        return "N/A"
    return f"{value:.2f}{unit}"
