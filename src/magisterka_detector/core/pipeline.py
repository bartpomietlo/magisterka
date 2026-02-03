"""Main analysis pipeline - orchestrates extractors, scoring, reporting."""

import os
from typing import Optional

from ..types import (
    AnalyzeOptions,
    AnalysisFeatures,
    AnalysisResult,
    ProgressCb,
    CancelCb,
)
from ..extractors.video_frames import detect_media_kind, extract_frames, load_image
from ..scoring.fuse import fuse_scores, compute_deepfake_score
from ..scoring.policy import get_verdict, combine_verdicts
from ..reporting.writers import write_txt_report, write_json_report


def run_pipeline(
    path: str,
    opts: AnalyzeOptions,
    progress: Optional[ProgressCb],
    cancel: Optional[CancelCb],
    run_dir: str,
) -> AnalysisResult:
    """Run the full analysis pipeline.
    
    Args:
        path: Path to media file
        opts: Analysis options
        progress: Progress callback
        cancel: Cancel callback
        run_dir: Run directory for reports
        
    Returns:
        AnalysisResult
    """
    # Detect media type
    media_kind = detect_media_kind(path)
    
    # Initialize features
    features = AnalysisFeatures(
        media_kind=media_kind,
        path=os.path.abspath(path),
    )
    
    flags = []
    
    # Extract frames (or load image)
    if media_kind == "video":
        frames, total_frames = extract_frames(
            path,
            max_frames=opts.max_frames,
            progress_callback=progress,
            check_stop=cancel,
        )
        features.total_frames = total_frames
    else:
        frames = [load_image(path)]
        features.total_frames = 1
    
    if not frames:
        flags.append("NO_FRAMES_EXTRACTED")
    
    # TODO: Call actual extractors here
    # For now, placeholder scores
    features.ai_face_score = 50.0
    features.ai_scene_score = 45.0
    features.ai_video_score = 55.0
    
    # Fuse AI scores
    ai_combined = fuse_scores(
        features.ai_face_score,
        features.ai_scene_score,
        features.ai_video_score,
    )
    
    # Compute deepfake score (if forensic available)
    deepfake_score = compute_deepfake_score(
        features.ai_face_score,
        features.ai_video_score,
        None,  # TODO: forensic score
    )
    
    # Get verdicts
    ai_verdict = get_verdict(ai_combined, opts.decision_policy)
    df_verdict = get_verdict(deepfake_score, opts.decision_policy)
    combined_verdict = combine_verdicts(ai_verdict, df_verdict)
    
    # Final score: max of ai_combined and deepfake_score
    final_score = max(
        s for s in [ai_combined, deepfake_score] if s is not None
    )
    
    # Create result
    result = AnalysisResult(
        verdict=combined_verdict,
        final_score=final_score,
        features=features,
        flags=flags,
    )
    
    # Write reports
    if opts.write_txt:
        txt_path = write_txt_report(result, run_dir)
        result.report_txt_path = txt_path
    
    if opts.write_json:
        json_path = write_json_report(result, run_dir)
        result.report_json_path = json_path
    
    result.report_folder = run_dir
    
    return result
