"""Run directory management."""

import os
from datetime import datetime


def create_run_dir(reports_root: str = "reports") -> str:
    """Create a timestamped run directory.
    
    Args:
        reports_root: Base directory for all reports
        
    Returns:
        Path to created run directory
    """
    os.makedirs(reports_root, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(reports_root, f"run_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
