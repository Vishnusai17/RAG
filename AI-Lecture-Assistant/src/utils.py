"""Utility functions for LectureChat."""

import re
import uuid
import cv2
from pathlib import Path
from typing import Optional


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string (MM:SS)
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def validate_youtube_url(url: str) -> bool:
    """
    Validate if a URL is a valid YouTube URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid YouTube URL, False otherwise
    """
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
    return bool(re.match(youtube_regex, url))


def generate_video_id() -> str:
    """
    Generate a unique video ID for this session.
    
    Returns:
        Unique video ID string
    """
    return str(uuid.uuid4())[:8]


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return duration


def create_directories():
    """Initialize all required data storage directories."""
    from src.config import VIDEOS_DIR, AUDIO_DIR, FRAMES_DIR, CHROMA_DIR
    
    for directory in [VIDEOS_DIR, AUDIO_DIR, FRAMES_DIR, CHROMA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
