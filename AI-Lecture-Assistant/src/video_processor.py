"""Video processing with smart scene detection and frame extraction."""

import os
import cv2
import imagehash
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import yt_dlp
from moviepy.editor import VideoFileClip
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images

from src.config import (
    VIDEOS_DIR, AUDIO_DIR, FRAMES_DIR,
    SCENE_THRESHOLD, SIMILARITY_THRESHOLD, MAX_FRAMES_PER_HOUR
)
from src.utils import generate_video_id, get_video_duration


class SceneDetector:
    """
    Smart scene detection with perceptual hash deduplication.
    Extracts middle frames from detected scenes and removes near-duplicates.
    """
    
    def __init__(self, threshold: float = SCENE_THRESHOLD, 
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize SceneDetector.
        
        Args:
            threshold: Scene change detection threshold (default: 27.0)
            similarity_threshold: Perceptual hash similarity threshold (default: 0.90)
        """
        self.threshold = threshold
        self.similarity_threshold = similarity_threshold
    
    def detect_scenes(self, video_path: str) -> List[Tuple[int, int]]:
        """
        Detect scene changes in video using AdaptiveDetector.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of (start_frame, end_frame) tuples for each scene
        """
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=self.threshold))
        
        # Detect scenes
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        
        # Convert to frame numbers
        scenes = []
        for scene in scene_list:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            scenes.append((start_frame, end_frame))
        
        return scenes
    
    def extract_middle_frame(self, video_path: str, start_frame: int, 
                            end_frame: int) -> Optional[np.ndarray]:
        """
        Extract the middle frame of a scene to avoid transition blur.
        
        Args:
            video_path: Path to video file
            start_frame: Start frame number
            end_frame: End frame number
            
        Returns:
            Frame as numpy array (BGR format) or None if extraction fails
        """
        cap = cv2.VideoCapture(video_path)
        
        # Calculate middle frame
        middle_frame = (start_frame + end_frame) // 2
        
        # Seek to middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        cap.release()
        
        return frame if ret else None
    
    def compute_perceptual_hash(self, frame: np.ndarray) -> imagehash.ImageHash:
        """
        Compute perceptual hash of a frame.
        
        Args:
            frame: Frame as numpy array (BGR format)
            
        Returns:
            Perceptual hash
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return imagehash.phash(pil_image)
    
    def calculate_similarity(self, hash1: imagehash.ImageHash, 
                           hash2: imagehash.ImageHash) -> float:
        """
        Calculate similarity between two perceptual hashes.
        
        Args:
            hash1: First perceptual hash
            hash2: Second perceptual hash
            
        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        # Hamming distance (number of differing bits)
        distance = hash1 - hash2
        # Convert to similarity (0-1 range, assuming 64-bit hash)
        similarity = 1 - (distance / 64.0)
        return similarity
    
    def deduplicate_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Remove near-duplicate frames using perceptual hash comparison.
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            List of unique frames after deduplication
        """
        if not frames:
            return []
        
        unique_frames = [frames[0]]
        prev_hash = self.compute_perceptual_hash(frames[0])
        
        for frame in frames[1:]:
            curr_hash = self.compute_perceptual_hash(frame)
            similarity = self.calculate_similarity(prev_hash, curr_hash)
            
            # Only keep frame if it's sufficiently different
            if similarity < self.similarity_threshold:
                unique_frames.append(frame)
                prev_hash = curr_hash
        
        return unique_frames
    
    def extract_unique_frames(self, video_path: str, 
                             output_dir: Path) -> List[Tuple[str, float]]:
        """
        Extract unique frames from video using scene detection and deduplication.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            
        Returns:
            List of (frame_path, timestamp) tuples
        """
        # Detect scenes
        scenes = self.detect_scenes(video_path)
        
        if not scenes:
            print("No scenes detected. Video may be too short or uniform.")
            return []
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Extract middle frame from each scene
        frames_with_timestamps = []
        frames = []
        
        for start_frame, end_frame in scenes:
            frame = self.extract_middle_frame(video_path, start_frame, end_frame)
            if frame is not None:
                middle_frame_num = (start_frame + end_frame) // 2
                timestamp = middle_frame_num / fps
                frames.append(frame)
                frames_with_timestamps.append((frame, timestamp))
        
        # Deduplicate frames
        unique_frames_only = self.deduplicate_frames(frames)
        
        # Map deduplicated frames back to timestamps
        deduplicated_with_timestamps = []
        unique_hashes = {id(frame): self.compute_perceptual_hash(frame) 
                        for frame in unique_frames_only}
        
        for frame, timestamp in frames_with_timestamps:
            frame_hash = self.compute_perceptual_hash(frame)
            # Check if this frame is in unique set
            for unique_frame in unique_frames_only:
                unique_hash = unique_hashes[id(unique_frame)]
                if self.calculate_similarity(frame_hash, unique_hash) >= self.similarity_threshold:
                    deduplicated_with_timestamps.append((frame, timestamp))
                    break
        
        # Save frames
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_frames = []
        
        for idx, (frame, timestamp) in enumerate(deduplicated_with_timestamps):
            frame_filename = f"frame_{idx:04d}_t{int(timestamp)}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append((str(frame_path), timestamp))
        
        print(f"Extracted {len(saved_frames)} unique frames from {len(scenes)} scenes")
        return saved_frames


class VideoProcessor:
    """
    Handle video ingestion from YouTube or file upload.
    Extract audio and key frames with smart scene detection.
    """
    
    def __init__(self):
        """Initialize VideoProcessor."""
        self.scene_detector = SceneDetector()
    
    def download_youtube(self, url: str, video_id: str) -> str:
        """
        Download video from YouTube URL.
        
        Args:
            url: YouTube URL
            video_id: Unique video ID for this session
            
        Returns:
            Path to downloaded video file
        """
        output_path = VIDEOS_DIR / f"{video_id}.mp4"
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_path),
            'quiet': False,
            'verbose': True,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',
            # Force mobile client to bypass some age gates/login requirements
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Aggressive file finding: look for ANY file that starts with the video_id
            # yt-dlp might add extensions like .webm, .mkv, or suffix like .f137
            possible_files = list(VIDEOS_DIR.glob(f"{video_id}*"))
            
            # Filter out the target path itself if it exists but is empty
            valid_files = [f for f in possible_files if f.stat().st_size > 0]
            
            if not valid_files:
                # Last resort: check if there are any new files in the directory
                # This is risky but useful for debugging
                raise Exception(f"Download finished but no file found with ID {video_id}. Found files: {[f.name for f in possible_files]}")
            
            # If we found chunks/parts, pick the largest one (likely the video)
            # or the one that isn't the target output path
            downloaded_file = max(valid_files, key=lambda p: p.stat().st_size)
            
            # If the file is not where we expect, move it
            if downloaded_file != output_path:
                print(f"Renaming {downloaded_file} to {output_path}")
                if output_path.exists():
                    output_path.unlink()
                downloaded_file.rename(output_path)
                    
            print(f"Downloaded video to {output_path}")
            return str(output_path)
        except Exception as e:
            raise Exception(f"Failed to download video: {str(e)}")
    
    def save_uploaded_video(self, uploaded_file, video_id: str) -> str:
        """
        Save uploaded video file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            video_id: Unique video ID for this session
            
        Returns:
            Path to saved video file
        """
        output_path = VIDEOS_DIR / f"{video_id}.mp4"
        
        try:
            with open(output_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            print(f"Saved uploaded video to {output_path}")
            return str(output_path)
        except Exception as e:
            raise Exception(f"Failed to save video: {str(e)}")
    
    def extract_audio(self, video_path: str, video_id: str) -> str:
        """
        Extract audio from video to WAV format.
        
        Args:
            video_path: Path to video file
            video_id: Unique video ID for this session
            
        Returns:
            Path to extracted audio file
        """
        audio_path = AUDIO_DIR / f"{video_id}.wav"
        
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(
                str(audio_path),
                codec='pcm_s16le',
                verbose=False,
                logger=None
            )
            video.close()
            print(f"Extracted audio to {audio_path}")
            return str(audio_path)
        except Exception as e:
            raise Exception(f"Failed to extract audio: {str(e)}")
    
    def extract_frames(self, video_path: str, video_id: str) -> List[Tuple[str, float]]:
        """
        Extract unique frames using smart scene detection.
        
        Args:
            video_path: Path to video file
            video_id: Unique video ID for this session
            
        Returns:
            List of (frame_path, timestamp) tuples
        """
        frames_dir = FRAMES_DIR / video_id
        
        try:
            frames = self.scene_detector.extract_unique_frames(video_path, frames_dir)
            return frames
        except Exception as e:
            raise Exception(f"Failed to extract frames: {str(e)}")
    
    def cleanup_temp_files(self, video_id: str):
        """
        Clean up temporary processing files.
        
        Args:
            video_id: Unique video ID for this session
        """
        import shutil
        
        # Remove video file
        video_path = VIDEOS_DIR / f"{video_id}.mp4"
        if video_path.exists():
            video_path.unlink()
        
        # Remove audio file
        audio_path = AUDIO_DIR / f"{video_id}.wav"
        if audio_path.exists():
            audio_path.unlink()
        
        # Remove frames directory
        frames_dir = FRAMES_DIR / video_id
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        
        print(f"Cleaned up temporary files for video {video_id}")
