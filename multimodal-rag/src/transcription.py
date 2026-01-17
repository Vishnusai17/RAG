"""Audio transcription using OpenAI Whisper with timestamps."""

import whisper
from typing import List, Dict
from pathlib import Path

from src.config import WHISPER_MODEL, CHUNK_SIZE


class AudioTranscriber:
    """Transcribe audio to text with word-level timestamps using Whisper."""
    
    def __init__(self, model_name: str = WHISPER_MODEL):
        """
        Initialize Whisper model.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        print(f"Loading Whisper model '{model_name}'...")
        self.model = whisper.load_model(model_name)
        print("Whisper model loaded successfully")
    
    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """
        Transcribe audio file with timestamps.
        
        Args:
            audio_path: Path to audio file (WAV format)
            
        Returns:
            List of dicts with keys: 'text', 'start_time', 'end_time'
        """
        print(f"Transcribing audio from {audio_path}...")
        
        # Transcribe with word timestamps
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False
        )
        
        # Extract segments with timestamps
        segments = []
        for segment in result['segments']:
            segments.append({
                'text': segment['text'].strip(),
                'start_time': segment['start'],
                'end_time': segment['end']
            })
        
        print(f"Transcription complete: {len(segments)} segments")
        return segments
    
    def chunk_transcription(self, segments: List[Dict], 
                          chunk_size: int = CHUNK_SIZE) -> List[Dict]:
        """
        Combine segments into larger chunks for better retrieval context.
        
        Args:
            segments: List of transcription segments
            chunk_size: Target character count per chunk
            
        Returns:
            List of chunked segments with combined text and timestamps
        """
        if not segments:
            return []
        
        chunks = []
        current_chunk = {
            'text': '',
            'start_time': segments[0]['start_time'],
            'end_time': segments[0]['end_time']
        }
        
        for segment in segments:
            # Check if adding this segment would exceed chunk size
            if len(current_chunk['text']) + len(segment['text']) > chunk_size and current_chunk['text']:
                # Save current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = {
                    'text': segment['text'],
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time']
                }
            else:
                # Add to current chunk
                if current_chunk['text']:
                    current_chunk['text'] += ' ' + segment['text']
                else:
                    current_chunk['text'] = segment['text']
                current_chunk['end_time'] = segment['end_time']
        
        # Add final chunk
        if current_chunk['text']:
            chunks.append(current_chunk)
        
        print(f"Created {len(chunks)} chunks from {len(segments)} segments")
        return chunks
