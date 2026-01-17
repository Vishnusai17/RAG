import yt_dlp
import os
from pathlib import Path

# The problematic URL
url = "https://www.youtube.com/watch?v=yUmDRxV0krg"
output_dir = Path("data/videos")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Attempting to download {url} to {output_dir}")

# CLEAN UP: Remove any partial files first
for f in output_dir.glob("*"):
    if f.is_file():
        f.unlink()

ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
    'quiet': False,
    'verbose': True,
    # CRITICAL: Do NOT ignore errors. We want to see why it fails.
    'ignoreerrors': False,
    'nocheckcertificate': True,
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    files = list(output_dir.glob("*"))
    print(f"\nFiles in directory after download: {files}")

except Exception as e:
    print(f"\n❌ CAUGHT EXCEPTION: {e}")
