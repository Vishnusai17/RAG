import yt_dlp
import os

url = "https://www.youtube.com/watch?v=yUmDRxV0krg"
output_path = "test_download.mp4"

print(f"Attempting to download {url}...")

# Original options used in the app
ydl_opts_original = {
    'format': 'best[ext=mp4]',
    'outtmpl': output_path,
    'quiet': False,
    'verbose': True
}

# New proposed options (more robust)
ydl_opts_improved = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Fallback chain
    'outtmpl': output_path,
    'quiet': False,
    'verbose': True
}

try:
    print("\n--- Trying original options ---")
    with yt_dlp.YoutubeDL(ydl_opts_original) as ydl:
        ydl.download([url])
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("✅ Download successful!")
    else:
        print("❌ Download failed: File is empty or missing")

except Exception as e:
    print(f"❌ Error: {e}")
