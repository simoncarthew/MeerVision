import yt_dlp

# where to save
SAVE_PATH = "/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/Data/YoutubeCameraTrap"

# link of the video to be downloaded
link = "https://www.youtube.com/watch?v=a1_KVqM_nGc"

# Set up the options for yt-dlp
ydl_opts = {
    'format': 'mp4',  # Specifies that the format should be mp4
    'outtmpl': SAVE_PATH + '/%(title)s.%(ext)s',  # Path and filename template
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Download the video
        ydl.download([link])
        print('Video downloaded successfully!')
except Exception as e:
    print(f"Some Error: {e}")
