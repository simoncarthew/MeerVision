# from pytube import YouTube

# # where to save
# SAVE_PATH = "/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/Testing/Vision"

# # link of the video to be downloaded
# link = "https://www.youtube.com/watch?v=W4og8g_X22o"

# try:
#     # object creation using YouTube
#     yt = YouTube(link)
#     print("Successfully connected to YouTube")
# except Exception as e:
#     # to handle exception
#     print(f"Connection Error: {e}")
#     exit(1)

# try:
#     # Get all streams and filter for mp4 files
#     mp4_streams = yt.streams.filter(file_extension='mp4')
#     print(f"Found {len(mp4_streams)} MP4 streams")

#     # get the video with the highest resolution
#     d_video = mp4_streams.get_highest_resolution()
#     print(f"Selected stream: {d_video}")

#     # downloading the video
#     d_video.download(output_path=SAVE_PATH)
#     print('Video downloaded successfully!')
# except Exception as e:
#     print(f"Some Error: {e}")

import yt_dlp

# where to save
SAVE_PATH = "//home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/data/camera_trap_footage"

# link of the video to be downloaded
link = "https://www.youtube.com/watch?v=oF8y6hPMaIM"

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
