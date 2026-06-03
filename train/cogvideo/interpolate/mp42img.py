import cv2
import os
import argparse

def extract_frames(mp4_path, output_dir, fps):
    """
    Extract frames from an MP4 file at a specified FPS and save them to an output directory.

    Args:
        mp4_path (str): Path to the MP4 file.
        output_dir (str): Directory where extracted frames will be saved.
        fps (int): Frames per second to extract.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(mp4_path)
    if not video.isOpened():
        print(f"Error: Unable to open video file {mp4_path}")
        return

    # Get the original video's frame rate
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Save every nth frame based on the desired FPS
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an MP4 file.")
    parser.add_argument("mp4_path", type=str, help="Path to the MP4 file.")
    parser.add_argument("output_dir", type=str, help="Directory to save extracted frames.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second to extract (default: 1).")

    args = parser.parse_args()

    extract_frames(args.mp4_path, args.output_dir, args.fps)