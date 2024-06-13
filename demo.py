import cv2
import os
import time
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Watcher:
    DIRECTORY_TO_WATCH = "video/"  # Change this to the directory you want to watch
    if not os.path.exists(DIRECTORY_TO_WATCH):
        print(f"Directory {DIRECTORY_TO_WATCH} does not exist.")

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()

class Handler(FileSystemEventHandler):

    @staticmethod
    def on_created(event):
        # Check if the file is a video file (you can adjust the extension as needed)
        if event.is_directory or not event.src_path.lower().endswith(('.mp4', '.avi', '.mov')):
            return
        print(f"Received created event - {event.src_path}.")
        extract_frames(event.src_path)

def extract_frames(video_path):
    current_time = datetime.now().strftime('%H-%M-%S')
    frames_dir = os.path.join(Watcher.DIRECTORY_TO_WATCH, current_time)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            frame_count += 1
        else:
            break
    cap.release()
    print("All frames extracted and saved.")

if __name__ == "__main__":
    w = Watcher()
    w.run()
