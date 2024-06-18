import cv2
import os
import time
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from infer import infer
class Watcher:
    DIRECTORY_TO_WATCH = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/demo/video/"  # Change this to the directory you want to watch
    DIRECTORY_TO_SAVE = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/demo/frames/"  # Change this to the directory you want to save the frames
    MODEL_LOCATION = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/demo/model/final_model.h5"  # Change this to the location of the model

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
        run_model(event.src_path, Watcher.MODEL_LOCATION)

def extract_frames(video_path , frames_dir):


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

def run_model(video_path, model_path):

    current_time = datetime.now().strftime('%H-%M-%S')
    frames_dir = os.path.join(Watcher.DIRECTORY_TO_SAVE, current_time)
    os.makedirs(frames_dir, exist_ok=True)
    extract_frames(video_path, frames_dir)
    infer(frames_dir, model_path, current_time)
    

if __name__ == "__main__":
    w = Watcher()
    w.run()
