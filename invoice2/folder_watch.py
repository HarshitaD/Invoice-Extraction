from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from tesseract_hocr import to_text
import subprocess
from pathlib import Path, PureWindowsPath
from config import hocr_folder, jpg_folder, os_for_pathlib


evaluate_hocr_path = Path("invoice2") / "evaluate_hocr.py"
class ExampleHandler(FileSystemEventHandler):
    def on_created(self, event):
        # when file is created
        # do something, eg. call your function to process the image
        print(event.src_path)
        out = to_text(Path(event.src_path))
        tmp = event.src_path.split('/')[-1][:-4]
        tmp = str(tmp)
        hocr_file = "{}.hocr".format(tmp)
        # hocr_filepath = Path(hocr_folder) / hocr_file
        # if os_for_pathlib == "Windows":
        #     hocr_filepath = PureWindowsPath(hocr_filepath)
        # hocr_filepath = str(hocr_filepath)
        with open(Path(hocr_folder) / hocr_file, 'a') as f:
            f.write(out.decode("utf-8"))
        print('HOCR written')  
        out = subprocess.call(["python", evaluate_hocr_path , hocr_filename])
        print(out)
        print('Output written in xlsx and json')
        
        # run_extractor(event.src_path)


observer = Observer()
event_handler = ExampleHandler() # create event handler
# set observer to use created handler in directory
if os_for_pathlib == 'Windows':
    observer.schedule(event_handler, path=str(PureWindowsPath(Path(jpg_folder))) )
else:
    observer.schedule(event_handler, path=str(jpg_folder) )
observer.start()

# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    print('Testing a file meanwhile')
    hocr_filepath = Path("6")
    
    if os_for_pathlib == "Windows":
        hocr_filepath = PureWindowsPath(hocr_filepath)
    
    hocr_filename = str(hocr_filepath)
    out = subprocess.call(["python", str(evaluate_hocr_path) , hocr_filename])
    print("Note: Add your files in the jpg_folder yo find the corrsponding xlsx in the output_folder")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()