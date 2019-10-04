from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from tesseract_hocr import to_text
import evaluate_hocr
import subprocess

class ExampleHandler(FileSystemEventHandler):
    def on_created(self, event): # when file is created
        # do something, eg. call your function to process the image
        print(event.src_path)
        out = to_text(event.src_path)
        filename = event.src_path.split('/')[-1][:-4]
        filename = str(filename)
        with open('invoice2/hocr/'+filename+'.hocr', 'a') as f:
            f.write(out.decode("utf-8"))
        print('HOCR written')   

        out = subprocess.call(["python","invoice2/evaluate_hocr.py",filename])
        print(out)
        print('Output written in xlsx and json')
        
        # run_extractor(event.src_path)

observer = Observer()
event_handler = ExampleHandler() # create event handler
# set observer to use created handler in directory
observer.schedule(event_handler, path='invoice2/jpg')
observer.start()

# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    while True:
        time.sleep(1)
        print('not yet')
except KeyboardInterrupt:
    observer.stop()

observer.join()