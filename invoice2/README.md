# Invoice-Extraction
Rule-based Invoice Extraction for GST receipts

## Pre-requisites:
1. Git for windows: https://gitforwindows.org/
2. Python 3.7.4: https://www.python.org/downloads/release/python-374/
3. Pip for python package management: 
  ```
  curl https://bootstrap.pypa.io/get-pip.py > get_pip.py
  set path=%PATH%;C:\Users\Admin\AppData\Local\Programs\Python\Python37\Scripts
  ```
## Setup:
### Install virtualenv
```
py -m pip install --user virtualenv
```
### Create virtualenv
```
py -m venv env
.\env\Scripts\activate
```
### Install dependencies in the virtualenv
```
pip install -r requirements.txt
```

## Run the invoice extraction module
```
python invoice2/folder_watch.py
```
This will process a sample file and store the result in the output_folder from the config_file

## Usage:
Paste the invoice jpg in the jpg folder.
Output will be saved in xlsx in the output folder
