# Invoice-Extraction
Rule-based Invoice Extraction for GST receipts

## Pre-requisites:
1. Git for windows: https://gitforwindows.org/
2. Python 3.7.4: https://www.python.org/downloads/release/python-374/

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
### Run the invoice extraction module
```
python invoice2/folder_watch.py
```

## Usage:
Paste the invoice jpg in the jpg folder.
Output will be saved in xlsx and json format in the output folder
