# Invoice-Extraction
Rule-based Invoice Extraction for GST receipts

## Setup:
### Install virtualvnv
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
pip install requirements.txt
```
### Run the invoice extraction module
```
python invoice2/folder_watch.py
```

## Usage:
Paste the invoice jpg in the jpg folder.
Output will be saved in xlsx and json format in the output folder
