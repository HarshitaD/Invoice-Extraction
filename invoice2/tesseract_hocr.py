# -*- coding: utf-8 -*-
from config import *
from pathlib import Path, PureWindowsPath

tesseract_run_path = Path("invoice2") / Path("tesseract_run.bat")
if os_for_pathlib == "Windows":
    tesseract_run_path = PureWindowsPath(tesseract_run_path)
tesseract_run_path = str(tesseract_run_path)
def to_text(path,  path_out = None):

    if os_for_pathlib == "Windows":
        import subprocess
        p = subprocess.Popen([tesseract_run_path, path, path_out, "shell=True"])
        out = p.communicate()
        return out

    """Wraps Tesseract OCR.

    Parameters
    ----------
    path : str
        path of electronic invoice in JPG or PNG format

    Returns
    -------
    extracted_str : str
        returns extracted text from image in JPG or PNG format

    """
    import subprocess
    from distutils import spawn

    # Check for dependencies. Needs Tesseract and Imagemagick installed.
    if not spawn.find_executable("tesseract"):
        raise EnvironmentError("tesseract not installed.")
    if not spawn.find_executable("magick"):
        raise EnvironmentError("imagemagick not installed.")

    # convert = "convert -density 350 %s -depth 8 tiff:-" % (path)
    convert = [
        # "convert",
        "magick",
        "-density",
        "400",
        path,
        "-depth",
        "8",
        "-alpha",
        "off",
        "png:-",
    ]
    p1 = subprocess.Popen(convert, stdout=subprocess.PIPE)

    tess = ["tesseract", "stdin", "stdout", "--dpi", "400", "hocr", "-c", "hocr_font_info=1"]
    # tess = ["tesseract", "stdin", "stdout"]
    p2 = subprocess.Popen(tess, stdin=p1.stdout, stdout=subprocess.PIPE)

    out, err = p2.communicate()

    extracted_str = out

    return extracted_str
