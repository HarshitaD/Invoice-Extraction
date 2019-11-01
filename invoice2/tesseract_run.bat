magick -density 400 %1 -depth 8 -alpha off tmp.png
tesseract tmp.png %2 --dpi 400 hocr -c hocr_font_info=1