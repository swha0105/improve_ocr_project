import argparse

from table_ocr.ocr_image import main

description="""Takes a single argument that is the image to OCR.
Remaining arguments are passed directly to Tesseract.

Attempts to make OCR more accurate by performing some modifications on the image.
Saves the modified image and the OCR text in an `ocr_data` directory.
Filenames are of the format for training with tesstrain."""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("image", help="filepath of image to perform OCR")
tess_args = ["--psm", "7", "-l", "kor", "--tessdata-dir", tessdata_dir]
print(tess_args,"printout")

args, tess_args = parser.parse_known_args()


print(main(args.image, tess_args))
