#%%
import os
import sys

import requests
import table_ocr.util
import extract_tables
import extract_cells
import ocr_image
import ocr_to_csv

# def download_image_to_tempdir(url, filename=None):
#     if filename is None:
#         filename = os.path.basename(url)
#     response = requests.get(url, stream=True)
#     tempdir = table_ocr.util.make_tempdir("demo")
#     filepath = os.path.join(tempdir, filename)
#     with open(filepath, 'wb') as f:
#         for chunk in response.iter_content():
#             f.write(chunk)
#     return filepath

def main(filename):
#    image_filepath = download_image_to_tempdir(url)
    image_filepath = filename
    image_tables = extract_tables.main([image_filepath])
    print("Running `{}`".format(f"extract_tables.main([{image_filepath}])."))
    print("Extracted the following tables from the image:")
    
    for image, tables in image_tables:
        print(f"Processing tables for {image}.")
        for table in tables:
            print(f"Processing table {table}.")
            cells = extract_cells.main(table)
            ocr = [
                ocr_image.main(cell, None)  #None = tessargs
                for cell in cells
            ]
            print("Extracted {} cells from {}".format(len(ocr), table))
            print("Cells:")
            for c, o in zip(cells[:3], ocr[:3]):
                with open(o) as ocr_file:
                    # Tesseract puts line feeds at end of text.
                    # Stript it out.
                    text = ocr_file.read().strip()
                    print("{}: {}".format(c, text))
            # If we have more than 3 cells (likely), print an ellipses
            # to show that we are truncating output for the demo.
            if len(cells) > 3:
                print("...")
            return ocr_to_csv.text_files_to_csv(ocr)


#%%

csv_output = main('test002.png')

#%%
if __name__ == "__main__":
    csv_output = main(sys.argv[1])
    print()
    print("Here is the entire CSV output:")
    print()
    print(csv_output)
