"""
process_images.py — OCR Pipeline for League of Legends Loot Table Screenshots
===============================================================================

This script reads screenshot images from a folder, runs optical character
recognition (OCR) on each one to extract text, then parses the text into
structured data columns (percentage, item_name, me_value) and writes the
results to a CSV file.

Designed for League of Legends Hextech Crafting loot table screenshots, where
each row in the image shows a drop probability, an item name, and a Mythic
Essence reward value.

PIPELINE OVERVIEW:
  1. Find all .png images in the input folder
  2. Run PaddleOCR on each image to detect text regions
  3. Filter out low-confidence detections
  4. Merge text boxes on the same visual row (OCR often splits one line
     into multiple detected regions)
  5. Parse each merged row into (percentage, item_name, me_value)
  6. Sort by percentage descending
  7. Write final CSV
     (deduplication is handled downstream by quality_assurance_b_tier.py)

USAGE:
  1. Place screenshot images in the INPUT_FOLDER directory.
  2. Run: python process_images.py
  3. Output CSV appears at OUTPUT_FILE.

DEPENDENCIES:
  - nvidia CUDA toolkit
  - paddleocr (pip install paddleocr)
  - paddlepaddle (the deep learning framework PaddleOCR runs on)
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────

# "os" provides functions for interacting with the operating system, such as
# listing files in a directory and constructing file paths.
import os

# "re" is python's regular expression (regex) module. regex is a mini-language
# for describing text patterns, e.g. "one or more digits followed by a % sign".
# we use it to extract percentages, numbers, and item names from OCR text.
import re

# "csv" provides tools for reading and writing CSV (comma-separated values)
# files, which are simple spreadsheet-like text files that programs like Excel
# can open.
import csv

# PaddleOCR is the OCR engine. it takes an image as input and returns a list of
# detected text regions, each with: the bounding box coordinates (where the text
# is on the image), the recognised text string, and a confidence score (0 to 1)
# indicating how certain the engine is that it read the text correctly.
from paddleocr import PaddleOCR


# ─── CONSTANTS ───────────────────────────────────────────────────────────────
# Constants are variables whose values never change during the program. By
# convention they are written in UPPER_CASE so they stand out from regular
# variables. Grouping them at the top makes them easy to find and adjust.

# The folder containing the original screenshot images.
INPUT_FOLDER = "/app/images"

# The file path where the final CSV output will be written.
OUTPUT_FILE = "/app/ocr_results.csv"

# The file path where raw OCR text is saved before any parsing or filtering.
# Stored alongside this script file regardless of where it is run from.
RAW_OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_raw_data.csv")

# Our pre-defined confidence threshold. PaddleOCR assigns each detected text
# region a confidence score between 0 (no confidence) and 1 (fully confident).
# If the score is below this threshold, we discard the text as unreliable.
CONF_THRESHOLD = 0.6


# The maximum vertical distance (in pixels) between two OCR bounding boxes for
# them to be considered part of the same visual row. PaddleOCR often detects the
# percentage and the item description as separate text boxes even though they
# appear on the same line in the image. By comparing their vertical midpoints
# (y-coordinates), we can merge boxes that are close together into a single row
# before parsing. If rows are incorrectly splitting, increase this value; if
# adjacent rows are incorrectly merging, decrease it.
Y_TOLERANCE = 15


# ─── FUNCTIONS ───────────────────────────────────────────────────────────────


def parse_line(text):
    """
    Takes a single string of merged OCR text (e.g. "0.05343% Got 'em + 2
    Mythic Essence") and extracts three fields from it:
      - percentage: the drop probability (e.g. "0.05343%")
      - name:       the item name (e.g. "Got 'em")
      - me_value:   the Mythic Essence reward as a plain integer string (e.g. "2")

    Returns a tuple of (percentage, name, me_value). Any field that cannot be
    found in the text is returned as an empty string "".

    This function uses three separate regex passes:
      1. Search for the percentage pattern
      2. Search for the Mythic Essence pattern
      3. Remove both patterns from the original text; whatever remains is the name
    """
    # Initialise all three fields as empty strings. If a field isn't found in
    # the text, it will remain empty rather than causing an error.
    percentage = ""
    name = ""
    me_value = ""

    # ── Extract the percentage ──
    #
    # Regex breakdown:  \d+  = one or more digits (e.g. "0", "51")
    #                   \.?  = an optional decimal point
    #                   \d+  = one or more digits after the decimal (e.g. "05343")
    #                   %    = a literal percent sign
    #
    # Example matches: "0.05343%", "51.37%", "9.129%"
    #
    # re.search() scans the entire string left to right and returns the first
    # match it finds, or None if no match exists. This is different from
    # re.match() which only checks the very start of the string.
    percent_match = re.search(r'\d+\.?\d+%', text)

    # If a match was found, .group() returns the full matched text.
    # e.g. if the input was "0.05343% Got 'em", .group() returns "0.05343%".
    if percent_match:
        percentage = percent_match.group()

    # ── Extract the Mythic Essence value ──
    #
    # Regex breakdown:  \+?          = an optional "+" sign
    #                   \s*          = zero or more whitespace characters
    #                   (\d+)        = one or more digits, captured as group 1
    #                   \s*          = zero or more whitespace characters
    #                   M[vy]thic    = "Mythic" or "Mvthic" (OCR sometimes reads
    #                                  'y' as 'v', e.g. "Mvthic Essence")
    #                   \s+Essence   = one or more spaces then "Essence"
    #
    # re.IGNORECASE makes the match case-insensitive so "mythic essence" also
    # matches, guarding against other capitalisation variations.
    #
    # Example matches: "2 Mythic Essence", "+ 2 Mythic Essence", "+2 Mvthic Essence"
    #
    # The parentheses around \d+ create a "capture group". This lets us extract
    # just the number without the surrounding text. The full match is group(0);
    # the first set of parentheses is group(1), the second would be group(2), etc.
    me_match = re.search(r'\+?\s*(\d+)\s*M[vy]thic\s*Essence\.*', text, re.IGNORECASE)

    # .group(1) returns the first capture group — just the digits (e.g. "2"),
    # not the full match (e.g. "+ 2 Mythic Essence").
    if me_match:
        me_value = me_match.group(1)

    # Fallback: if the "Mythic Essence" label was not detected at all (e.g. the
    # OCR text box for it was missed entirely), a trailing "+ N" or "+N" at the
    # end of the merged row text still signals the ME value.
    # e.g. "0.05343% BetterTogether+2" or "0.05343% Rakan Romance Icon + 2"
    if not me_value:
        trailing_match = re.search(r'\+\s*(\d+)\s*$', text)
        if trailing_match:
            me_value = trailing_match.group(1)

    # ── Extract the item name ──
    #
    # Strategy: the item name is whatever is left after removing the percentage
    # and the Mythic Essence expression. We do this by substitution: re.sub()
    # replaces every occurrence of the pattern with an empty string, effectively
    # deleting it from the text.
    cleaned = text

    # Remove the percentage (e.g. "0.05343%"). Note: \d* (zero or more digits)
    # is used after the decimal instead of \d+ to also catch edge cases like "5.%"
    cleaned = re.sub(r'\d+\.?\d*%', '', cleaned)

    # Remove the full Mythic Essence expression including any leading "+"
    # (e.g. "+ 2 Mythic Essence", "2 Mythic Essence", "+2 Mvthic Essence")
    cleaned = re.sub(r'\s*\+?\s*\d+\s*M[vy]thic\s*Essence\.*', '', cleaned, flags=re.IGNORECASE)

    # Remove any standalone "Mythic/Mvthic Essence" text that might remain after
    # the above substitution, even if separated by irregular spacing
    cleaned = re.sub(r'M[vy]thic\s*Essence', '', cleaned, flags=re.IGNORECASE)

    # Remove a trailing "+ N" pattern left when the ME label was absent entirely
    # (fallback case — mirrors the fallback extraction above)
    cleaned = re.sub(r'\+\s*\d+\s*$', '', cleaned)

    # Clean up any leftover whitespace or stray "+" characters from the edges.
    # .strip() removes leading/trailing whitespace. .strip('+') then removes
    # any leading/trailing "+" characters. The final .strip() catches any
    # whitespace that was hiding behind the "+".
    cleaned = cleaned.strip().strip('+').strip()

    # If anything remains after all the removal, that's our item name.
    if cleaned:
        name = cleaned

    # Return all three extracted fields as a tuple. In python, writing
    # "return a, b, c" is shorthand for "return (a, b, c)" — a tuple.
    return percentage, name, me_value


def process_images(file_list, ocr):
    """
    Takes a list of image filenames and an initialised PaddleOCR instance.
    Runs OCR on each image, merges text boxes that belong to the same visual
    row, parses each merged row into structured fields, and returns a list of
    (percentage, item_name, me_value, filename) tuples.

    The merging step is necessary because PaddleOCR detects text at the level
    of individual "text regions" — contiguous blocks of characters that the
    model groups together. A single visual row like:

        0.05343%    Got 'em    + 2 Mythic Essence

    might be returned as two or three separate detections:
        detection 1: "0.05343%"             at position (x=50,  y=200)
        detection 2: "Got 'em"              at position (x=300, y=201)
        detection 3: "+ 2 Mythic Essence"   at position (x=600, y=199)

    We merge these by grouping detections whose vertical midpoints (y_mid) are
    within Y_TOLERANCE pixels of each other, then sorting each group left to
    right by horizontal position (x_mid) and joining the text fragments into
    one string.

    Parameters:
        file_list:  list of image filenames (strings), e.g. ["image01.png", ...]
        ocr:        an initialised PaddleOCR instance (reused across all images
                    to avoid reloading the neural network models each time)

    Returns:
        list of (percentage, name, me_value, filename) tuples — one per parsed
        row across all images. Rows that have neither a percentage nor an ME
        value (likely decorative text or OCR noise) are excluded.
    """
    # This list accumulates all parsed rows across all images. Each element is
    # a tuple of (percentage, name, me_value, filename).
    batch_rows = []

    # Raw merged text before any parsing — one entry per merged visual row.
    # Each element is a (merged_text, filename) tuple.
    raw_rows = []

    # Iterate over every image filename in the list.
    for filename in file_list:
        # Build the full path to the image.
        path = os.path.join(INPUT_FOLDER, filename)

        # try/except is python's error handling mechanism. If the code inside
        # "try" raises an error (an "exception"), python jumps to the "except"
        # block instead of crashing the entire program. This is important here
        # because a single corrupted image shouldn't abort processing of all
        # the remaining images.
        try:
            # Run the OCR engine on the image. The result is a nested list:
            #   result[0] = list of detected text regions for the first page
            #               (images only have one page; PDFs could have multiple)
            #
            # Each region in result[0] is structured as:
            #   [bounding_box, (text_string, confidence_score)]
            #
            # Where bounding_box is a list of four [x, y] corner points:
            #   [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            #   (top-left, top-right, bottom-right, bottom-left)
            #
            # Example:
            #   [[[50, 195], [150, 195], [150, 210], [50, 210]],
            #    ("0.05343%", 0.97)]
            result = ocr.ocr(path)

        except Exception as e:
            # "Exception as e" catches any error type and stores it in the
            # variable "e" so we can print a useful warning message.
            # "continue" skips to the next iteration of the for loop (i.e.
            # the next filename), without executing any more code below.
            print(f"[WARN] OCR failed on {filename}: {e}")
            continue

        # If PaddleOCR returned nothing (None) or the first page has no
        # detected text regions (empty list), skip this image.
        if not result or not result[0]:
            continue

        # ── Step 1: Filter by confidence and extract coordinates ──
        #
        # We collect only the text regions that meet our confidence threshold,
        # along with their x and y midpoints for later merging and sorting.
        #
        # confident_lines will hold tuples of (y_mid, x_mid, text).
        confident_lines = []
        for line in result[0]:
            # line[1] is a tuple of (text_string, confidence_score).
            text = line[1][0]  # the recognised text, e.g. "0.05343%"
            conf = line[1][1]  # confidence score, e.g. 0.97

            # Skip this text region if the OCR engine isn't confident enough.
            if conf < CONF_THRESHOLD:
                continue

            # line[0] contains the 4 corner points of the bounding box as
            # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]. We average all 4 corners
            # to find the centroid (midpoint) of the box, giving us its
            # position on the image.
            #   y_mid = vertical position (how far down the image; 0 = top)
            #   x_mid = horizontal position (how far right; 0 = left edge)
            #
            # "sum(point[1] for point in line[0])" is a generator expression
            # that iterates over each [x, y] point and sums the y-values (index 1).
            # Dividing by 4 gives the average.
            y_mid = sum(point[1] for point in line[0]) / 4
            x_mid = sum(point[0] for point in line[0]) / 4
            confident_lines.append((y_mid, x_mid, text))

        # ── Step 2: Sort by vertical position (top to bottom) ──
        #
        # "key=lambda x: x[0]" tells python to sort by the first element of
        # each tuple (y_mid). A lambda is a small anonymous (unnamed) function:
        # "lambda x: x[0]" means "given input x, return x[0]".
        #
        # This is equivalent to writing:
        #   def get_y(x):
        #       return x[0]
        #   confident_lines.sort(key=get_y)
        #
        # Sorting top-to-bottom ensures that when we merge rows in the next
        # step, we process them in the natural reading order of the image.
        confident_lines.sort(key=lambda x: x[0])

        # ── Step 3: Merge text boxes on the same visual row ──
        #
        # PaddleOCR often detects "0.05343%" and "Got 'em + 2 Mythic Essence"
        # as two separate text boxes. We need to merge them into a single
        # string before parsing. Two boxes are considered to be on the same row
        # if their vertical midpoints (y_mid) are within Y_TOLERANCE pixels of
        # each other.
        #
        # Data structure:
        #   merged_rows is a list of (y_mid, fragments) tuples, where:
        #     - y_mid is the vertical midpoint of the first box in the group
        #       (used as the reference point for subsequent boxes)
        #     - fragments is a list of (x_mid, text) tuples — all the text
        #       boxes that belong to this visual row
        #
        # Example after merging:
        #   [(200.5, [(50.0, "0.05343%"), (300.0, "Got 'em + 2 Mythic Essence")]),
        #    (230.0, [(50.0, "51.37%"), (400.0, "5")]),
        #    ...]
        merged_rows = []
        for y_mid, x_mid, text in confident_lines:
            # If there is a previous group and this box's y-coordinate is close
            # enough to it, add this box to that group. merged_rows[-1] refers
            # to the last (most recent) element in the list — the row we are
            # currently building.
            if merged_rows and abs(y_mid - merged_rows[-1][0]) < Y_TOLERANCE:
                merged_rows[-1][1].append((x_mid, text))
            # Otherwise, start a new group for this row. The double brackets
            # [(x_mid, text)] create a new list with one initial fragment.
            else:
                merged_rows.append((y_mid, [(x_mid, text)]))

        # ── Step 4: Sort each row's fragments left-to-right, then parse ──
        #
        # Within a merged row, the fragments may not be in left-to-right order
        # (PaddleOCR returns them in whatever order it detected them). Sorting
        # by x_mid ensures the percentage always comes before the item name,
        # producing consistent strings like "0.05343% Got 'em + 2 Mythic Essence".
        for _, fragments in merged_rows:
            # Sort this row's text fragments by their horizontal position.
            # "key=lambda f: f[0]" sorts by the first element of each
            # (x_mid, text) tuple, i.e. the horizontal coordinate.
            fragments.sort(key=lambda f: f[0])

            # Join all the text fragments into a single string separated by
            # spaces. The "for _, text in fragments" unpacks each (x_mid, text)
            # tuple, discarding x_mid (we don't need it anymore, hence the
            # underscore _ which is python's convention for "throwaway variable").
            #
            # " ".join(...) takes a list of strings and concatenates them with
            # a space between each pair. For example:
            #   " ".join(["0.05343%", "Got 'em", "+ 2 Mythic Essence"])
            #   → "0.05343% Got 'em + 2 Mythic Essence"
            merged_text = " ".join(text for _, text in fragments)

            # Save the raw merged text before any parsing or filtering.
            raw_rows.append((merged_text, filename))

            # Print the merged text for debugging so we can verify correct
            # merging. The "f" prefix creates an f-string, which lets us embed
            # variables directly inside curly braces: {filename} becomes the
            # actual filename value.
            print(f"[DEBUG] {filename}: '{merged_text}'")

            # Parse the merged string into its three components using our
            # parse_line() function defined above.
            percentage, name, me_value = parse_line(merged_text)

            # Only keep rows that have at least a percentage or ME value.
            # Rows with neither are likely decorative text, navigation buttons,
            # or other OCR noise from the game UI.
            if percentage or me_value:
                batch_rows.append((percentage, name, me_value, filename))

    # Return parsed rows and unedited raw rows from all images.
    return batch_rows, raw_rows


def main():
    """
    The main function that orchestrates the entire pipeline:
      1. Find all .png images in the input folder
      2. Initialise the OCR engine (once, reused for all images)
      3. Run OCR on every image to extract and parse text
      4. Sort results by percentage descending
      5. Write the final data to a CSV file
         (deduplication is handled downstream by quality_assurance_b_tier.py)
    """
    # ── Step 1: Find images ──
    #
    # os.listdir() returns a list of all filenames in the folder. The list
    # comprehension filters it to only include .png files. .lower() converts
    # the filename to lowercase so that "Image01.PNG" is also matched.
    # sorted() puts the filenames in ascending alphabetical order (image01,
    # image02, ...) so the output is deterministic regardless of the order
    # the OS happens to return them in.
    #
    # List comprehension syntax:
    #   [expression for variable in iterable if condition]
    # is equivalent to:
    #   result = []
    #   for variable in iterable:
    #       if condition:
    #           result.append(expression)
    image_files = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".png")
    ])

    # If the list is empty, no .png files were found. This could mean the
    # folder path is wrong or the images are a different file type.
    if not image_files:
        print("No images found in", INPUT_FOLDER)
        return

    print(f"Processing {len(image_files)} images\n")

    # ── Step 2: Initialise the OCR engine ──
    #
    # We create a single PaddleOCR instance and reuse it for all images. This
    # avoids loading the neural network models from disk multiple times, which
    # would be slow (the models are hundreds of megabytes).
    print("Initialising OCR models...")
    ocr = PaddleOCR(
        use_angle_cls=False,  # Disabled: game UI text is always horizontal,
                              # so we skip the angle classification model
                              # entirely, saving time on every image.
        lang='en',            # Use the english language recognition model.
        use_gpu=True,         # Run inference on the GPU for faster processing.
                              # Falls back to CPU if no GPU is available.
        show_log=False,       # Suppress PaddleOCR's verbose internal logging.
        rec_batch_num=8,      # Process up to 8 detected text regions at once
                              # through the recognition model, rather than one
                              # at a time, for better GPU throughput.
    )
    print("Models ready\n")

    # ── Step 3: Run OCR on all images ──
    all_rows, raw_rows = process_images(image_files, ocr)

    # ── Step 3b: Save raw OCR text before any parsing or filtering ──
    #
    # Written immediately after OCR, before sorting, deduplication, or any
    # field extraction. Each row is the merged text exactly as OCR produced it.
    with open(RAW_OUTPUT_FILE, "w", newline="", encoding="utf-8") as raw_file:
        raw_writer = csv.writer(raw_file)
        raw_writer.writerow(["processed_text", "source_file"])
        raw_writer.writerows(raw_rows)

    print(f"Raw OCR data ({len(raw_rows)} rows) saved to {RAW_OUTPUT_FILE}")

    # ── Step 4: Sort by percentage descending ──
    #
    # Define a helper function that converts a row's percentage string to a
    # negative float for sorting. Python's .sort() is ascending by default
    # (smallest first), so negating the value makes the largest percentages
    # appear first (e.g. 51.37% before 0.05343%).
    #
    # The original data in all_rows is not modified — the negative value is
    # only used as a sort key (a temporary comparison value).
    def sort_key(row):
        try:
            # row[0] is the percentage string, e.g. "51.37%".
            # .rstrip('%') removes the trailing "%" so float() can parse it.
            # The minus sign negates the value for descending order.
            return -float(row[0].rstrip('%'))

        except (ValueError, AttributeError):
            # If the percentage is missing or malformed (e.g. empty string ""),
            # float() would raise a ValueError. Returning 0 places these rows
            # at the end of the sorted list rather than crashing the program.
            # AttributeError catches the case where row[0] is None.
            return 0

    all_rows.sort(key=sort_key)

    # ── Step 5: Write to CSV ──
    #
    # Open the output file for writing. The parameters:
    #   "w"              = write mode (creates the file, or overwrites if it
    #                      already exists)
    #   newline=""       = prevents csv.writer from adding extra blank lines
    #                      on Windows (a known quirk of python's csv module)
    #   encoding="utf-8" = supports special characters like apostrophes in
    #                      item names (e.g. "Got 'em", "Groovin'")
    #
    # The "with" statement is a context manager: it guarantees that the file
    # is properly closed when the block ends, even if an error occurs. Without
    # "with", you'd need to manually call csvfile.close().
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        # csv.writer handles CSV formatting: it adds commas between fields and
        # wraps fields in quotes if they contain commas or special characters.
        # e.g. the name 'Who, Me?' would be written as '"Who, Me?"' to avoid
        # the comma being interpreted as a field separator.
        writer = csv.writer(csvfile)

        # Write the header row (column names).
        writer.writerow(["percentage", "item_name", "me_value", "source_file"])

        # Write all data rows at once. Each element of all_rows is a
        # tuple like ("0.05343%", "Got 'em", "2", "image01.png") which
        # becomes one line in the CSV.
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} rows written to {OUTPUT_FILE}")


# This is a python convention. When you run a .py file directly (e.g.
# "python3 process_images.py"), python sets the special variable __name__ to
# the string "__main__". When the file is imported by another script (e.g.
# "from process_images import parse_line"), __name__ is set to "process_images"
# instead. This if-statement means: "only run main() if this file is being
# executed directly, not if it's being imported as a module."
if __name__ == "__main__":
    main()
