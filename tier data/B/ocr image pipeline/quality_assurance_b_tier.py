"""
quality_assurance_b_tier.py
===========================
Reads the raw OCR output CSV and produces a normalised B-tier items CSV that
matches the column structure of a_tier_items_normalised.csv.

Input:  ./dockerised image processor/ocr_results.csv
Output: ../b_tier_items_normalised.csv  (one folder above this script)

Steps applied:
  1. Strip % symbols and fix OCR artefacts in item names
       - trailing " ." removed
       - incomplete ellipses ".." completed to "..."
       - Roman numeral suffixes corrected for known item families (l/i/1 → I, etc.)
       - word-initial I misread as ll corrected (e.g. "llustration" → "Illustration")
  2. Rename/restructure columns into target schema
  3. Fill 'id' for permanent rows as "permanent (X pts)"
  4. Deduplicate on the final 'id' value (exact match after all cleaning)
  5. If percentages don't sum to 89.5, recompute the two truncated-display
     groups (0.05343 and 0.02964) as exact fractions, then warn if still off
"""

import csv
import difflib
import re
from fractions import Fraction
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent

INPUT_FILE  = SCRIPT_DIR / "dockerised image processor" / "ocr_results.csv"

# "folder ABOVE where the script runs from"
OUTPUT_FILE = SCRIPT_DIR.parent / "b_tier_items_normalised.csv"


# ── Constants ──────────────────────────────────────────────────────────────────

FIELDNAMES = ["id", "subset", "type", "percentage_probability", "points"]

TARGET_SUM = Fraction("89.5")

# The two per-item display values that are truncated by the game UI.
# Groups are identified by matching against these Fraction values so the
# comparison works whether the CSV stores "0.05343" or a fraction string
# from a previous run.
EMOTE_PCT = Fraction("0.05343")
ICON_PCT  = Fraction("0.02964")

# Each entry maps a regex pattern (prefix + noisy numeral + optional suffix)
# to the set of valid Roman numerals for that item family.  Used by
# fix_roman_suffix() to validate and correct OCR-mangled numeral tokens.
_ROMAN_SUFFIX_CONTEXTS = {
    # Hangzhou 2022 emotes and icons are numbered I through X.
    r'^(Hangzhou 2022 )([IVXivxlL1]+)( Icon)?$': [
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    ],
    # Pentakill albums I, II and III all have numbered items in this tier.
    r'^(Pentakill )([IVXivxlL1]+)( Icon)?$': ["I", "II", "III"],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def fix_roman_suffix(name):
    """
    Correct OCR character errors in Roman numeral suffixes for known item
    name patterns (see _ROMAN_SUFFIX_CONTEXTS above).

    OCR errors seen in practice:
      - l (lowercase L) and i (lowercase i) misread instead of I (uppercase I)
      - digit 1 misread instead of I
      - extra characters inserted by OCR (e.g. "ViIll" for "VIII")
      - missing characters (e.g. "Il" for "III" in Pentakill Il Icon)
      - space lost before I so OCR merges it with preceding text as T
        (e.g. "Hangzhou 2022T Icon" → "Hangzhou 2022 I Icon")

    Strategy for each matching pattern:
      1. Restore the missing space in the Hangzhou "2022T" edge case.
      2. In the numeral portion replace every l, i, or 1 with I and
         uppercase the result.  Valid Roman numerals (I–X) need only
         I, V, and X, so this substitution cannot introduce new errors.
      3. If the normalised string is not in the known valid set (e.g.
         "VIIII" from an extra character, or "II" from a missing one),
         pick the closest match via difflib.  This is safe because the
         function only acts on strings that already match a known prefix
         pattern — never on free-form text.
    """
    # "Hangzhou 2022T ..." — OCR dropped the space before I and read it as T
    name = re.sub(r'(Hangzhou 2022)T\b', r'\1 I', name, flags=re.IGNORECASE)

    for pattern, valid_numerals in _ROMAN_SUFFIX_CONTEXTS.items():
        m = re.match(pattern, name)
        if not m:
            continue

        prefix, numeral, suffix = m.group(1), m.group(2), m.group(3) or ""

        # Replace l (lowercase L), i (lowercase i), and digit 1 with uppercase I.
        # .upper() then normalises any remaining lowercase (e.g. v → V).
        # Existing uppercase I, V, X are unchanged by both operations.
        normalized = re.sub(r'[li1]', 'I', numeral).upper()

        if normalized in valid_numerals:
            return prefix + normalized + suffix

        # Normalised string is not a recognised numeral (extra/missing character
        # due to OCR noise).  Find the closest known numeral by sequence similarity.
        best = difflib.get_close_matches(normalized, valid_numerals, n=1, cutoff=0.0)
        return prefix + (best[0] if best else normalized) + suffix

    return name


# ── Main ───────────────────────────────────────────────────────────────────────

def main():

    # ── Read raw OCR CSV ───────────────────────────────────────────────────────

    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))


    # ── Step 1: Remove % symbols ───────────────────────────────────────────────
    # replace() is a no-op if % is already absent, so this is safe to re-run.

    for row in raw_rows:
        row["percentage"] = row["percentage"].replace("%", "")
        row["item_name"] = row["item_name"].replace(" .", "")  # trailing periods
        row["item_name"] = re.sub(r'(?<!\.)\.\.(?!\.)', '...', row["item_name"])  # incomplete ellipses
        row["item_name"] = fix_roman_suffix(row["item_name"])
        row["item_name"] = re.sub(r'\bll', 'Il', row["item_name"])  # word-initial I misread as l (e.g. "llustration" → "Illustration")


    # ── Steps 2 & 3: Restructure columns ──────────────────────────────────────
    #
    # Column mapping:
    #   percentage  → percentage_probability   (2a)
    #   source_file → (dropped)                (2b)
    #   item_name   → id                       (2c)
    #   (new)       → subset = 3               (2d)
    #   me_value    → points                   (2e)
    #   (new)       → type  (2f: empty item_name → "permanent", else "one_time")
    #
    # Step 3: permanent rows have no meaningful item name so id is built from
    # type and points: "permanent (X pts)"

    processed = []

    for row in raw_rows:
        pct       = row["percentage"]
        item_name = row["item_name"]
        points    = row["me_value"].strip()
        # source_file is intentionally not carried forward

        # All B-tier one_time items carry a fixed Mythic Essence value of 2.
        # OCR occasionally misses the ME text box entirely; default to "2" here
        # so downstream scripts can always parse points as a float.
        if not points and item_name != "":
            points = "2"

        type_val = "permanent" if item_name == "" else "one_time"

        if type_val == "permanent":
            id_val = f"Permanent ({points} pts)"
        else:
            id_val = item_name

        processed.append({
            "id":                     id_val,
            "subset":                 "3",
            "type":                   type_val,
            "percentage_probability": pct,
            "points":                 points,
        })


    # ── Step 4: Deduplicate on final 'id' ─────────────────────────────────────
    #
    # Deduplication is performed here — after all name cleaning — rather than
    # in the OCR script.  In the OCR script the key is built from raw,
    # unprocessed strings; OCR variants of the same item (e.g. "VIIl" vs
    # "VIII" before the Roman numeral fix) produce different keys and both
    # survive.  At this point the id field is fully cleaned, so exact matching
    # is reliable.
    #
    # Note: fuzzy matching on id was considered for handling arbitrary OCR
    # character substitutions, but intentionally similar item names (e.g.
    # "Hangzhou 2022 VII" and "Hangzhou 2022 VIII") differ by only one
    # character and would collapse incorrectly under any reasonable threshold.
    # The Roman numeral correction above is instead the targeted fix for the
    # known Hangzhou substitution patterns.

    seen_ids  = set()
    deduped   = []
    for row in processed:
        if row["id"] not in seen_ids:
            seen_ids.add(row["id"])
            deduped.append(row)

    duplicates_removed = len(processed) - len(deduped)
    if duplicates_removed:
        print(f"Deduplication removed {duplicates_removed} duplicate row(s)")
    processed = deduped


    # ── Step 5: Check and (if needed) correct percentage_probability sum ───────

    total = sum(Fraction(r["percentage_probability"]) for r in processed)

    if total != TARGET_SUM:

        permanent_sum = sum(
            Fraction(r["percentage_probability"])
            for r in processed if r["type"] == "permanent"
        )
        # Each of the two variable tiers receives an equal share of what remains.
        # Dividing by 2 then by the group's row count gives the exact per-item value.
        tier_total = (TARGET_SUM - permanent_sum) / 2

        # 5a: icon tier — rows matching 0.02964
        icon_rows = [r for r in processed
                     if Fraction(r["percentage_probability"]) == ICON_PCT]
        if icon_rows:
            per_item = tier_total / len(icon_rows)
            for r in icon_rows:
                r["percentage_probability"] = str(per_item)

        # 5b: emote tier — rows matching 0.05343
        emote_rows = [r for r in processed
                      if Fraction(r["percentage_probability"]) == EMOTE_PCT]
        if emote_rows:
            per_item = tier_total / len(emote_rows)
            for r in emote_rows:
                r["percentage_probability"] = str(per_item)

        # 5c: final verification
        total_after = sum(Fraction(r["percentage_probability"]) for r in processed)
        if total_after != TARGET_SUM:
            print("warning: percentage_probability does not sum to 89.5")


    # ── Write output CSV ───────────────────────────────────────────────────────

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(processed)

    print(f"Done — {len(processed)} rows written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()