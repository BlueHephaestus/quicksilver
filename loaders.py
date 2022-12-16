import streamlit as st
import pandas as pd
import numpy as np
import sys
from Constants import *

"""
This file handles the various types of dataset types we will have from various sources.

It has to automagically handle whichever is thrown at it and work.
These are the possible formats:
    1. One sample per row, headers on first row, analytes / features are per-column.
        Used by: New York, sometimes Utah
        Calling these ALPHAS
    2. Beginning and end of file has two control samples, which need to be discarded until use for calibration curves.
        Headers are after these.
        Otherwise, same as #1
        Used by: Texas
        Calling these BRAVOS
    3. One feature/analyte per ROW, one sample made up of multiple rows, headers on first row, multiple files possible.
        Used by: Utah
        Calling these CHARLIES
        
If multiple files are specified, we always treat them the same:
    Try to concatenate them all together so we can form a larger dataset. 
    Each will be handled the same as a usual one, except the headers will be asserted to be the same between them.
        we might be able to use recursion here, actually. we can actually do that yea
"""
def nan_rows_count(df):
    # Return # of empty rows, aka nan rows
    return len(df[df.isna().all(axis=1)])


#@st.cache()
def load_input_data(input_files):
    print("Reading input file from disk...")
    print(input_files)
    if len(input_files) == 0:
        # No file given, use default
        raw_data = pd.read_csv(DEFAULT_INPUT_FILE, index_col=0)
        st.write("Using default file, since no file provided yet.")
        return raw_data, DEFAULT_INPUT_FILE

    if len(input_files) == 1:
        # One file given, load it.
        # First thing we have to do is determine if it's an alpha bravo or charlie.
        input_file = input_files[0]
        if os.path.splitext(input_file.name)[-1] in [".xls", ".xlsx"]:
            df = pd.read_excel(input_file)
        else: #txt or csv
            df = pd.read_csv(input_file)

        # metadata for determining
        cols = [col.lower() for col in df.columns]
        head, tail = df.head(10), df.tail(10)

        # If it has analytename as a column then we assume that's a charlie
        if "AnalyteName" in cols:
            # CHARLIE
            pass
            return

        # It's now either a bravo or alpha, if it's got two empty rows in it's head and tail it's a bravo
        total_nan_rows_head = nan_rows_count(head)
        total_nan_rows_tail = nan_rows_count(tail)
        if total_nan_rows_head >= 2 and total_nan_rows_tail >= 2:
            # BRAVO
            # Remove all rows up to and including the first two empty rows, and from the last two to the end.
            data = df.values
            isnanrow = lambda row: all(pd.isnull(row))
            head_nan_rows = 0
            tail_nan_rows = 0

            # HEAD HANDLING
            for i,row in enumerate(data):
                if isnanrow(row):
                    head_nan_rows += 1
                if head_nan_rows == total_nan_rows_head:
                    # End of disposable head section
                    # Cut off the head and stop iter
                    data = data[i+1:]
                    break

            # TAIL HANDLING
            for i, row in enumerate(reversed(data)): #reminder, index will increase like normal, but rows are reverse
                if isnanrow(row):
                    tail_nan_rows += 1
                if tail_nan_rows == total_nan_rows_tail:
                    # End of disposable tail section
                    # Cut off the tail and stop iter
                    data = data[:-i-1]
                    break

            # Data is now same as usual, we re-pandas it and return
            df = pd.DataFrame(data[1:], columns=data[0])

            return df, input_file

        # ALPHA
        # It's now an alpha, which means it's normal and ez, return as usual.
        # Note: Streamlit doesn't let you rename the index easily, it shows in other apps but not streamlit.
        return df, input_file






    if len(input_files) > 1:
        # Multiple files given, load each and concatenate - print error if headers are mismatching, however.

        try:
            pass
        except:
            "Unable to load file. Did you make sure it was a CSV formatted file?"
            sys.exit()
