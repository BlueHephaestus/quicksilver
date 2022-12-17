import traceback

import pandas.errors
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


@st.cache()
def load_input_data(input_files):
    """

    :param input_files:
    :return:
        dataframe: The data after load.
        accession_col: The column name for the "id" column, identifying each sample.
        data_fname: filename(s) the data was loaded from.
        data_type: the identifier we use for the type of file it was. 'alpha','bravo',...etc.
    """
    print("Reading input file from disk...")
    if len(input_files) == 0:
        # No file given, use default
        raw_data = pd.read_csv(DEFAULT_INPUT_FILE, index_col=0)
        st.write("Using default file, since no file provided yet.")
        return raw_data, raw_data.columns[0], DEFAULT_INPUT_FILE, "alpha"

    if len(input_files) == 1:
        # One file given, load it.
        # First thing we have to do is determine if it's an alpha bravo or charlie.
        input_file = input_files[0]
        df = pandas_read_flexible(input_file)

        # metadata for determining
        cols = [col.lower() for col in df.columns]
        head, tail = df.head(10), df.tail(10)

        # If it has analytename as a column then we assume that's a charlie
        if "analytename" in cols:
            # CHARLIE
            df = load_charlie_data(df)
            return df, df.columns[0], input_file, "charlie"

        # It's now either a bravo or alpha, if it's got two empty rows in it's head and tail it's a bravo
        total_nan_rows_head = nan_rows_count(head)
        total_nan_rows_tail = nan_rows_count(tail)
        if total_nan_rows_head >= 2 and total_nan_rows_tail >= 2:
            # BRAVO
            df = load_bravo_data(df, total_nan_rows_head, total_nan_rows_tail)
            return df, df.columns[0], input_file, "bravo"

        # ALPHA
        # It's now an alpha, which means it's normal, but may have some noise rows of stats or empty rows.
        # We identify these by if they have the second and third columns empty
        df = df[~(pd.isnull(df[df.columns[1]]) & pd.isnull(df[df.columns[2]]))]
        # Note: Streamlit doesn't let you rename the index easily, it shows in other apps but not streamlit.
        return df, df.columns[0], input_file, "alpha"

    if len(input_files) > 1:
        # Multiple files given, load each and concatenate - print error if headers are mismatching, however.
        # Get the metadata of accession column and data type so that we can check if each new one matches.
        # We get this from the first one.
        df, accession_col, data_fname, data_type = load_input_data([input_files[0]])
        dfs = [df]
        columns = df.columns
        data_fnames = [data_fname]
        for input_file in input_files[1:]:
            df, accession_col_new, data_fname_new, data_type_new = load_input_data([input_file])
            columns_new = df.columns
            # Check each attr make sure they're alright

            # Index column deserves specific error case, I suppose.
            if accession_col_new != accession_col:
                st.write(f"File '*{data_fname_new.name}*' has a different identifier column / accession number than the one in '*{data_fname.name}*'. ('{accession_col_new}' vs. '{accession_col}') Reconcile these differences to load both files together.")
                sys.exit()

            # Overall format compare
            if data_type_new != data_type:
                st.write(f"File '*{data_fname_new.name}*' has a different format than the one in '*{data_fnames.name}*'. Reconcile these differences to load both files together.")
                sys.exit()

            # Finally check to make sure the columns sets are the same.
            if len(columns.union(columns_new)) > len(columns):
                st.write(f"File '*{data_fname_new.name}*' has extra and/or different columns than those in '*{data_fname.name}*'. Reconcile these differences to load both files together.")
                sys.exit()

            # This is a compatible dataframe, add it and continue.
            dfs.append(df)
            data_fnames.append(data_fname_new)

        # If we got here, we have successfully got all the dataframes, we can concat them into one big one now.
        df = pd.concat(dfs, ignore_index=True)

        # Return with the attributes we've established are consistent across all files, and the series of data fnames
        # that created this dataframe.
        return df, accession_col, data_fnames, data_type

def pandas_read_flexible(input_file):
    """
    Load from input file, without knowing if it's a CSV, Excel doc, or TSV.
        Attempts to infer the filetype and read accordingly.
    :param input_file: Input file of uncertain filetype
    :return: Dataframe from loading input file.


    IMPORTANT NOTE

    I couldn't find any stackoverflow or bug reports seconding this, but i found when I loaded a file with read_csv,
    evne if the first load was only with nrows=1, the second load would subsequently only occur way later in the file.

    This might have to do with pandas loading and iterating through the buffer a certain point (it was about 5688 rows)
    even when loaded into one row, combined with the streamlit loading these files into memory, idk.

    But we can only reallly load them once, so that's what we're doing now.
    """
    try:
        if os.path.splitext(input_file.name)[-1] in [".xls", ".xlsx"]:
            df = pd.read_excel(input_file, engine='openpyxl')
        else:  # txt or csv or tsv
            # check if tsv or csv via how many columns we get using either, the one with more is assumed correct
            try:
                df_tab = pd.read_csv(input_file, sep='\t')
                tab = df_tab.shape[1]
            except pandas.errors.EmptyDataError:
                tab = 0

            try:
                df_com = pd.read_csv(input_file, sep='\t')
                com = df_com.shape[1]
            except pandas.errors.EmptyDataError:
                com = 0

            if tab > com:
                return df_tab
            else:
                return df_com
    except:
        st.write(f"Error encountered on loading file '*{input_file.name}*':")
        st.markdown("```" + traceback.format_exc() + "```")
        sys.exit()
    return df

def load_charlie_data(df):
    """
    Keep only necessary columns:
    AnalyteName: distinguishing analytes
    PlateID: distinguishing plates
    Specimen: determining if control or first or second
    value: value

    IMPORTANT
    This runs on each of a multiple-plate file input. Each file has it's own plate ID,
    and we are going to assume the plate will not be in later files.

    We convert this one-analyte-per-row, multiple-rows-per-sample
        into a new dataframe with one-analyte-per-column, one-row-per-sample.

    We do keep the controls because we want to leave that up to the accession filtering feature we'll add later.
    """
    # strip to just these columns
    cols = ["AnalyteName", "PlateID", "Specimen", "Value"]
    df = df[cols]
    df = df.to_numpy()
    # Cols = Analyte, Plate, Specimen, Value
    ANALYTE = 0
    PLATE = 1
    SPECIMEN = 2
    VALUE = 3

    # Iterate through each analyte - first loop
    # Get the analytes and the plate(s)
    analytes = np.unique(df[:, ANALYTE])
    samples = np.unique(df[:, SPECIMEN])
    #plates = np.unique(df[:, PLATE])
    # control_sample = lambda row: row[SPECIMEN].startswith("AAAC")
    # each entry has analyte_name: plate: values for each non-control sample with this analyte

    # Start creating new condensed Dataframe
    cols_new = [cols[SPECIMEN], cols[PLATE]]
    cols_new.extend(analytes)

    # Create quick column -> index lookup for creating our list
    cols_i = {col:i for i,col in enumerate(cols_new)}

    # same for rows -> index, using specimen
    rows_i = {row:i for i,row in enumerate(samples)}

    # We will iteratively create the dataframe via a ndarray,
    # and have to fill it in one at a time, and we can't assume any regular order to the analyte
    # or specimen order in the rows.
    # Cue fancy indexing magic

    data = np.zeros((len(samples), len(cols_new)), dtype=np.object)
    data.fill(np.nan)
    for row in df:
        i = rows_i[row[SPECIMEN]]
        j = cols_i[row[ANALYTE]]
        data[i, 0] = row[SPECIMEN]
        data[i, 1] = row[PLATE]
        data[i, j] = row[VALUE]

    df_new = pd.DataFrame(data, columns=cols_new).infer_objects()


    # analyte_plate_pop = {analyte: {} for analyte in analytes}
    # for analyte in analytes:
    #     for plate in plates:
    #         analyte_plate_pop[analyte][plate] = []
    # # Populate dictionary of analyte values.
    # for row in df:
    #     # if not control_sample(row):
    #     analyte_plate_pop[row[ANALYTE]][row[PLATE]].append(row[VALUE])
    # ##### END COPY PASTA
    return df_new



def load_bravo_data(df, total_nan_rows_head, total_nan_rows_tail):
    # Remove all rows up to and including the first two empty rows, and from the last two to the end.
    data = df.values
    isnanrow = lambda row: all(pd.isnull(row))
    head_nan_rows = 0
    tail_nan_rows = 0
    # HEAD HANDLING
    for i, row in enumerate(data):
        if isnanrow(row):
            head_nan_rows += 1
        if head_nan_rows == total_nan_rows_head:
            # End of disposable head section
            # Cut off the head and stop iter
            data = data[i + 1:]
            break
    # TAIL HANDLING
    for i, row in enumerate(reversed(data)):  # reminder, index will increase like normal, but rows are reverse
        if isnanrow(row):
            tail_nan_rows += 1
        if tail_nan_rows == total_nan_rows_tail:
            # End of disposable tail section
            # Cut off the tail and stop iter
            data = data[:-i - 1]
            break
    # Data is now same as usual, we re-pandas it, infer datatypes, and return
    df = pd.DataFrame(data[1:], columns=data[0]).infer_objects()
    return df

# TESTCASES, YOU WILL NEED TO REMOVE THE ".NAME" BIT FROM READ_FLEXIBLE IF YOU WANT TO USE THESE.
# Multiple charlies
# f1 = "data/Utah_Dataset_Example/ZZPlateResultAAAC03202100069-1328139.txt"
# f2 = "data/Utah_Dataset_Example/ZZPlateResultAAAC03202100077-1328157.txt"
# f3 = "data/Utah_Dataset_Example/ZZPlateResultAAAC03202100079-1328179.txt"
# f4 = "data/Utah_Dataset_Example/ZZPlateResultAAAC07202100091-1328083.txt"
# load_input_data([f1, f2, f3, f4])

# Problem case with charlies
# f1 = "data/Utah_Dataset_Example/ZZPlateResultAAAC07202100085-1327806.txt"
# f2 = "data/Utah_Dataset_Example/ZZPlateResultAAAC07202100086-1327807.txt"
# load_input_data([f1,f2])


# Multiple / one bravos
# f1 = "data/Texas_Dataset_Example/mer30608N01084_Conc1.xlsx"
# f2 = "data/Texas_Dataset_Example/gim30815N01469_Conc1.xlsx"
# load_input_data([f1, f2])

# Single alpha
#f1 = "data/NewYork_Dataset_Example/For APHL QI Machine_edited.xlsx"
#load_input_data([f1])

# conflicting alphas. should error.
#f1 = "data/NewYork_Dataset_Example/For APHL QI Machine_edited.xlsx"
#f2 = "/home/blue/Projects/quicksilver/data/default.csv"
#load_input_data([f1,f2])





