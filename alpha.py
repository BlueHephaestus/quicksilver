"""
App for handling population analysis of datasets.
    currently designed for UPHL.

Currently meant to handle stuff like the default.csv file,
	however I am making it able to have files uploaded different than the default.
"""

import streamlit as st
import numpy as np
import pandas as pd
from cross_referencing import *
import sys

# Update styling for multiselect so it's not tiny
st.markdown("<style>.stMultiSelect > label {font-size:105%; font-weight:bold} </style>",unsafe_allow_html=True)

"# UPHL Population Analysis Tests"
default_input_file = "default.csv"
input_file = st.file_uploader("Choose a file for Population Analysis. If no file is chosen, the default will be used.", type=["txt","csv"])

#@st.cache(allow_output_mutation=True) # dont enable, this is a problem for now
def get_input_file():
    if input_file is not None:
        try:
            # Note: Streamlit doesn't let you rename the index easily, it shows in other apps but not streamlit.
            df = pd.read_csv(input_file, index_col=0)
        except:
            "Unable to load file. Did you make sure it was a CSV formatted file?"
            sys.exit()
    else:
        df = pd.read_csv(default_input_file, index_col=0)
    return df

df = get_input_file()

"### Raw Input File"
df

##### PREFIXES #####
# We now have input file
# Let them choose the types of samples to use, via the prefixes on 
# the "EpisodeNumber" column. F is only one checked by default,
# however we may also have AAC and S. These are the supported options
#"### Sample Types for Analysis:"
#prefixes = ["F","AAC","S"]
#sample_types = [st.checkbox(prefix,True) if prefix == "F" else st.checkbox(prefix) for prefix in prefixes]
prefixes = ["F","AAC","S"]
chosen_prefixes = st.multiselect("Choose which Sample Types to use for this Analysis (by prefix):", prefixes, ["F"])

@st.cache(allow_output_mutation=True)
def filter_by_prefix(df, prefixes, chosen_prefixes):

    # Keep only the ones with the prefixes selected
    # can't do a negative mask because there are prefixes not in our list.
    #
    # Get samples with correct prefix
    masks = [df["EpisodeNumber"].str.startswith(prefix) for prefix in chosen_prefixes]
    # Combine all of them with logical OR into one
    mask = masks[0]
    if len(masks) > 1:
        for m in masks[1:]:
            mask = mask | m
    df = df[mask]

    #for prefix in prefixes:
        #if prefix not in chosen_prefixes:
            #df.drop(df[df["EpisodeNumber"].str.startswith(prefix)].index, inplace=True)
    return df

# df now updated
df = filter_by_prefix(df, prefixes, chosen_prefixes)
"### Resulting File:"
df

##### HEADERS #####
# Choice of all headers to include in resulting dataset. 
container = st.container() # lets us have these out of order displayed but affect each other
no_ratios = st.checkbox("Exclude Ratio (A/B) Columns")
no_nans = st.checkbox("Exclude Columns with N/A Values")
numerical_only = st.checkbox("Only Include Columns with Numerical Values")

headers = [str(col) for col in df.columns]
if no_ratios:
    # remove all ratio headers currently selected
    headers = [h for h in headers if "/" not in h]

if no_nans:
    headers = [h for h in headers if not df[h].isnull().values.any()]

if numerical_only:
    headers = [h for h in headers if np.issubdtype(df[h].dtype, np.number)]

chosen_headers = container.multiselect("Choose which Column Headers to Keep:", headers, headers)

@st.cache(allow_output_mutation=True)
def filter_by_header(df, headers, chosen_headers):
    return df[chosen_headers].copy()

df = filter_by_header(df, headers, chosen_headers)
"### Resulting File:"
df

"""#### This file will be used for input to the upcoming analysis. 
If you wish to change it, do so above.
## Cross Referencing
Choose your input and output variables, as well as type of graph, to perform a cross-reference
    graph generation of the data selected and open it in a new tab."""

# User will now be able to add an arbitrary amount of data cross-reference 
# graphs for analyzing the above data.
# We tried getting it so we could have this remain in one page, however streamlit really
# didn't like having tabs with dynamic content, that could be changed according to each 
# cross-references values selected in each.
#
# So instead we let them pick their parameters, and generate new ones with the Generate button.

# NOTE TEMPORARY DEFAULTS TO SPEED DEBUGGING
inputs = st.multiselect("Input / Independent Variables: ", chosen_headers)
outputs = st.multiselect("Output / Dependent Variables: ", [oh for oh in chosen_headers if oh not in inputs])
graph_type = st.radio("Graph Type: ", ["Line Graph", "Scatter Plot", "Pie Chart", "Table", "Histogram of Bins", "2D Heatmap of Bins"], 1)

is_valid = is_valid_inputs(inputs, outputs, graph_type)

# TODO doesn't work with customising sliders, won't copy slider settings at the moment.
# can't work with customisable, but it should copy the settings.
st.button("Generate in New Tab", on_click=cross_reference, args=(df, inputs, outputs, graph_type), kwargs=({"new_tab":True}), disabled=not is_valid)
cross_reference(df, inputs, outputs, graph_type, new_tab=False)


# TODO improve caching and performance
# TODO improve "unique element" threshold for pie charts, so that it will make an "other" category automatically
# TODO change table to be a modified sort of 2d heatmap
# TODO make bin changing more intuitive
# TODO add color coding and other multiple-line graphs
#
# TODO add TRANSFORMATIONS to the data before cross reference.





