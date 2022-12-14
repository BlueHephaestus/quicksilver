"""
App for handling population analysis of datasets.
    currently designed for UPHL.

Currently meant to handle stuff like the default.csv file,
	however I am making it able to have files uploaded different than the default.
OI FUTURE SELF

YOOOO THAT WORKS! WE DON'T NEED TO LOAD THE FILE EVERY TIME NOW, I THINK.

though i did just test it and it isn't loading at all so long as we have the st.cache() there... 
    maybe more will crystallize over time as we work on it and mess with it.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import power_transform
from Constants import *
from sessions import *
from graphs import *
from tables import *
from sections import *
from loaders import *
import sys, traceback
st.set_page_config(page_title="Quicksilver", layout="wide")

# Update styling for multiselect so it's not tiny
# st.markdown("""<style>.stMultiSelect > label {
#             font-size:105%; font-weight:bold
#             } </style>""",unsafe_allow_html=True)
# Make sure large multiselects go to scroll rather than filling the page
st.markdown("""<style>.stMultiSelect > div {
            max-height:300px;
            overflow-y:scroll;
            } </style>""",unsafe_allow_html=True)
# Make labels for selectbox not tiny either
# st.markdown("""<style>.stSelectbox > label {
#             font-size:105%; font-weight:bold
#             } </style>""",unsafe_allow_html=True)

# Update fonts for other elements
# st.markdown("""<style>.stSlider > label {
#             font-size:105%; font-weight:bold
#             } </style>""",unsafe_allow_html=True)
# st.markdown("""<style>.stNumberInput > label {
#             font-size:105%; font-weight:normal
#             } </style>""",unsafe_allow_html=True)

# Replicate the actual form template without it's restrictions
st.markdown("""<style>div[data-testid="stHorizontalBlock"] {
            #border: 1px solid darkslategray;
            #padding: 10px;
            #border-radius: 5px;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: calc(1em - 1px);
            box-sizing: border-box;
            } </style>""",unsafe_allow_html=True)

# Big fucking buttons.
st.markdown("""<style>div.stButton > button:first-child {
            #color: #fff;
            #background-color: #28a745;
            #background-color: #00f;
            font-size:200%;
            font-weight: bold;
            #height:5em;
            #width:10em;
            }
            """, unsafe_allow_html=True)



"# Data Preparation"

input_file = st.file_uploader("Choose file(s) for Population Analysis. If no file(s) is chosen, the default will be used.", accept_multiple_files=True, type=["txt","csv","xls","xlsx"])

# DO NOT MODIFY DATA_REF, ONLY DE-REFERENCE TO GET OUR DATAFRAME FOR USE IN THE FUNCTION
# OTHERWISE WE CAN'T CACHE IT FOR LATER USE ON RE-RUNS
data_master, accession_col, data_fname, data_type = load_input_data(input_file)

@st.cache(allow_output_mutation=True)
def load_session(data_master, accession_col):
    print("RE INIT OF SESSION")
    return Session(data_master, accession_col)

session = load_session(data_master, accession_col)

print("INPUT FILE CHANGED: ", accession_col)
data = data_master.copy()
# TODO: we might need something for when we specifically change the input data, so maybe put stuff there to make sure we reset everything when we reload a new file.

# Get identifier column data
accessions = session.data_master[accession_col]

# Check if we have "prefix" filtering, this is a Utah-specific feature.
accession_filtering = accession_col == "EpisodeNumber"

f"#### Current Data: "
print("Writing current data to container")
data_container = st.empty()
data_container.container().write(session.data)
#data_container.container().write(f"{len(session.data)} samples in dataset.")

##### DATA PREPARATION AND MODIFICATION TIME #####
def update_data():
    """
    This way we ONLY UPDATE THE DATAFRAME AND RUN THOSE COSTLY FUNCTIONS ONCE WE'RE SURE

    Called after any filters for data modification are implemented, and applies them.
    :return:
    """
    print("Updating Data...")
    print(f"{len(data_master)} Samples at Initialization.")

    ####### FILTERING #######
    ##### ROW FILTERING #####

    ### PREFIX FILTERING ###
    # Keep only the rows with our prefixes selected
    # can't do a negative mask because there are prefixes which are not in our list.
    #
    # Get samples with correct prefix
    if accession_filtering:
        masks = [data_master[session.accession_col].str.startswith(prefix) for prefix in session.row_filter_opts]
        # Combine all of them with logical OR into one
        if len(masks) != 0:
            mask = np.logical_or.reduce(masks)
            data = data_master[mask]
        else:
            # Empty dataframe
            data = data_master[0:0]
            return
    else:
        data = data_master.copy()
    print(f"{len(data)} Samples remaining after Row Prefix Filtering.")

    ### CATEGORICAL FILTERING ###
    # If they chose a categorical divisor, and deselected one or more of the categories,
    # we now filter down the dataset to only be the remaining categories.
    print(session.categorical_col, session.categorical_divisions, session.categories)
    if session.categorical_col != "None" and np.sum(session.categorical_divisions) != len(session.categories):
        # Ensure they didn't deselect everything, if so we ignore (we warned them)
        if np.sum(session.categorical_divisions) != 0:

            # For all selected categorical divisions, enforce that as a mask for samples.
            masks = []

            # I could make this a one-line list comprehension but that would be unnecessarily and harder to read.
            for category, division in zip(session.categories, session.categorical_divisions):
                if division:
                    masks.append(data[session.categorical_col] == category)
            mask = np.logical_or.reduce(masks)
            # So we don't destroy any prefix filtering that was already done, we use data again.
            data = data[mask]

    print(f"{len(data)} Samples remaining after Categorical Filtering.")
    ##### COLUMN FILTERING #####
    # Don't remove columns until the very end, so we can still do our row filtering based on column values
    data = data[session.col_filter_opts]

    ####### TRANSFORMATIONS ########
    # Now that all data is filtered, we operate on what's left with various data transformations

    ### GET STATISTICS FOR NUMERICAL COLUMNS ###
    # Remove all non-numerical so we can operate on the numbers and put them back later
    # Additionally get the columns so we can retain their order
    cols = data.columns.tolist()
    cols_non_numerical = data.select_dtypes(exclude=[np.number]).columns.values
    cols_numerical = data.select_dtypes(include=[np.number]).columns.values

    data_non_numerical = data[cols_non_numerical]
    data = data[cols_numerical]
    # Convert all numerical columns to matching datatype to avoid problems in later use.
    data = data.astype(np.float64)

    mean = data.mean()
    std = data.std()
    median = data.median()
    min = data.min()
    max = data.max()
    range = max-min

    ##### MISSING DATA HANDLING #####
    if session.missing_data_opt == "Exclude Rows with N/A values (default)":
        data = data.dropna()

    elif session.missing_data_opt == "Use Linear Regression to Interpolate Missing Values":
        data = data.interpolate()

    elif session.missing_data_opt == "Replace with Mean Value":
        data = data.fillna(mean)

    elif session.missing_data_opt == "Replace with Median Value":
        data = data.fillna(median)

    ##### LOGARITHMIC SCALING #####
    if session.log_opt == "None (default)":
        # what did you expect?
        pass

    elif session.log_opt == "Log Base 10":
        for col in data:
            data[col] = np.log10(data[col])

    elif session.log_opt == "Log Base 2":
        for col in data:
            data[col] = np.log2(data[col])

    elif session.log_opt == "Log Base e (natural log)":
        for col in data:
            data[col] = np.log(data[col])

    ##### SCALING / NORMALIZATION #####
    if session.scaling_opt == "None (default)":
        # what did you expect?
        pass

    elif session.scaling_opt == "Z-Score Normalization":
        # Convert all values to their z-scores, i.e. normal distribution representatives.
        data = (data-mean) / std

    elif session.scaling_opt == "Min-Max Scaling":
        # Ensure all are in the range 0-1, with min value being 0 and max being 1.
        # We enforce a distribution into that range.
        # Similar to z score but without distribution stuff
        data = (data - min) / range

    elif session.scaling_opt == "Multiple-of-Mean Standardization":
        # Represent all data as multiples of the mean value.
        # Mean value is 1, and for any value, multiplying it by the mean will obtain the original.
        data = data / mean

    elif session.scaling_opt == "Multiple-of-Median Standardization":
        # Same as above
        data = data / median

    elif session.scaling_opt == "Multiple-of-Standard-Deviation Standardization":
        # Same as above
        data = data / std

    elif session.scaling_opt == "Percentile of Max Standardization":
        # Convert all values to their percentile, such that for a given percentile p,
        # p % of the values in the list are below the value of that percentile.
        # 50% percentile would be the median of the list (sorted).
        # We keep them in the range 0-1 rather than 0-100, however.
        for col in data:
            data[col] = stats.rankdata(data[col], "average")/len(data[col])

    ##### DATA TRANSFORMATIONS #####
    if session.transformation_opt == "None (default)":
        # lookin' sus
        pass

    elif session.transformation_opt == "Yeo-Johnson":
        for col in data:
            data[col] = power_transform(data[col].values.reshape(-1,1), method="yeo-johnson")

    elif session.transformation_opt == "Box-Cox (Only applied to columns with all positive values, otherwise columns will be skipped)":
        # Can only be applied to positive values, if data is 0 then add epsilon to it.
        for col in data:
            if (data[col] >= 0).all():
                if (data[col] == 0).any():
                    # Add epsilon to let power transform happen and avoid error
                    data[col] += 1e-8
                data[col] = power_transform(data[col].values.reshape(-1,1), method="box-cox")


    # PUT BACK NON NUMERICAL COLUMNS
    # Number ops are done, put them back with updated values.
    data[cols_non_numerical] = data_non_numerical
    # Preserve original ordering of columns
    data = data[cols]

    # Update displayed dataframe, always do this last
    # TODO: Do we need to store data in the session state if we are doing container write?
    session.data = data
    data_container.empty()
    data_container.write(session.data)

col1,col2 = st.columns(2)
with col1:
    ##### HEADERS #####
    # Choice of all headers to include in resulting dataset.
    container = st.container()  # lets us have these out of order displayed but affect each other
    no_ratios = st.checkbox("Exclude Ratio (A/B) Columns")
    no_nans = st.checkbox("Exclude Columns with N/A Values")
    numerical_only = st.checkbox("Only Include Columns with Numerical Values")
    print("UPDATING FORM DATA")
    headers = [str(col) for col in data.columns]
    if no_ratios:
        # remove all ratio headers currently selected
        headers = [h for h in headers if "/" not in h]

    if no_nans:
        headers = [h for h in headers if not data[h].isnull().values.any()]

    if numerical_only:
        headers = [h for h in headers if np.issubdtype(data[h].dtype, np.number)]

    session.col_filter_opts = container.multiselect("Choose which Column Headers to Keep:", headers, headers)

    # Let them choose the types of samples to use, via the prefixes on
    # the accession_col column. F is only one checked by default,
    # however we may also have AAC and S. These are the supported options
    if accession_filtering:
        prefixes = ["F", "AAC", "S", "R"]
        session.row_filter_opts = st.multiselect("Choose which Sample Types to use for this Analysis (by prefix):", prefixes, ["F"])

    #### CATEGORICAL HEADER FILTERING ####
    # Give them the option to choose a column to divide the dataset into.
    # Unless they pick None, this will result in color coding of samples, and possibly also filtering down the dataset.

    # Get categorical columns, via if they have less than 10 unique values in their set.
    categorical_columns = ["None"] + [h for h in headers if len(pd.unique(data[h])) < 10]
    session.categorical_col = st.selectbox("Choose a Categorical Column to label the dataset with:", categorical_columns)
    if session.categorical_col != "None":
        #print("CATEGORICAL COL IS NOT NONE", session.categorical_col, type(session.categorical_col))
        # I can use multiselect for this but I don't like the UI as much as multiple checkboxes.
        st.write("Choose which categories of samples to include in the dataset (scatterplots will be color-coded)")
        session.categories = sorted(pd.unique(data[session.categorical_col]))
        session.categorical_divisions = []
        for category in session.categories:
            session.categorical_divisions.append(st.checkbox(str(category), value=True))

        if np.sum(session.categorical_divisions) == 0:
            st.warning("Removing all categorical divisions will result in an empty dataset and cause errors." + \
                       "\n\nTo avoid this, the system will now ignore this category.")


with col2:
    session.missing_data_opt = st.selectbox(
        "Missing Data Handling",
        ("Exclude Rows with N/A values (default)",
         "Use Linear Regression to Interpolate Missing Values",
         "Replace with Mean Value",
         "Replace with Median Value",
         ))
    session.log_opt = st.selectbox(
        "Logarithmic Scaling Method",
        ("None (default)",
         "Log Base 10",
         "Log Base 2",
         "Log Base e (natural log)",
         ))
    session.scaling_opt = st.selectbox(
        "Scaling / Normalization Method",
        ("None (default)",
         "Z-Score Normalization",
         "Min-Max Scaling",
         "Multiple-of-Mean Standardization",
         "Multiple-of-Median Standardization",
         "Multiple-of-Standard-Deviation Standardization",
         "Percentile of Max Standardization",
         ))
    session.transformation_opt = st.selectbox(
        "Data Transformations (Make sure your data is in the correct ranges for your chosen transform)",
        ("None (default)",
         "Yeo-Johnson",
         "Box-Cox (Only applied to columns with all positive values, otherwise columns will be skipped)",
         ))

    # TODO: Add reset button?
    # submitted = st.form_submit_button("UPDATE DATA")
    st.button("UPDATE DATA", on_click=update_data)
    st.success("Sometimes requires pressing again, e.g. if refreshing the graph.")

"# Data Visualization & Analysis"
"Using the above dataframe, you can now visualize and analyse the resulting data however you like."


gfcol1, gfcol2 = st.columns(2)


#print([data[col].dtype for col in data.columns])
numeric_cols = [col for col in session.data.columns if np.issubdtype(session.data[col].dtype, np.number) and col != session.categorical_col]
print("COLUMNS", session.data.columns)
#print("CATEGORICAL COL", session.categorical_col, type(session.categorical_col))
with gfcol1:
    # Only allow choice of numeric columns, for now.
    session.graph_type = st.selectbox("Choose the type of graph generated: ", GRAPH_TYPES)
    session.x.col = st.selectbox("X-Axis Variable: ", numeric_cols, index=len(numeric_cols)//2)

    # Remove second variable if single variable analysis chosen
    session.single_var = session.graph_type == GRAPH_TYPES[1]
    session.thresholding = session.graph_type == GRAPH_TYPES[0]
    if not session.single_var:
        session.y.col = st.selectbox("Y-Axis Variable: ", numeric_cols, index=len(numeric_cols)//2+1)
        session.scatter_enable = st.checkbox("Render graph as scatterplot instead of histogram (this will lower performance)")

with gfcol2:
    st.markdown("#")
    #st.button("GENERATE GRAPH", on_click=lambda x:x, args=(input, output))

if session.thresholding:
    session, grid_ns, grid_ps = generate_threshold_section(session)
elif session.single_var:
    session = generate_singlevar_section(session)

# Final container! Update with the values for our two columns in question,
# Displaying all the data in a table data dump (or in a copy-paste format as well.)
st.markdown("# Analysis Values:")
acol1, acol2, acol3, acol4 = st.columns(4)
with acol1:
    st.markdown(f"""
    **Initial Number of Samples**: {len(data_master)}

    **Filtered Number of Samples**: {len(session.data)}

    {f"**Prefix Filters**: {session.row_filter_opts}" if accession_filtering else ""}

    **Categorical Column**: {session.categorical_col}
    
    **Categories Used**: {[category for i,category in enumerate(session.categories) if session.categorical_divisions[i]] if session.categorical_col != "None" else "None"}
    
    ### Transformations Used:
    
    **Missing Data Handling**: {session.missing_data_opt}
    
    **Logarithmic Scaling Method**: {session.log_opt}
    
    **Scaling / Normalization Method**: {session.scaling_opt}
    
    **Data Transformations**: {session.transformation_opt}
    
    **Graph / Analysis Type**: {session.graph_type}
    """)

def generate_variable_markdown(var):
    md = f"""

    ### {var.col}

    **Mean**: {var.mean:.4f}

    **STD**: {var.std:.4f}

    **Minimum**: {var.min:.4f}

    **Maximum**: {var.max:.4f}
    """
    if session.thresholding:
        md += f"""
    **Low Threshold Value**: {var.lo:.4f}

    **High Threshold Value**: {var.hi:.4f}
    
    **Low Threshold Percentile**: {num2perc(var.data, var.lo):.4f}
    
    **High Threshold Percentile**: {num2perc(var.data, var.hi):.4f}
    
    **Low Threshold Multiple of STD**: {num2std(var.std, var.lo):.4f}
    
    **High Threshold Multiple of STD**: {num2std(var.std, var.hi):.4f}
    """
    return md


with acol2:
    st.markdown(generate_variable_markdown(session.x))

if not session.single_var:
    with acol3:
        st.markdown(generate_variable_markdown(session.y))

if session.graph_type == "Threshold Testing":
    def generate_markdown_table(grid, col_lambda=lambda n: n):
        gs = ""
        n = len(grid)
        for i,row in enumerate(grid):
            s = "|"
            for j,col in enumerate(row):
                if i == n-1 or j == n-1:
                    s += f" **{col_lambda(col)}** |"
                else:
                    s += f" {col_lambda(col)} |"
            gs += s + "\n"
        return gs

    with acol4:

        # markdown doesn't like tabs
        st.markdown(f"""
### Threshold Areas:

| | | | |
| --- | --- | --- | --- |
{generate_markdown_table(grid_ns)}

# 

### Threshold Area Percentages:

| | | | |
| --- | --- | --- | --- |
{generate_markdown_table(grid_ps, lambda col: f"{col:.2f}%")}
    """)

    st.warning("There is a known issue where the number input fields in the visualization section will not always sync with each other. "
               "Unfortunately this is a bug in the library being used here, so it is currently not avoidable. "
               "Fortunately, the absolute values that are rendered in the visualization section can be verified here, where they are correctly synced. ")

#st.success("Made by Blue Hephaestus")
st.markdown("<div style='text-align: right'> Quicksilver, 2022 <br> Made by Blue Hephaestus for UPHL </div>", unsafe_allow_html=True)

#st.write(f"")
#st.write(f"Filtered Number of Samples: {len(data)}")

#st.session_state

# """#### This file will be used for input to the upcoming analysis.
# If you wish to change it, do so above.
# ## Cross Referencing
# Choose your input and output variables, as well as type of graph, to perform a cross-reference
#     graph generation of the data selected and open it in a new tab."""

# User will now be able to add an arbitrary amount of data cross-reference 
# graphs for analyzing the above data.
# We tried getting it so we could have this remain in one page, however streamlit really
# didn't like having tabs with dynamic content, that could be changed according to each 
# cross-references values selected in each.
#
# So instead we let them pick their parameters, and generate new ones with the Generate button.

# NOTE TEMPORARY DEFAULTS TO SPEED DEBUGGING
# inputs = st.multiselect("Input / Independent Variables: ", chosen_headers)
# outputs = st.multiselect("Output / Dependent Variables: ", [oh for oh in chosen_headers if oh not in inputs])
# graph_type = st.radio("Graph Type: ", ["Line Graph", "Scatter Plot", "Pie Chart", "Table", "Histogram of Bins", "2D Heatmap of Bins"], 1)
#
# is_valid = is_valid_inputs(inputs, outputs, graph_type)
#
# # TODO doesn't work with customising sliders, won't copy slider settings at the moment.
# # can't work with customisable, but it should copy the settings.
# st.button("Generate in New Tab", on_click=cross_reference, args=(data, inputs, outputs, graph_type), kwargs=({"new_tab":True}), disabled=not is_valid)
# cross_reference(data, inputs, outputs, graph_type, new_tab=False)
#

# TODO improve caching and performance
# TODO improve "unique element" threshold for pie charts, so that it will make an "other" category automatically
# TODO change table to be a modified sort of 2d heatmap
# TODO make bin changing more intuitive
# TODO add color coding and other multiple-line graphs
#
# TODO add TRANSFORMATIONS to the data before cross reference.





