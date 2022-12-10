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
from cross_referencing import *
from sklearn.preprocessing import power_transform
import plotly.graph_objects as go
import sys
st.set_page_config(page_title="Quicksilver", layout="wide")

# Update styling for multiselect so it's not tiny
st.markdown("""<style>.stMultiSelect > label {
            font-size:105%; font-weight:bold
            } </style>""",unsafe_allow_html=True)
# Make sure large multiselects go to scroll rather than filling the page
st.markdown("""<style>.stMultiSelect > div {
            max-height:300px;
            overflow-y:scroll;
            } </style>""",unsafe_allow_html=True)
# Make labels for selectbox not tiny either
st.markdown("""<style>.stSelectbox > label {
            font-size:105%; font-weight:bold
            } </style>""",unsafe_allow_html=True)

# Update fonts for other elements
st.markdown("""<style>.stSlider > label {
            font-size:105%; font-weight:bold
            } </style>""",unsafe_allow_html=True)
st.markdown("""<style>.stNumberInput > label {
            font-size:105%; font-weight:normal
            } </style>""",unsafe_allow_html=True)

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
default_input_file = "default.csv"
input_file = st.file_uploader("Choose a file for Population Analysis. If no file is chosen, the default will be used.", type=["txt","csv"])

#@st.cache(allow_output_mutation=True) # dont enable, this is a problem for now

@st.cache()
def read_input_file():
    print("Reading input file from disk...")
    if input_file is not None:
        try:
            # Note: Streamlit doesn't let you rename the index easily, it shows in other apps but not streamlit.
            raw_data = pd.read_csv(input_file, index_col=0)
            return raw_data, input_file
        except:
            "Unable to load file. Did you make sure it was a CSV formatted file?"
            sys.exit()
    else:
        raw_data = pd.read_csv(default_input_file, index_col=0)
        return raw_data, default_input_file

# DO NOT MODIFY DATA_REF, ONLY DE-REFERENCE TO GET OUR DATAFRAME FOR USE IN THE FUNCTION
# OTHERWISE WE CAN'T CACHE IT FOR LATER USE ON RE-RUNS
data_master, data_fname = read_input_file()
data = data_master.copy()

#print(list(st.session_state.keys()))
if len(st.session_state.keys()) < 1:
#if "data" not in st.session_state:
    print("Resetting session")
    st.session_state = {
        "data": data,
        "row_filter_opts": [],
        "col_filter_opts": [],
        "update": False,
        "x_mean": 0.0,
        "x_std": 1.0,
        "x_lo": -1.0,
        "x_hi": 1.0,
        "x_min": -4.0,
        "x_max": 4.0,
        "y_mean": 0.0,
        "y_std": 1.0,
        "y_lo": -1.0,
        "y_hi": 1.0,
        "y_min": -4.0,
        "y_max": 4.0,
    }
# Lambdas for reference of mean and std of values to then use for graphing ranges
x_mean = lambda: st.session_state["x_mean"]
y_mean = lambda: st.session_state["y_mean"]
x_std = lambda: st.session_state["x_std"]
y_std = lambda: st.session_state["y_std"]

# Lambdas for getting the values for the i'th interval from the mean, e.g. 2 => 2 stds below and above the mean.
x_interval = lambda i: [float(x_mean()-i*x_std()), float(x_mean()+i*x_std())]
y_interval = lambda i: [float(y_mean()-i*y_std()), float(y_mean()+i*y_std())]

# Lambdas for reference of min max values - we additionally add 1 std to each to give them buffer area
x_min = lambda: st.session_state["x_min"]-x_std()
y_min = lambda: st.session_state["y_min"]-y_std()
x_max = lambda: st.session_state["x_max"]+x_std()
y_max = lambda: st.session_state["y_max"]+y_std()


# Lambdas for threshold values reference
x_lo = lambda: st.session_state["x_lo"]
x_hi = lambda: st.session_state["x_hi"]
y_lo = lambda: st.session_state["y_lo"]
y_hi = lambda: st.session_state["y_hi"]

sess = lambda s: st.session_state[s]
f"#### Current Data: "
data_container = st.empty()
#if not sess("update"):
data_container.container().write(st.session_state["data"])
#st.session_state["data"]

##### DATA PREPARATION AND MODIFICATION TIME #####

def update_data():
    """
    This way we ONLY UPDATE THE DATAFRAME AND RUN THOSE COSTLY FUNCTIONS ONCE WE'RE SURE

    Called after any filters for data modification are implemented, and applies them.
    :return:
    """
    print("Updating Data...")

    ####### FILTERING #######
    ##### ROW FILTERING #####

    ### PREFIX FILTERING ###
    # Keep only the rows with our prefixes selected
    # can't do a negative mask because there are prefixes which are not in our list.
    #
    # Get samples with correct prefix
    masks = [data_master["EpisodeNumber"].str.startswith(prefix) for prefix in sess("row_filter_opts")]
    # Combine all of them with logical OR into one
    if len(masks) != 0:
        mask = masks[0]
        if len(masks) > 1:
            for m in masks[1:]:
                mask = mask | m
        data = data_master[mask]
    else:
        # Empty dataframe
        data = data_master[0:0]
        return

    ##### COLUMN FILTERING #####
    # Don't remove columns until the very end, so we can still do our row filtering based on column values
    data = data[sess("col_filter_opts")]

    ####### TRANSFORMATIONS ########
    # Now that all data is filtered, we operate on what's left with various data transformations

    ### GET STATISTICS FOR NUMERICAL COLUMNS ###
    # Remove all non-numerical so we can operate on the numbers and put them back later
    #print(data.isnull().any())
    # Additionally get the columns so we can retain their order
    cols = data.columns.tolist()
    cols_non_numerical = data.select_dtypes(exclude=[np.number]).columns.values
    cols_numerical = data.select_dtypes(include=[np.number]).columns.values

    data_non_numerical = data[cols_non_numerical]
    data = data[cols_numerical]

    mean = data.mean()
    std = data.std()
    median = data.median()
    min = data.min()
    max = data.max()
    range = max-min

    ##### MISSING DATA HANDLING #####
    match st.session_state["missing_data_opt"]:
        case "Exclude Rows with N/A values (default)":
            data = data.dropna()

        case "Use Linear Regression to Interpolate Missing Values":
            data = data.interpolate()

        case "Replace with Mean Value":
            data = data.fillna(mean)

        case "Replace with Median Value":
            data = data.fillna(median)

    # print(data[data["C24:0"].isnull()])
    # print(data[data["C24:0"].isna()])

    ##### OUTLIER HANDLING #####
    if st.session_state["outlier_opt"]:
        # We don't apply anything here yet,
        # this is just for us getting the method to use later when making the graphing setup
        pass

    ##### SCALING / NORMALIZATION #####
    match st.session_state["scaling_opt"]:
        case "None (default)":
            # what did you expect?
            pass

        case "Z-Score Normalization":
            # Convert all values to their z-scores, i.e. normal distribution representatives.
            data = (data-mean) / std

        case "Min-Max Scaling":
            # Ensure all are in the range 0-1, with min value being 0 and max being 1.
            # We enforce a distribution into that range.
            # Similar to z score but without distribution stuff
            data = (data - min) / range

        case "Multiple-of-Mean Standardization":
            # Represent all data as multiples of the mean value.
            # Mean value is 1, and for any value, multiplying it by the mean will obtain the original.
            data = data / mean

        case "Multiple-of-Median Standardization":
            # Same as above
            data = data / median

        case "Multiple-of-Standard-Deviation Standardization":
            # Same as above
            data = data / std

        case "Percentile of Max Standardization":
            # Convert all values to their percentile, such that for a given percentile p,
            # p % of the values in the list are below the value of that percentile.
            # 50% percentile would be the median of the list (sorted).
            # We keep them in the range 0-1 rather than 0-100, however.
            for col in data:
                data[col] = stats.rankdata(data[col], "average")/len(data[col])

    ##### DATA TRANSFORMATIONS #####
    match st.session_state["transformation_opt"]:
        case "None (default)":
            # lookin' sus
            pass

        case "Log Scaling (base 10)":
            for col in data:
                data[col] = np.log10(data[col])

        case "Square Root":
            for col in data:
                data[col] = np.sqrt(data[col])

        case "Yeo-Johnson":
            for col in data:
                data[col] = power_transform(data[col].values.reshape(-1,1), method="yeo-johnson")

        case "Box-Cox (Only applied to columns with all positive values, otherwise columns will be skipped)":
            # Can only be applied to positive values, if data is 0 then add epsilon to it.
            for col in data:
                if (data[col] >= 0).all():
                    if (data[col] == 0).any():
                        # Add epsilon to let power transform happen and avoid error
                        data[col] += 1e-8
                    data[col] = power_transform(data[col].values.reshape(-1,1), method="box-cox")

        case "Inverse Fourier":
            # Have to cast this to float64 in order to get rid of complex part of the imaginary num.
            for col in data:
                data[col] = np.fft.ifft(data[col].values).astype(np.float64)


    # PUT BACK NON NUMERICAL COLUMNS
    # Number ops are done, put them back with updated values.
    data[cols_non_numerical] = data_non_numerical
    # Preserve original ordering of columns
    data = data[cols]

    # Update displayed dataframe, always do this last
    # TODO: Do we need to store data in the session state if we are doing container write?
    st.session_state["data"] = data
    data_container.empty()
    data_container.write(st.session_state["data"])

#with st.form("data_prep"):
col1,col2 = st.columns(2)
with col1:
    #st.markdown("<u><b>Columns Filtering<b></u>", unsafe_allow_html=True)
    #st.write("Columns Filtering")
    ##### HEADERS #####
    # Choice of all headers to include in resulting dataset.
    container = st.container()  # lets us have these out of order displayed but affect each other
    no_ratios = st.checkbox("Exclude Ratio (A/B) Columns")
    no_nans = st.checkbox("Exclude Columns with N/A Values")
    numerical_only = st.checkbox("Only Include Columns with Numerical Values")

    headers = [str(col) for col in data.columns]
    if no_ratios:
        # remove all ratio headers currently selected
        headers = [h for h in headers if "/" not in h]

    if no_nans:
        headers = [h for h in headers if not data[h].isnull().values.any()]

    if numerical_only:
        headers = [h for h in headers if np.issubdtype(data[h].dtype, np.number)]

    st.session_state["col_filter_opts"] = container.multiselect("Choose which Column Headers to Keep:", headers, headers)

    # Let them choose the types of samples to use, via the prefixes on
    # the "EpisodeNumber" column. F is only one checked by default,
    # however we may also have AAC and S. These are the supported options
    prefixes = ["F", "AAC", "S", "R"]
    st.session_state["row_filter_opts"] = st.multiselect("Choose which Sample Types to use for this Analysis (by prefix):", prefixes, ["F"])

with col2:
    st.session_state["missing_data_opt"] = st.selectbox(
        "Missing Data Handling",
        ("Exclude Rows with N/A values (default)",
         "Use Linear Regression to Interpolate Missing Values",
         "Replace with Mean Value",
         "Replace with Median Value",
         ))
    st.session_state["outlier_opt"] = st.selectbox(
        "Outlier Handling (Choose only the Method, Parameters are chosen in the 'Data Visualization' section)",
        ("Value Thresholds (default)",
         "Percentage Thresholds",
         "Quantiles",
         "Multiple of Standard-Deviation",
         "Do Nothing",
         ), disabled=True)
    st.session_state["scaling_opt"] = st.selectbox(
        "Scaling / Normalization Method",
        ("None (default)",
         "Z-Score Normalization",
         "Min-Max Scaling",
         "Multiple-of-Mean Standardization",
         "Multiple-of-Median Standardization",
         "Multiple-of-Standard-Deviation Standardization",
         "Percentile of Max Standardization",
         ))
    st.session_state["transformation_opt"] = st.selectbox(
        "Data Transformations (Make sure your data is in the correct ranges for your chosen transform)",
        ("None (default)",
         "Log Scaling (base 10)",
         "Yeo-Johnson",
         "Box-Cox (Only applied to columns with all positive values, otherwise columns will be skipped)",
         ))

    # TODO: Add reset button?
    # submitted = st.form_submit_button("UPDATE DATA")
    st.button("UPDATE DATA", on_click=update_data)

"# Data Visualization & Analysis"
"Using the above dataframe, you can now visualize and analyse the resulting data however you like."
"Lets just start with one var each."

data = st.session_state["data"]


#x = data["C24:0"]
#y = data["C26:0"]
@st.cache
def get_x():
    return np.random.randn(10000)

@st.cache
def get_y():
    return np.random.randn(10000)

_1, _2 = st.columns(2)

gcol1, gcol2 = st.columns((1,3), gap="large")

# BRYCE

# UNCOMMENT, THEN COMMENT AGAIN

#st.session_state["x_mean"] = 0.0
#st.session_state["x_std"] = 0.0
#st.session_state["x_lo"] = 0.0
#st.session_state["x_hi"] = 0.0
#st.session_state["x_min"] = 0.0
#st.session_state["x_max"] = 0.0

#st.session_state["y_mean"] = 0.0
#st.session_state["y_std"] = 0.0
##st.session_state["y_lo"] = 0.0
#st.session_state["y_hi"] = 0.0
#st.session_state["y_min"] = 0.0
#st.session_state["y_max"] = 0.0
#st.session_state["y_mean"] = 0.0
#st.session_state["y_mean"] = 0.0
#st.session_state["x_mean"] = 0.0
#st.session_state["x_mean"] = 0.0
#st.session_state["x_mean"] = 0.0




#@st.cache(allow_output_mutation=True)
def get_fig(x, y):
    #x = get_x()
    #y = get_y()
    fig = go.Figure(
        data=[
            go.Histogram2d(
                x=x,
                y=y,
                nbinsx=400,
                nbinsy=400,
                colorscale="Blues",
                colorbar=dict(tickfont=dict(size=30)),
            )
        ],
        layout_height=800,
    )

    # Set these up to be changing relative to the session state values, which change as the slider changes.
    # TODO: can make shift into x and y and make it relative to the size of the vrect and hrect if we want
    shift = 0
    fontsize = 30
    fig.update_xaxes(showgrid=False, tickfont=dict(size=fontsize))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=fontsize))

    # Init these
    st.session_state["x_min"] = np.amin(x)
    st.session_state["y_min"] = np.amin(y)
    st.session_state["x_max"] = np.amax(x)
    st.session_state["y_max"] = np.amax(y)

    st.session_state["x_mean"] = np.mean(x)
    st.session_state["x_std"] = np.std(x)
    st.session_state["y_mean"] = np.mean(y)
    st.session_state["y_std"] = np.std(y)

    #print(x_min(),x_max())
    #st.session_state["x_thresh"] = x_interval(2)
    #st.session_state["y_thresh"] = y_interval(2)

    # Threshold lines - we use these to draw the main quadrant thresholds
    fig.add_vline(x=x_lo(), annotation_text=f"{round(x_lo(),4)}", annotation_font=dict(size=fontsize), annotation_xshift=-shift, annotation_position="left")
    fig.add_vline(x=x_hi(), annotation_text=f"{round(x_hi(),4)}", annotation_font=dict(size=fontsize), annotation_xshift=shift, annotation_position="right")
    fig.add_hline(y=y_lo(), annotation_text=f"{round(y_lo(),4)}", annotation_font=dict(size=fontsize), annotation_yshift=-shift, annotation_position="bottom")
    fig.add_hline(y=y_hi(), annotation_text=f"{round(y_hi(),4)}", annotation_font=dict(size=fontsize), annotation_yshift=shift, annotation_position="top")

    # Threshold areas (rectangles)
    fig.add_vrect(x0=x_min(), x1=x_lo(), fillcolor="blue", opacity=.2)
    fig.add_vrect(x0=x_hi(), x1=x_max(), fillcolor="blue", opacity=.2)
    fig.add_hrect(y0=y_min(), y1=y_lo(), fillcolor="red", opacity=.2)
    fig.add_hrect(y0=y_hi(), y1=y_max(), fillcolor="red", opacity=.2)

    # Compute center of all octants so we can put annotations directly in the middle of them.
    # Several of these are simplified to avoid repeated calls to y_hi() x_lo() and so on.
    # e.g. (y_max-y_hi())/4 + y_hi() => y_max/4 - y_hi()/4 + y_hi() => y_max/4 + 3*y_hi()/4
    # e.g. (x_max - x_hi())/2 + x_hi() => x_max/2 - x_hi()/2 + x_hi() => x_max/2 + x_hi()/2
    left = x_lo()/2 + x_min()/2
    top = y_max()/4 + 3*y_hi()/4 # smaller because of height of text, hence /4
    right = x_max()/2 + x_hi()/2
    bot = y_lo()/4 + 3*y_min()/4
    mid_x = x_hi()/2 + x_lo()/2
    mid_y = y_hi()/2 + y_lo()/2

    # Compute values for all octants for the text to go in each - # of values inside and % of values inside.
    # We create masks for each to avoid having to repeat our calculations for the corners, where we AND them together.
    mask_x_hi = x > x_hi()
    mask_x_lo = x < x_lo()
    mask_y_hi = y > y_hi()
    mask_y_lo = y < y_lo()

    # Middle one is actually the most complicated in terms of logic
    mask_mid = np.logical_and(np.logical_and(x >= x_lo(), x <= x_hi()), np.logical_and(y >= y_lo(), y <= y_hi()))

    # len
    n = len(x) # EXPECTS X AND Y TO BE SAME LENGTH

    mask_n = lambda m: np.sum(m) # compute # in masked area
    mask_and_n = lambda m1, m2: np.sum(np.logical_and(m1,m2)) # compute # in intersecting masked area

    mask_p = lambda m: round(mask_n(m)/n*100., 4) # compute % of whole in masked area
    mask_and_p = lambda m1, m2: round(mask_and_n(m1, m2)/n*100., 4) # compute % of whole in intersecting masked area

    # Combine both into larger easier to reference one
    mask_stats = lambda m: (mask_n(m), mask_p(m))
    mask_and_stats = lambda m1,m2: (mask_and_n(m1,m2), mask_and_p(m1,m2))

    # Given this, now we can assemble the 9 values
    grid = [
        [mask_and_stats(mask_x_lo, mask_y_hi), mask_stats(mask_y_hi), mask_and_stats(mask_x_hi, mask_y_hi)],
        [mask_stats(mask_x_lo), mask_stats(mask_mid), mask_stats(mask_x_hi)],
        [mask_and_stats(mask_x_lo, mask_y_lo), mask_stats(mask_y_lo), mask_and_stats(mask_x_hi, mask_y_lo)],
    ]

    # Format string lambda
    grid_str = lambda i,j: f"{grid[i][j][0]}, {grid[i][j][1]}%"

    # Finally put all the values in
    fig.add_annotation(x=left, y=top, text=grid_str(0,0), font=dict(size=fontsize))
    fig.add_annotation(x=mid_x, y=top, text=grid_str(0,1), font=dict(size=fontsize))
    fig.add_annotation(x=right, y=top, text=grid_str(0,2), font=dict(size=fontsize))

    fig.add_annotation(x=left, y=mid_y, text=grid_str(1,0), font=dict(size=fontsize))
    fig.add_annotation(x=mid_x, y=mid_y, text=grid_str(1,1), font=dict(size=fontsize))
    fig.add_annotation(x=right, y=mid_y, text=grid_str(1,2), font=dict(size=fontsize))

    fig.add_annotation(x=left, y=bot, text=grid_str(2,0), font=dict(size=fontsize))
    fig.add_annotation(x=mid_x, y=bot, text=grid_str(2,1), font=dict(size=fontsize))
    fig.add_annotation(x=right, y=bot, text=grid_str(2,2), font=dict(size=fontsize))

    return fig

with _1:
    inputs = st.multiselect("Input / Independent Variables: ", data.columns)
    outputs = st.multiselect("Output / Dependent Variables: ", [col for col in data.columns if col not in inputs])
with _2:
    st.markdown("#")
    #st.button("GENERATE GRAPH", on_click=generate_graph, args=(inputs[0], outputs[0]))

#def generate_graph(x_label,y_label):
#x = data[inputs[0]]
#y = data[outputs[0]]

# If the user hasn't specified values for this, then don't show anything yet.
if len(inputs) == 0 or len(outputs) == 0:
    # Not enough values, specify text for this.
    st.markdown("### Not enough values specified; select an input and and output variable to generate a graph.")

else:

    with gcol1:
        # Settings for x threshold
        st.session_state["x_lo"], st.session_state["x_hi"] = st.slider(
            'X threshold',
            x_min(), x_max(), x_interval(2), 0.01, format="%0.4f")

        # Can also be controlled with more granularity
        st.session_state["x_lo"] = st.number_input(
            'X Lower Threshold',
            x_min(), x_hi(), x_lo(), 0.0001, format="%0.4f")
        st.session_state["x_hi"] = st.number_input(
            'X Higher Threshold',
            x_lo(), x_max(), x_hi(), 0.0001, format="%0.4f")

        # Spacer
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")

        # Settings for y threshold
        st.session_state["y_lo"], st.session_state["y_hi"] = st.slider(
            'Y threshold',
            y_min(), y_max(), y_interval(2), 0.01, format="%0.4f")

        st.session_state["y_lo"] = st.number_input(
            'Y Lower Threshold',
            y_min(), y_hi(), y_lo(), 0.0001, format="%0.4f")
        st.session_state["y_hi"] = st.number_input(
            'Y Higher Threshold',
            y_lo(), y_max(), y_hi(), 0.0001, format="%0.4f")

        # on change, change the lines.
        #st.write('Values:', values) # and then we can add on the % etc.

    with gcol2:
        graph_container = st.container()
        x = data[inputs[0]]
        y = data[outputs[0]]
        fig = get_fig(x, y)
        #fig.update_layout(autosize=False, height=800)
        # Set up some reasonable margins and heights so we actually get a more square-like graph
        # rather than the wide boi streamlit wants it to be
        fig.layout.height=1000
        fig.layout.margin=dict(l=100, r=100, t=0, b=0)
        graph_container.plotly_chart(fig, use_container_width=True)

st.session_state

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





