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
from Session import *
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

if len(st.session_state.keys()) < 4:
#if "data" not in st.session_state:
    print("Resetting session")
    st.session_state = {
        "data": data,
        "row_filter_opts": [],
        "col_filter_opts": [],
        "update": False,
    }

sess = lambda s: st.session_state[s]
f"#### Current Data: "
data_container = st.empty()
data_container.container().write(st.session_state["data"])

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
    # Convert all numerical columns to matching datatype to avoid problems in later use.
    data = data.astype(np.float64)

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

data = st.session_state["data"]
session = Session(data)

gfcol1, gfcol2 = st.columns(2)

gcol1x, gcol1y, gcol2, gcol3 = st.columns((1,1,4,2), gap="large")



#@st.cache(allow_output_mutation=True)
def get_threshold_graph():
    #x = get_x()
    #y = get_y()
    accessions = data_master["EpisodeNumber"]
    if session.scatter_enable:
        # Scatterplot
        hovertemplate="<b>%{customdata}</b><br>" + session.x.col + ": %{x}<br>" + session.y.col + ": %{y}<br><extra></extra>"
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=session.x.data,
                    y=session.y.data,
                    mode="markers",
                    customdata=accessions,
                    hovertemplate=hovertemplate
                )
            ],
            layout_height=800,
        )
    else:
        # Histogram
        # oi future self not sure how or if we can have accessions on the histogram2d points
        hovertemplate="<b>Population: %{z}</b><br>" + session.x.col + ": %{x}<br>" + session.y.col + ": %{y}<br><extra></extra>"
        fig = go.Figure(
            data=[
                go.Histogram2d(
                    x=session.x.data,
                    y=session.y.data,
                    nbinsx=400,
                    nbinsy=400,
                    customdata=accessions,
                    hovertemplate=hovertemplate,
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
    fig.update_xaxes(showgrid=False, title=session.x.col, titlefont=dict(size=fontsize), tickfont=dict(size=fontsize))
    fig.update_yaxes(showgrid=False, title=session.y.col, titlefont=dict(size=fontsize), tickfont=dict(size=fontsize))

    #print(session.x.min,session.x.max)
    #st.session_state["x_thresh"] = x_interval(2)
    #st.session_state["y_thresh"] = y_interval(2)

    # Threshold lines - we use these to draw the main quadrant thresholds
    fig.add_vline(x=session.x.lo, annotation_text=f"{round(session.x.lo,4)}", annotation_font=dict(size=fontsize), annotation_xshift=-shift, annotation_position="left")
    fig.add_vline(x=session.x.hi, annotation_text=f"{round(session.x.hi,4)}", annotation_font=dict(size=fontsize), annotation_xshift=shift, annotation_position="right")
    fig.add_hline(y=session.y.lo, annotation_text=f"{round(session.y.lo,4)}", annotation_font=dict(size=fontsize), annotation_yshift=-shift, annotation_position="bottom")
    fig.add_hline(y=session.y.hi, annotation_text=f"{round(session.y.hi,4)}", annotation_font=dict(size=fontsize), annotation_yshift=shift, annotation_position="top")

    # Threshold areas (rectangles)
    # for some reason need an extra buffer for the minimum x one.
    fig.add_vrect(x0=session.x.min-session.x.std, x1=session.x.lo, fillcolor="blue", opacity=.2)
    fig.add_vrect(x0=session.x.hi, x1=session.x.max, fillcolor="blue", opacity=.2)
    fig.add_hrect(y0=session.y.min-session.y.std, y1=session.y.lo, fillcolor="red", opacity=.2)
    fig.add_hrect(y0=session.y.hi, y1=session.y.max, fillcolor="red", opacity=.2)


    return fig

def get_threshold_tables():
    # Get 3x3 tables for the areas in each of our sections in our threshold graph.

    # Compute values for all octants for the text to go in each - # of values inside and % of values inside.
    # We create masks for each to avoid having to repeat our calculations for the corners, where we AND them together.
    mask_x_hi = session.x.data > session.x.hi
    mask_x_lo = session.x.data < session.x.lo
    mask_y_hi = session.y.data > session.y.hi
    mask_y_lo = session.y.data < session.y.lo

    # Middle one is actually the most complicated in terms of logic
    mask_mid = np.logical_and(np.logical_and(session.x.data >= session.x.lo, session.x.data <= session.x.hi), np.logical_and(session.y.data >= session.y.lo, session.y.data <= session.y.hi))

    # len
    n = len(session.x.data) # EXPECTS X AND Y TO BE SAME LENGTH

    mask_n = lambda m: np.sum(m) # compute # in masked area
    mask_and_n = lambda m1, m2: np.sum(np.logical_and(m1,m2)) # compute # in intersecting masked area

    mask_p = lambda m: round(mask_n(m)/n*100., 4) # compute % of whole in masked area
    mask_and_p = lambda m1, m2: round(mask_and_n(m1, m2)/n*100., 4) # compute % of whole in intersecting masked area

    # Given this, now we can assemble the 9 values
    grid_ns = [
        [mask_and_n(mask_x_lo, mask_y_hi), mask_n(mask_y_hi), mask_and_n(mask_x_hi, mask_y_hi)],
        [mask_n(mask_x_lo), mask_n(mask_mid), mask_n(mask_x_hi)],
        [mask_and_n(mask_x_lo, mask_y_lo), mask_n(mask_y_lo), mask_and_n(mask_x_hi, mask_y_lo)],
    ]
    grid_ps = [
        [mask_and_p(mask_x_lo, mask_y_hi), mask_p(mask_y_hi), mask_and_p(mask_x_hi, mask_y_hi)],
        [mask_p(mask_x_lo), mask_p(mask_mid), mask_p(mask_x_hi)],
        [mask_and_p(mask_x_lo, mask_y_lo), mask_p(mask_y_lo), mask_and_p(mask_x_hi, mask_y_lo)],
    ]

    for i in range(3):
        for j in range(3):
            grid_ns[i][j] = " <br>" + str(grid_ns[i][j])
            grid_ps[i][j] = " <br>" + str(grid_ps[i][j])

    red = "#f9c9cc"
    blue = "#c6c9ff"
    purple = "#d1a1cc"
    # transposed because plotly transposes the table
    colors = [
        [purple, blue, purple],
        [red, "white", red],
        [purple, blue, purple]
    ]

    # Plotly does this the reverse way so we transpose to match them
    # REMEMBER TO DO THIS LAST
    grid_ns = np.transpose(grid_ns)
    grid_ps = np.transpose(grid_ps)
    fig_ns = go.Figure(data=[go.Table(
        header=None,
        cells=dict(values=grid_ns, height=130),
    )])
    fig_ps = go.Figure(data=[go.Table(
        header=None,
        cells=dict(values=grid_ps, height=130, suffix="%"),
    )])
    # Remove headers via making invisible
    fig_ns.for_each_trace(lambda t: t.update(header_fill_color='rgba(0,0,0,0)'))
    fig_ps.for_each_trace(lambda t: t.update(header_fill_color='rgba(0,0,0,0)'))
    fig_ns.update_traces(cells_fill_color=colors)
    fig_ps.update_traces(cells_fill_color=colors)

    fig_ns.update_layout(font_size=32, margin=dict(t=10, b=10))
    fig_ps.update_layout(font_size=32, margin=dict(t=10, b=10))
    return fig_ns, fig_ps

#print([data[col].dtype for col in data.columns])
numeric_cols = [col for col in data.columns if np.issubdtype(data[col].dtype, np.number)]
with gfcol1:
    # Only allow choice of numeric columns, for now.
    session.x.col = st.selectbox("Input / Independent Variables: ", numeric_cols)
    session.y.col = st.selectbox("Output / Dependent Variables: ", [col for col in numeric_cols if col != session.x.col])
    session.scatter_enable = st.checkbox("Render graph as scatterplot instead of histogram (this will lower performance)")
with gfcol2:
    st.markdown("#")
    #st.button("GENERATE GRAPH", on_click=lambda x:x, args=(input, output))

#def generate_graph(x_label,y_label):
#x = data[inputs[0]]
#y = data[outputs[0]]

# TODO set this up so that they have to press update to update the graph?
# If the user hasn't specified values for this, then don't show anything yet.

# Update these attributes when we have columns for them
#session.x.col = input
#session.y.col = output
session.x.update(session.data)
session.y.update(session.data)
#session.x.print()

xname = session.x.col
yname = session.y.col
mask_n = lambda m: np.sum(m)/len(m)*100  # compute % in masked area
#lambda perc2num()
perc2num = lambda data, p: np.percentile(data, p)
num2perc = lambda data, n: np.sum(data < n)/len(data)*100


# PROBLEM: TODO: Updates to values in later widgets don't update earlier ones. if i change the number it doesn't move slider.
# Unfortunately this is a limitation of streamlit, and can't be fixed yet. Fortuantely, it still changes the graph.
with gcol1x:
    # Settings for x threshold
    session.x.lo, session.x.hi = st.slider(
        f'{xname} threshold',
        session.x.min, session.x.max, session.x.interval(2), 0.01, format="%0.4f")

    # Can also be controlled with more granularity
    session.x.lo = st.number_input(
        f'{xname} Lower Threshold',
        session.x.min, session.x.hi, session.x.lo, 0.0001, format="%0.4f")
    session.x.hi = st.number_input(
        f'{xname} Higher Threshold',
        session.x.lo, session.x.max, session.x.hi, 0.0001, format="%0.4f")

    # And via percentiles
    session.x.lo = perc2num(session.x.data, st.number_input(
        f'{xname} Lower Threshold (Percentile)',
        0., num2perc(session.x.data, session.x.hi), num2perc(session.x.data, session.x.lo), .1, format="%.2f"))#Streamlit does not allow % symbol here
    session.x.hi = perc2num(session.x.data, st.number_input(
        f'{xname} Higher Threshold (Percentile)',
        num2perc(session.x.data, session.x.lo), 100., num2perc(session.x.data, session.x.hi), .1, format="%.2f"))

with gcol1y:
    # Spacer
    # st.markdown("#")
    # st.markdown("#")
    # st.markdown("#")
    # st.markdown("#")

    # Settings for y threshold
    session.y.lo, session.y.hi = st.slider(
        f'{yname} threshold',
        session.y.min, session.y.max, session.y.interval(2), 0.01, format="%0.4f")

    session.y.lo = st.number_input(
        f'{yname} Lower Threshold',
        session.y.min, session.y.hi, session.y.lo, 0.0001, format="%0.4f")

    session.y.hi = st.number_input(
        f'{yname} Higher Threshold',
        session.y.lo, session.y.max, session.y.hi, 0.0001, format="%0.4f")

    # And via percentiles
    session.y.lo = perc2num(session.y.data, st.number_input(
        f'{yname} Lower Threshold (Percentile)',
        0., num2perc(session.y.data, session.y.hi), num2perc(session.y.data, session.y.lo), .1, format="%.2f"))#Streamlit does not allow % symbol here
    session.y.hi = perc2num(session.y.data, st.number_input(
        f'{yname} Higher Threshold (Percentile)',
        num2perc(session.y.data, session.y.lo), 100., num2perc(session.y.data, session.y.hi), .1, format="%.2f"))

    # on change, change the lines.
    #st.write('Values:', values) # and then we can add on the % etc.

with gcol2:
    # TODO remove width stuff?
    graph_container = st.container()
    #x = data[inputs[0]]
    #y = data[outputs[0]]
    fig = get_threshold_graph()
    #fig.update_layout(autosize=False, height=800)
    # Set up some reasonable margins and heights so we actually get a more square-like graph
    # rather than the wide boi streamlit wants it to be
    fig.layout.height=1000
    fig.layout.margin=dict(l=100, r=100, t=0, b=0)
    graph_container.plotly_chart(fig, use_container_width=True)

with gcol3:
    table_container = st.container()
    table_ns, table_ps = get_threshold_tables()
    table_container.markdown("### Threshold Areas")
    table_container.plotly_chart(table_ns, use_container_width=True)
    table_container.markdown("### Threshold Area Percentages")
    table_container.plotly_chart(table_ps, use_container_width=True)


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





