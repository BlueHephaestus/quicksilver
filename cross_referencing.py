import streamlit as st
import numpy as np
import pandas as pd
import sys

GRAPHS_1D = ["Pie Chart", "Histogram of Bins"]
GRAPHS_2D = ["Line Graph", "Scatter Plot", "Table", "2D Heatmap of Bins"]

def is_valid_inputs(inputs, outputs, graph_type):
    """
    Determines if the provided parameters can generate the graph type provided. True/False.
    """
    n_i = len(inputs)
    n_o = len(outputs)
    if graph_type in GRAPHS_2D:
        # Actual cross reference, requires one of each var at least
        return n_i >= 1 and n_o >= 1
    elif graph_type in GRAPHS_1D:
        # Only requires one of either var
        return n_i + n_o >= 1


def cross_reference(df, inputs, outputs, graph_type):
    # For each cross_reference on the dataframe, the user will provide:
    #   1. Inputs / Independent Variables
    #   2. Outputs / Dependent Variables
    #   3. Type of graph to use.
    #
    # Each call will produce one or more graphs based on the options selected.
    #
    # If the user provides more than one input variable, this will produce an equivalent number
    # of graphs, with each input variable getting its own graph. 
    #   Note: each input variable's graph will be sorted by the input variable.
    #
    # Conversely, each output variable will be graphed on all graphs, with a legend provided to 
    # differentiate between each one. 
    #
    # For example, if we have five attributes: Age, # Siblings, Race, Ethnicity, Eye Color,
    # and we want age and eye color as our inputs, versus the other three, the only way that makes
    # sense is to have one graph which is age vs. # siblings, age vs. race, and age vs. ethnicity
    # and a corresponding other graph for eye color vs the remaining three. Otherwise we'd have to
    # deal with all combinations of both input variables, which is a can of worms I don't want to get 
    # into right now. More info on that at the bottom of this description.
    #
    #
    # Type of graph is applied to all generated graphs, and the options are:
    #   1. Line Graph (Default, simple)
    #   2. Scatter Plot (same as line, just dots instead)
    #   3. 1D Histogram of Bins
    #       These are unique in that they can only graph one variable at a time as an input,
    #       with the output being the number of samples inside of a discrete range of input 
    #       variable values. 
    #       As a result, we make one graph for each input and output variable, and things are not
    #       strictly cross-referenced.
    #       Users will have a slider for # of bins.
    #   4. Pie Chart
    #       These are the same as 1D histogram of bins, except they require each "bin" to 
    #       be a unique value. They also can only graph one variable, and we will skip a variable
    #       if there are more than 100 unique values for it, but otherwise graph each input
    #       and output individually.
    #   5. 2D Histogram of Bins - I might not add this it depends
    #       These allow the benefit of bins and histograms, but also allow cross-reference,
    #       via one input and one output variable per graph. This means that it can produce the most
    #       graphs, with # input * # output graphs produced. 
    #
    #       Each axis has bin control (if I can get that working), and produces a grid-like graph,
    #       with each square/rect in the grid being a "heat" color, with higher heat meaning
    #       higher number of samples in that range of input vars and output vars.
    #   6. Table - Same as 2d Histogram of Bins, but with unique values. 
    #       Basically a 2d pie chart.
    #       
    #
    # TODO / POSSIBLE FUTURE ADDITION
    #
    # For some variables with a low amount of unique values in the population, such as
    # Eye Color and Hair Color, it would make sense to support combining both variables into
    # "Eye Color & Hair Color", which would create a new variable, with 
    #       (# of unique eye colors) * (# of unique hair colors) 
    # unique values of its own. 
    #
    # This could be combined with an arbitrary number of variables, so long as the amount of
    # resultant combinations was not too excessive (like, over 100 maybe).
    #
    # But given the complications of this feature i'm not going to add it at the start.
    
    # is_valid_inputs already ensures we will have enough to do whatever graph_type we have.
    import plotly.express as px
    import plotly.graph_objects as go
    if graph_type in GRAPHS_1D:
        # Do a lot of common work before graph-specific stuff
        # Since 1D, we can ignore input/output and just put them in one list.
        inputs = inputs + outputs
        
        # Generate a fig for each
        # TODO make cutoff number to just include the rest in "Other" for piechart?
        for i in inputs:
            if graph_type == "Pie Chart":
                names,values = np.unique(df[i], return_counts=True)
                if len(names) <= 100:
                    fig = px.pie(df, values=values, names=names, title=i)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Unable to create graph for {}, too many unique values.".format(i))

            elif graph_type == "Histogram of Bins":
                container = st.container()
                fig = px.histogram(df, x=i, title=i, nbins=st.slider("Number of Bins", key=i))
                container.plotly_chart(fig, use_container_width=True)

    elif graph_type in GRAPHS_2D:
        for i in inputs: # 1 fig per input, x is always sample index
            for o in outputs: # 1 fig per input, x is always sample index
                # TODO add 3rd variable option here???
                if graph_type == "Line Graph" or graph_type == "Scatter Plot":
                    mode = "lines" if graph_type == "Line Graph" else "markers"
                    container = st.container()
                    tmp = df.sort_values(by=i)
                    fig = go.Figure(data=(go.Scatter(x=tmp[i], y=tmp[o], mode=mode)))
                    container.plotly_chart(fig, use_container_width=True)

                if graph_type == "Table":
                    # Number of samples is the value in each cell.
                    # Checks for not too many uniques, like with pie chart
                    rows, row_totals = np.unique(df[i], return_counts=True)
                    cols, col_totals = np.unique(df[o], return_counts=True)

                    values = np.zeros((len(rows)+1, len(cols)+1), dtype=int)

                    for r,row in enumerate(rows):
                        for c,col in enumerate(cols):
                            values[r][c] = len(df[(df[i]==row) & (df[o]==col)])
                    for r,total in enumerate(row_totals):
                        values[r][-1]=total
                    for c,total in enumerate(col_totals):
                        values[-1][c]=total
                    values[-1,-1] = sum(row_totals) # same as col_totals

                    # Add row headings
                    row_headers = list(rows)+["Total"]
                    row_headers = ["<b>{}</b>".format(h) for h in row_headers]
                    row_headers = np.array(row_headers).reshape(-1,1)
                    values = np.hstack((row_headers, values))

                    # Col headings
                    col_headers = [""]+list(cols)+["Total"]
                    col_headers = ["<b>{}</b>".format(h) for h in col_headers]

                    # Plotly does this the reverse way so we transpose to match them
                    # REMEMBER TO DO THIS LAST
                    values = np.transpose(values)
                    container = st.container()
                    fig = go.Figure(data=[go.Table(
                        header = dict(values=col_headers),
                        cells=dict(values=values),
                        )])
                    fig.update_layout(title_text="{} vs. {}".format(i,o))
                    container.plotly_chart(fig, use_container_width=True)

                if graph_type == "2D Heatmap of Bins":
                    # last one!!!111!!
                    # This is basically whwat the table should have been... maybe we redo later.
                    container = st.container()
                    fig = px.density_heatmap(df, x=i, y=o, nbinsx=st.slider("Number of X Bins", key=i+"_"+o+"_x"), nbinsy=st.slider("Number of Y Bins",key=i+"_"+o+"_y"), marginal_x="histogram",marginal_y="histogram")
                    container.plotly_chart(fig, use_container_width=True)


        

