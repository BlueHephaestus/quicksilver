import plotly.graph_objects as go
import streamlit as st
from Constants import *
import numpy as np
import pandas as pd
#@st.cache(allow_output_mutation=True)
def get_threshold_graph(session, data_master):
    #x = get_x()
    #y = get_y()
    #accessions = data_master["EpisodeNumber"]
    accessions = data_master[session.accession_col]
    if session.scatter_enable:
        # Scatterplot
        hovertemplate="<b>" + session.accession_col + ": %{customdata}</b><br>" + session.x.col + ": %{x}<br>" + session.y.col + ": %{y}<br><extra></extra>"

        # Get color codes for each point based on categorical column.
        color_coding = False
        if session.categorical_col != "None":
            #divisions only, remember.
            # color = index of the value in the categories array ( we already know it's in there, and will have value.)
            # f, m, u -> [colormap[0], colormap[1], colormap[2]]

            # we already know it's filtered
            # we also know we won't be graphing any of the categorical col
            # for category in session.data[session.categorical_col]:
            #     COLORMAP[categories.index cat])

            # Create zeroed colors array, then update the ones that match each category to their respective color.
            color_coding = True
            colors = np.zeros((len(session.x.data)), dtype=np.object)
            for i, category in enumerate(session.categories):
                #if session.categorical_divisions[i]:
                colors[session.data[session.categorical_col] == category] = COLORMAP[i]

        print(session.categorical_divisions)
        print(len(session.data))

        if color_coding:
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=session.x.data[session.data[session.categorical_col] == category],
                        y=session.y.data[session.data[session.categorical_col] == category],
                        mode="markers",
                        marker=dict(color=COLORMAP[i]),
                        name=f"{session.categorical_col} = {category}",
                        customdata=accessions,
                        hovertemplate=hovertemplate
                    ) for i, category in enumerate(session.categories)
                ],
                layout_height=800,
            )
            fig.update_layout(font=dict(size=20))
        else:
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
    fig.update_xaxes(showgrid=False, title=session.x.col, titlefont=dict(size=FONTSIZE), tickfont=dict(size=FONTSIZE))
    fig.update_yaxes(showgrid=False, title=session.y.col, titlefont=dict(size=FONTSIZE), tickfont=dict(size=FONTSIZE))

    # Threshold lines - we use these to draw the main quadrant thresholds
    fig.add_vline(x=session.x.lo, annotation_text=f"{round(session.x.lo,2)}", annotation_font=dict(size=FONTSIZE), annotation_xshift=-shift, annotation_position="left")
    fig.add_vline(x=session.x.hi, annotation_text=f"{round(session.x.hi,2)}", annotation_font=dict(size=FONTSIZE), annotation_xshift=shift, annotation_position="right")
    fig.add_hline(y=session.y.lo, annotation_text=f"{round(session.y.lo,2)}", annotation_font=dict(size=FONTSIZE), annotation_yshift=-shift, annotation_position="bottom")
    fig.add_hline(y=session.y.hi, annotation_text=f"{round(session.y.hi,2)}", annotation_font=dict(size=FONTSIZE), annotation_yshift=shift, annotation_position="top")

    # Threshold areas (rectangles)
    # for some reason need an extra buffer for the minimum x one.
    fig.add_vrect(x0=session.x.min-session.x.std, x1=session.x.lo, fillcolor="blue", opacity=.2)
    fig.add_vrect(x0=session.x.hi, x1=session.x.max+session.x.std, fillcolor="blue", opacity=.2)
    fig.add_hrect(y0=session.y.min-session.y.std, y1=session.y.lo, fillcolor="red", opacity=.2)
    fig.add_hrect(y0=session.y.hi, y1=session.y.max+session.y.std, fillcolor="red", opacity=.2)


    return fig

def get_singlevar_histogram(session):

    default_bins = int(np.ceil(np.sqrt(len(session.x.data))))
    fig = go.Figure(
        data=[
            go.Histogram(
                x=session.x.data,
                nbinsx=st.slider("Number of Bins", 1, 4*default_bins, default_bins),
            )
        ],
        layout_height=800,
    )
    #print(fig.data[0].nbinsx)
    fig.update_layout(
        title_text = f"<b>'{session.x.col}'</b> Analysis on {len(session.data)} Samples",
        title_font=dict(size=FONTSIZE),
    )
    fig.update_xaxes(showgrid=False, title=session.x.col, titlefont=dict(size=FONTSIZE), tickfont=dict(size=FONTSIZE))
    fig.update_yaxes(showgrid=False, title="Population", titlefont=dict(size=FONTSIZE), tickfont=dict(size=FONTSIZE))
    return fig
