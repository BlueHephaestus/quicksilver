import plotly.graph_objects as go

#@st.cache(allow_output_mutation=True)
def get_threshold_graph(session, data_master):
    #x = get_x()
    #y = get_y()
    #accessions = data_master["EpisodeNumber"]
    accessions = data_master[session.accession_col]
    if session.scatter_enable:
        # Scatterplot
        hovertemplate="<b>" + session.accession_col + ": %{customdata}</b><br>" + session.x.col + ": %{x}<br>" + session.y.col + ": %{y}<br><extra></extra>"
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
