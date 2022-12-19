import numpy as np
import plotly.graph_objects as go
from Constants import *

def get_threshold_tables(session):
    # Get 3x3 tables for the areas in each of our sections in our threshold graph.
    # Make those into 4x4 tables via sum rows (to show area in that entire section)

    # Compute values for all octants for the text to go in each - # of values inside and % of values inside.
    # We create masks for each to avoid having to repeat our calculations for the corners, where we AND them together.

    # I love boolean logic
    mask_x_hi = session.x.data > session.x.hi
    mask_x_lo = session.x.data < session.x.lo
    mask_x_mid = ~mask_x_hi & ~mask_x_lo

    mask_y_hi = session.y.data > session.y.hi
    mask_y_lo = session.y.data < session.y.lo
    mask_y_mid = ~mask_y_hi & ~mask_y_lo

    mask_mid = mask_x_mid & mask_y_mid

    # Middle one is actually the most complicated in terms of logic,
    # mask_mid = np.logical_and(np.logical_and(session.x.data >= session.x.lo, session.x.data <= session.x.hi),
    #                           np.logical_and(session.y.data >= session.y.lo, session.y.data <= session.y.hi))

    # len
    n = len(session.x.data)  # EXPECTS X AND Y TO BE SAME LENGTH

    # Generic way to chain together masks and get the total # of samples that satisfy all conditions
    area = lambda *masks: np.sum(np.logical_and.reduce([mask for mask in masks])) # note to self always use [] with *
    perc = lambda a: a/n*100 # a = result of area()

    # Given this, now we can assemble the 9 values
    # We also leave extra spaces for the 7 sum-total values.
    grid_ns = np.array([
        [area(mask_x_lo, mask_y_hi),  area(mask_x_mid, mask_y_hi),  area(mask_x_hi, mask_y_hi),     0],
        [area(mask_x_lo, mask_y_mid), area(mask_mid),               area(mask_x_hi, mask_y_mid),    0],
        [area(mask_x_lo, mask_y_lo),  area(mask_x_mid, mask_y_lo),  area(mask_x_hi, mask_y_lo),     0],
        [0,                           0,                            0,                              0]
    ], dtype=np.object)

    # Compute totals
    # 3 verticals and 3 horizontals
    for i in range(3):
        grid_ns[i, -1] = np.sum(grid_ns[i, :-1]) # row
        grid_ns[-1, i] = np.sum(grid_ns[:-1, i]) # col

    # Sum total at the end (equal to either col or row sum)
    grid_ns[-1,-1] = np.sum(grid_ns[:,-1])

    # Percentages is basically all the values but as percentages, so pretty ez.
    grid_ps = [[perc(x) for x in row] for row in grid_ns]

    # Tried to add vertical alignment to plotly table cells via CSS, but it doesn't work with plotly's SVG attribute blocking it.
    # So instead we have to do newline stuff.
    for i in range(4):
        for j in range(4):
            grid_ns[i][j] = f" <br>{grid_ns[i][j]}"
            grid_ps[i][j] = f" <br>{grid_ps[i][j]:.2f}"

    # Chosen to match the ones generated by plotly for these colors
    red = "#f9c9cc"
    blue = "#c6c9ff"
    purple = "#d1a1cc"

    #redtotal
    # added 16% to the saturation values of the earlier red + blue values.
    totalred = "#faa2a8"
    totalblue = "#9ea1ff"

    totaltotal = "#d180c9"
    # transposed because plotly transposes the table
    colors = [
        [purple, blue, purple, totalblue],
        [red, "white", red, totalblue],
        [purple, blue, purple, totalblue],
        [totalred, totalred, totalred, totaltotal]
    ]

    # Plotly does this the reverse way so we transpose to match them
    # REMEMBER TO DO THIS LAST
    grid_ns = np.transpose(grid_ns)
    grid_ps = np.transpose(grid_ps)
    fig_ns = go.Figure(data=[go.Table(
        header=None,
        cells=dict(values=grid_ns, height=80),
    )])
    fig_ps = go.Figure(data=[go.Table(
        header=None,
        cells=dict(values=grid_ps, height=80, align="center", suffix="%"),
    )])
    # Remove headers via making invisible
    fig_ns.for_each_trace(lambda t: t.update(header_fill_color='rgba(0,0,0,0)'))
    fig_ps.for_each_trace(lambda t: t.update(header_fill_color='rgba(0,0,0,0)'))
    fig_ns.update_traces(cells_fill_color=colors)
    fig_ps.update_traces(cells_fill_color=colors)

    # Update this so there's room for everything
    fig_ns.update_layout(font_size=FONTSIZE, margin=dict(t=10, b=10, l=10, r=10))
    fig_ps.update_layout(font_size=FONTSIZE, margin=dict(t=10, b=10, l=10, r=10))
    return fig_ns, fig_ps
