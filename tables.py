import numpy as np
import plotly.graph_objects as go

def get_threshold_tables(session):
    # Get 3x3 tables for the areas in each of our sections in our threshold graph.

    # Compute values for all octants for the text to go in each - # of values inside and % of values inside.
    # We create masks for each to avoid having to repeat our calculations for the corners, where we AND them together.
    mask_x_hi = session.x.data > session.x.hi
    mask_x_lo = session.x.data < session.x.lo
    mask_y_hi = session.y.data > session.y.hi
    mask_y_lo = session.y.data < session.y.lo

    # Middle one is actually the most complicated in terms of logic
    mask_mid = np.logical_and(np.logical_and(session.x.data >= session.x.lo, session.x.data <= session.x.hi),
                              np.logical_and(session.y.data >= session.y.lo, session.y.data <= session.y.hi))

    # len
    n = len(session.x.data)  # EXPECTS X AND Y TO BE SAME LENGTH

    mask_n = lambda m: np.sum(m)  # compute # in masked area
    mask_and_n = lambda m1, m2: np.sum(np.logical_and(m1, m2))  # compute # in intersecting masked area

    mask_p = lambda m: round(mask_n(m) / n * 100., 4)  # compute % of whole in masked area
    mask_and_p = lambda m1, m2: round(mask_and_n(m1, m2) / n * 100.,
                                      4)  # compute % of whole in intersecting masked area

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
