import traceback
import streamlit as st
import numpy as np
from Constants import *
from graphs import *
from tables import *
# TODO set this up so that they have to press update to update the graph?
# If the user hasn't specified values for this, then don't show anything yet.
def generate_threshold_section(session):
    # Update these attributes when we have columns for them
    gcol1x, gcol1y, gcol2, gcol3 = st.columns((1, 1, 4, 2), gap="small")
    session.x.update(session.data)
    session.y.update(session.data)
    xname = session.x.col
    yname = session.y.col

    # PROBLEM: TODO: Updates to values in later widgets don't update earlier ones. if i change the number it doesn't move slider.
    # Unfortunately this is a limitation of streamlit, and can't be fixed yet. Fortuantely, it still changes the graph.
    try:
        with gcol1x:
            # Settings for x threshold
            session.x.lo, session.x.hi = st.slider(
                f'{xname} Threshold',
                session.x.min, session.x.max, session.x.interval(2), 0.01, format="%0.2f")

            # Can also be controlled with more granularity
            session.x.lo = st.number_input(
                f'{xname} Lower Threshold',
                session.x.min, session.x.hi, session.x.lo, 0.0001, format="%0.4f")
            session.x.hi = st.number_input(
                f'{xname} Higher Threshold',
                session.x.lo, session.x.max, session.x.hi, 0.0001, format="%0.4f")
            st.markdown("---")

            # And via percentiles
            session.x.lo = perc2num(session.x.data, st.number_input(
                f'{xname} Lower Threshold (Percentile)',
                0., num2perc(session.x.data, session.x.hi), num2perc(session.x.data, session.x.lo), .1,
                format="%.2f"))  # Streamlit does not allow % symbol here
            session.x.hi = perc2num(session.x.data, st.number_input(
                f'{xname} Higher Threshold (Percentile)',
                num2perc(session.x.data, session.x.lo), 100., num2perc(session.x.data, session.x.hi), .1,
                format="%.2f"))
            st.markdown("---")

            # And via stddevs
            session.x.lo = std2num(session.x.std, st.number_input(
                f'{xname} Lower Threshold (Mult. of STD)',
                0., num2std(session.x.std, session.x.hi), num2std(session.x.std, session.x.lo), .1, format="%.2f"))
            session.x.hi = std2num(session.x.std, st.number_input(
                f'{xname} Higher Threshold (Mult. of STD)',
                num2std(session.x.std, session.x.lo), 100., num2std(session.x.std, session.x.hi), .1, format="%.2f"))

    except st.errors.StreamlitAPIException:
        st.write(ERROR_MSG_TEMPLATE.format(yname))
        print(traceback.format_exc())
        st.write(traceback.format_exc())
        st.write(ERROR_MSG_TEMPLATE.format(xname))
        print(traceback.format_exc())
        st.write(traceback.format_exc())
    try:
        with gcol1y:
            # Settings for y threshold
            session.y.lo, session.y.hi = st.slider(
                f'{yname} Threshold',
                session.y.min, session.y.max, session.y.interval(2), 0.01, format="%0.2f")

            session.y.lo = st.number_input(
                f'{yname} Lower Threshold',
                session.y.min, session.y.hi, session.y.lo, 0.0001, format="%0.4f")
            session.y.hi = st.number_input(
                f'{yname} Higher Threshold',
                session.y.lo, session.y.max, session.y.hi, 0.0001, format="%0.4f")
            st.markdown("---")

            # And via percentiles
            session.y.lo = perc2num(session.y.data, st.number_input(
                f'{yname} Lower Threshold (Percentile)',
                0., num2perc(session.y.data, session.y.hi), num2perc(session.y.data, session.y.lo), .1,
                format="%.2f"))  # Streamlit does not allow % symbol here
            session.y.hi = perc2num(session.y.data, st.number_input(
                f'{yname} Higher Threshold (Percentile)',
                num2perc(session.y.data, session.y.lo), 100., num2perc(session.y.data, session.y.hi), .1,
                format="%.2f"))
            st.markdown("---")

            # And via stddevs
            session.y.lo = std2num(session.y.std, st.number_input(
                f'{yname} Lower Threshold (Mult. of STD)',
                0., num2std(session.y.std, session.y.hi), num2std(session.y.std, session.y.lo), .1, format="%.2f"))
            session.y.hi = std2num(session.y.std, st.number_input(
                f'{yname} Higher Threshold (Mult. of STD)',
                num2std(session.y.std, session.y.lo), 100., num2std(session.y.std, session.y.hi), .1, format="%.2f"))

            # on change, change the lines.
            # st.write('Values:', values) # and then we can add on the % etc.
    except st.errors.StreamlitAPIException:
        st.write(ERROR_MSG_TEMPLATE.format(yname))
        print(traceback.format_exc())
        st.write(traceback.format_exc())
    with gcol2:
        # TODO remove width stuff?
        graph_container = st.container()
        fig = get_threshold_graph(session, session.data_master)
        # Set up some reasonable margins and heights so we actually get a more square-like graph
        # rather than the wide boi streamlit wants it to be
        # fig.layout.height=1000
        # fig.layout.margin=dict(l=100, r=100, t=0, b=0)
        graph_container.plotly_chart(fig, use_container_width=True)
    with gcol3:
        table_container = st.container()
        table_ns, table_ps, grid_ns, grid_ps = get_threshold_tables(session)
        table_container.markdown("### Threshold Areas")
        table_container.plotly_chart(table_ns, use_container_width=True)
        table_container.markdown("### Threshold Area Percentages")
        table_container.plotly_chart(table_ps, use_container_width=True)
        return session, grid_ns, grid_ps
    #

def generate_singlevar_section(session):
    #gcol1x, gcol1y, gcol2, gcol3 = st.columns((1, 1, 4, 2), gap="small")
    session.x.update(session.data)
    xname = session.x.col

    #fig = px.histogram(df, x=i, title=i, nbins=st.slider("Number of Bins", key=str(new_tab) + i))
    graph_container = st.container()
    fig = get_singlevar_histogram(session)
    graph_container.plotly_chart(fig, use_container_width=True)

    return session