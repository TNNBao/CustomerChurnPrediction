import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Introduction", "Prediction"],
        icons=["house", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Introduction":
    import intro as intro
    intro.show()
elif selected == "Prediction":
    import predict as predict
    predict.show()
