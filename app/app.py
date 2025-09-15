import os
import streamlit as st
from t2sql.agent import get_sql_agent
import asyncio

# Global loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


def run_async_function(async_func, *args, **kwargs):
    return loop.run_until_complete(async_func(*args, **kwargs))


def initialize_session_state():
    """Initialize session state variables"""
    if "question" not in st.session_state:
        st.session_state.question = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "sql" not in st.session_state:
        st.session_state.sql = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "editing_sql" not in st.session_state:
        st.session_state.editing_sql = None
    if "descriptor_base_path" not in st.session_state:
        st.session_state.descriptor_base_path = None
    if "agent" not in st.session_state:
        st.session_state.agent = get_sql_agent()
        try:
            os.makedirs(st.session_state.agent._docs_md_folder)
            os.mkdir(st.session_state.agent._examples_extended_folder)
        except:
            pass


def main():
    from app.pages.playground import display_chat_tab
    from app.pages.knowledge_base import display_data_tab
    from app.pages.examples import display_examples_tab
    from app.pages.business_rules import display_rules_tab
    from app.pages.settings import display_settings_tab
    from app.pages.sql_instructions import display_instruction_tab

    st.set_page_config(
        page_title="Datrics Text2SQL", layout="wide", initial_sidebar_state="expanded"
    )

    initialize_session_state()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Playground", "Documentation", "Examples", "Business rules", "SQL instructions", "Settings"]
    )

    with tab1:
        display_chat_tab()

    with tab2:
        display_data_tab()

    with tab3:
        display_examples_tab()

    with tab4:
        display_rules_tab()

    with tab5:
        display_instruction_tab()

    with tab6:
        display_settings_tab()

