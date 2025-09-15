import streamlit as st
from t2sql.controller.ingest_documentation import (
    update_prompts,
)
import time


def display_instruction_tab():
    # App header
    st.title("Default SQL instructions")
    st.markdown("""
    ### How to use:
    1. Click 'Edit' to add more text 
    2. Click 'Save All' to save your instructions changes
    """)

    # Initialize session state for the rules if it doesn't exist
    if "sql_instr" not in st.session_state:
        st.session_state.sql_instr = st.session_state.agent.get_prompt_string("DEFAULT_SQL_INSTRUCTIONS")

    st.download_button(
        "Export Data",
        st.session_state.sql_instr,
        key="export_sql_inst",
        file_name="default_sql_instruction.txt",
        mime="text/plain",
    )

    st.divider()

    # Create text area
    text_content = st.text_area(
        label="sql_instructions", value=st.session_state.sql_instr, height=500
    )

    # Save all texts button
    if st.button("Save All", type="primary"):
        st.session_state.sql_instr = text_content.strip()
        update_prompts(text_content.strip(), st.session_state.agent)
        st.toast("SQL instructions were updated successfully")
        time.sleep(3)
        st.rerun()
