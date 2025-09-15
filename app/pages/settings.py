import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from t2sql.agent import get_sql_agent
import json
import os
import time


def get_descriptors_folder(file: UploadedFile):
    return json.loads(file.read().decode("utf-8")).get("descriptors_path")


def display_settings_tab():
    st.title("Settings")

    st.markdown(
        "Choose`t2sql_descriptor.json`, if not chosen, the default descriptor will be created/used under `descriptors/default/` local path."
    )

    if "descriptor_base_path" in st.session_state:
        st.markdown(
            f"Descriptor path: `{st.session_state.descriptor_base_path or 'descriptors/default/t2sql_descriptor.json'}`"
        )

    with st.form("choose_descriptor", clear_on_submit=True):
        descriptor_file = st.file_uploader(
            "Choose t2sql_descriptor.json",
            type=["json"],
            accept_multiple_files=False,
        )
        submitted = st.form_submit_button("Upload")
        if submitted and descriptor_file:
            st.session_state.descriptor_base_path = get_descriptors_folder(
                descriptor_file
            )
            st.session_state.agent = get_sql_agent(
                st.session_state.descriptor_base_path
            )
            try:
                os.makedirs(st.session_state.agent._docs_md_folder)
                os.mkdir(st.session_state.agent._examples_extended_folder)
            except:
                pass
            st.toast("Reloaded SQL agent")
            time.sleep(3)
            st.rerun()
