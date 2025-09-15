import json
import streamlit as st
from t2sql.controller.ingest_documentation import (
    update_business_rules,
)
import time


def display_rules_tab():
    # App header
    st.title("Business rules")
    st.markdown("""
    ### How to use:
    1. Click 'Add New Rule' to add more text entries
    2. Edit the text in each text area
    3. Use the 🗑️ button to remove unwanted entries
    4. Click 'Save All Rules' to save your changes
    5. All texts will be saved to a local file named `system_prompts.json` under descriptor directory (`descriptors/default` as default dir).
    """)

    # Initialize session state for the rules if it doesn't exist
    if "texts" not in st.session_state:
        st.session_state.texts = st.session_state.agent.business_rules

    if "export_rules" not in st.session_state:
        st.session_state.export_rules = False
        if len(st.session_state.texts) > 0:
            st.session_state.export_rules = True

    loaded_texts_json = json.dumps(
        [text.strip() for text in st.session_state.texts if text.strip()], indent=4
    )
    if st.session_state.export_rules:
        st.download_button(
            "Export Data",
            loaded_texts_json,
            key="export_system_prompt",
            file_name="system_prompt.json",
            mime="text/json",
        )

    st.divider()

    # Create text areas for each entry with delete buttons
    updated_texts = []
    for i, text in enumerate(st.session_state.texts):
        col1, col2 = st.columns([10, 1])
        with col1:
            with st.expander(f"Rule #{i + 1}", expanded=False):
                # Text area for each entry
                text_content = st.text_area(
                    label="business rules", value=text, height=200, key=f"text_{i}"
                )
            updated_texts.append(text_content)
        with col2:
            # Delete button for each entry
            if st.button("🗑️", key=f"delete_prompt_{i}"):
                del st.session_state.texts[i]
                if len(st.session_state.texts) == 0:
                    st.session_state.export_rules = False
                st.toast("Rule was deleted successfully")
                time.sleep(3)
                st.rerun()

    # Add new text field button
    if st.button("Add New Rule"):
        st.session_state.texts.append("")
        st.rerun()

    # Save all texts button
    if st.button("Save All Rules", type="primary"):
        if any(
            text.strip() for text in updated_texts
        ):  # Check if any text is not empty
            # Update session state
            st.session_state.texts = updated_texts
            # Save to JSON file
            st.session_state.processing = True
            update_business_rules(updated_texts, st.session_state.agent)
            st.session_state.export_rules = True
            st.session_state.processing = False
            st.toast("Rules were updated successfully")
            time.sleep(3)
            st.rerun()
        else:
            st.warning("Please enter some text before saving.")
