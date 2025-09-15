import json
import streamlit as st
from app.app import run_async_function
from t2sql.controller.ingest_documentation import (
    load_examples,
    delete_example,
    ingest_example,
)
import time


def change_add_example_button_states():
    st.session_state.button_add_example_disabled = True
    st.session_state.button_save_example_disabled = False


def change_save_example_button_states():
    st.session_state.button_add_example_disabled = False
    st.session_state.button_save_example_disabled = True


def save_example(question: str, sql: str):
    if question.strip() and sql.strip():
        st.session_state.processing = True
        run_async_function(
            ingest_example,
            question,
            sql,
            st.session_state.agent,
        )
        st.session_state.processing = False
        st.session_state.export_examples = True


def remove_example(question: str):
    st.session_state.processing = True
    run_async_function(
        delete_example,
        question,
        st.session_state.agent,
    )
    st.session_state.processing = False
    st.rerun()


def display_examples_tab():
    # App header
    st.title("SQL Examples")
    st.markdown("""
    ### How to use:
    1. Click 'Add New Example' to add more text entries
    2. Edit the text in each text area
    3. Use the 🗑️ button to remove unwanted entries
    4. Click 'Save Example' to save your changes
    5. Questions should be unique and will be overwritten if duplicates
    6. All examples will be saved to a local folder from variable `examples_extended_folder` from descriptor, default is local dir `training_data_storage/train_examples`
    """)

    # Initialize session state for the rules if it doesn't exist
    if "examples" not in st.session_state:
        st.session_state.examples = load_examples(st.session_state.agent)

    if "button_add_example_disabled" not in st.session_state:
        st.session_state.button_add_example_disabled = False

    if "button_save_example_disabled" not in st.session_state:
        st.session_state.button_save_example_disabled = True

    if "export_examples" not in st.session_state:
        st.session_state.export_examples = False
        if len(st.session_state.examples) > 0:
            st.session_state.export_examples = True

    loaded_texts_json = json.dumps(
        [
            example
            for example in st.session_state.examples
            if example["question"].strip()
        ],
        indent=4,
    )
    if st.session_state.export_examples:
        st.download_button(
            "Export Data",
            loaded_texts_json,
            key="export_examp",
            file_name="examples.json",
            mime="text/json",
        )

    st.divider()

    updated_texts = []
    for i, text in enumerate(st.session_state.examples):
        with st.container():
            col1, col2 = st.columns([10, 1])
            with col1:
                with st.expander(f"Example #{i + 1}", expanded=False):
                    question = st.text_area(
                        "Question:",
                        value=text["question"],
                        height=70,
                        key=f"question_{i}",
                        disabled=True if i < len(st.session_state.examples) - 1 else False
                    )
                    sql = st.text_area(
                        "SQL:", value=text["sql"], height=230, key=f"sql_{i}",
                        disabled=True if i < len(st.session_state.examples) - 1 else False
                    )
                updated_texts.append({"question": question, "sql": sql})
            with col2:
                if st.button("🗑️", key=f"delete_example_{i}"):
                    with st.spinner("Removing example..."):
                        remove_example(text["question"])
                        del st.session_state.examples[i]
                    if len(st.session_state.examples) == 0:
                        st.session_state.export_examples = False

                    st.toast("SQL Example was deleted successfully")
                    time.sleep(3)
                    st.rerun()

    # Add new text field button
    if st.button(
            "Add New Example",
            on_click=change_add_example_button_states,
            disabled=st.session_state.button_add_example_disabled,
    ):
        st.session_state.examples.append({"question": "", "sql": ""})
        st.rerun()

    # Save latest example
    if st.button(
            "Save Example",
            type="primary",
            on_click=change_save_example_button_states,
            disabled=st.session_state.button_save_example_disabled,
    ):
        with st.spinner("Adding example..."):
            save_example(updated_texts[-1]["question"], updated_texts[-1]["sql"])

        st.toast("SQL Example was saved successfully")
        time.sleep(3)
        st.rerun()
