import time
from datetime import datetime
import streamlit as st
from app.app import run_async_function
from streamlit.runtime.uploaded_file_manager import UploadedFile
from t2sql.controller.ingest_documentation import (
    ingest_text_file,
    get_documentation,
    delete_text_files,
    index_schema,
)


def init_session_state():
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = False


def get_documentation_data():
    return run_async_function(get_documentation, st.session_state.agent)


def get_filtered_data(data, search_query):
    if not search_query:
        return data

    search_query = search_query.lower()
    return [
        item
        for item in data
        if search_query in item["name"].lower()
        or search_query in item["document"].lower()
    ]


async def process_data(data: UploadedFile, type: str = "md") -> None:
    if type in ["md", "txt"]:
        content = data.read().decode("utf-8")
        await ingest_text_file(data.name, content, st.session_state.agent)


def handle_uploaded_files(uploaded_files: list[UploadedFile] | None):
    files_count = len(uploaded_files)
    current_progress = 0
    files_processed = []

    if uploaded_files and files_count > 0:
        progress_bar = st.progress(0)

        try:
            st.session_state.processing = True
            for file in uploaded_files:
                current_progress += 1
                run_async_function(process_data, file, type="txt")
                progress_bar.progress(
                    current_progress / files_count, text=f"Processing: {file.name}"
                )
                files_processed.append(file.name)
        except Exception as e:
            st.error(f"Error processing file #{current_progress}: {str(e)}")
        finally:
            st.session_state.processing = False
            progress_bar.empty()

    if current_progress > 0:
        st.session_state["file_uploader"] = None
        text = "All files processed successfully!"
        if len(files_processed) > 0:
            text += "\nFiles processed: \n"
            text += ", ".join(files_processed)
        st.success(text)


def delete_items(doc_name: str):
    st.session_state.processing = True
    run_async_function(delete_text_files, doc_name, st.session_state.agent)
    st.session_state.processing = False


def index_schema_tables():
    st.session_state.processing = True
    run_async_function(index_schema, st.session_state.agent)
    st.session_state.processing = False


def clean_search():
    if "last_search_time" in st.session_state:
        del st.session_state["last_search_time"]
    if "last_search_query" in st.session_state:
        del st.session_state["last_search_query"]
    if "filtered_results" in st.session_state:
        del st.session_state["filtered_results"]


def show_documentation_data():
    data = get_documentation_data()

    # Initialize session state for search
    if "last_search_time" not in st.session_state:
        st.session_state.last_search_time = datetime.now()
    if "last_search_query" not in st.session_state:
        st.session_state.last_search_query = ""
    if (
        "filtered_results" not in st.session_state
        or not st.session_state.filtered_results
    ):
        st.session_state.filtered_results = data

    search_query = st.text_input("🔍 Search in docs...", key="search_history")
    # Check if search query changed
    if search_query != st.session_state.last_search_query:
        st.session_state.last_search_time = datetime.now()
        st.session_state.last_search_query = search_query
        time.sleep(1)  # Wait for 1 second
        st.session_state.filtered_results = get_filtered_data(data, search_query)

    filtered_data = st.session_state.filtered_results

    # Display number of results if searching
    if search_query:
        st.write(f"Found {len(filtered_data)} results")

    # Create headers
    if filtered_data:
        # First create headers
        st.write("### Documentation")
        # Then create rows for each item
        with st.container():
            index = 0
            for item in filtered_data:
                with st.container():
                    tcol1, tcol2 = st.columns([10, 2])
                    with tcol1:
                        with st.expander(f"{item['name']}", expanded=False):
                            st.code(item["document"], language="markdown")
                    with tcol2:
                        if st.button(
                            "🗑️",
                            key=f"delete_{index}",
                        ):
                            delete_items(item["name"])
                            st.session_state.filtered_results = None
                            st.toast("Doc was deleted successfully")
                            time.sleep(3)
                            st.rerun()
                index += 1
    elif search_query:
        st.write("No results found")


def display_data_tab():
    st.title("Knowledgebase")

    init_session_state()

    st.markdown("""Here you can manage the knowledgebase of your agent.""")
    st.markdown("""
    Supported files formats are: txt and md.
    If you upload file with the filename that already exists
    it will reindex the file.
    """)

    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Choose a *.md or *.txt file with documentation",
            type=["txt", "md"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Upload")

    if submitted and uploaded_files is not None and len(uploaded_files) > 0:
        handle_uploaded_files(uploaded_files)
        st.session_state.files_processed = True
        st.session_state.filtered_results = None
        clean_search()

    st.divider()

    st.markdown(
        "If you do not have documentation you can run schema scan to index the tables metadata. It will not reindex already indexed tables loaded as docs."
    )

    if st.button(
        "Run schema indexing",
        key="index_schema",
    ):
        with st.spinner("Indexing tables..."):
            index_schema_tables()
            st.rerun()

    st.divider()

    show_documentation_data()
