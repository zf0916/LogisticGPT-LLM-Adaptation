from typing import Optional
import streamlit as st
from streamlit_monaco import st_monaco
from app.utils import handle_error
from t2sql.controller.make_answer import make_answer, run_sql
from t2sql.controller.ingest_documentation import ingest_example
import time
from app.app import run_async_function

from evalx.canonicalization import canonicalize_sql

async def process_question(question: str, sql_query: Optional[str] = None):
    try:
        st.session_state.processing = True
        st.session_state.editing_sql = None
        sql_placeholder = st.empty()

        with st.spinner("Generating SQL..."):
            if sql_query:
                sql = sql_query
            else:
                sql = await make_answer(question, st.session_state.agent)
                sql_placeholder.code(sql, language="sql")

            st.session_state.question = question

        # Add canonicalization step here to help process OS model query generation
        sql = canonicalize_sql(sql)

        with st.spinner("Retrieving data..."):
            new_sql, df = await run_sql(sql, st.session_state.agent)
            if new_sql != sql:
                st.toast("SQL has been updated during execution")
                time.sleep(3)
            sql_placeholder.code(new_sql, language="sql")
            st.session_state.df = df
            st.session_state.sql = new_sql
            st.session_state.editing_sql = None
    except Exception as e:
        st.error(handle_error(e))
    finally:
        st.session_state.processing = False


async def process_save_example(question: str, sql: str):
    try:
        st.session_state.processing = True
        st.session_state.editing_sql = None

        await ingest_example(question, sql, st.session_state.agent)

        st.toast("SQL Example was saved successfully")
        time.sleep(3)
    except Exception as e:
        st.error(handle_error(e))
    finally:
        st.session_state.processing = False


def display_code_editor(sql: str):
    st.session_state.editing_sql = sql


def display_chat_tab():
    # App header
    st.title("Chat")
    st.markdown("""
            Ask questions about your data and get SQL queries with results in CSV.
        """)

    # Chat input
    question = st.text_input(
        "Ask a question",
        key="question_input",
        placeholder="Enter your question here...",
        disabled=st.session_state.processing
    )

    # Process button
    if st.button("Submit", key="submit_button", disabled=st.session_state.processing):
        if question:
            run_async_function(process_question, question)

    # Loading indicator
    if st.session_state.processing:
        st.markdown(
            """<div class="loading">Processing your question...</div>""",
            unsafe_allow_html=True,
        )

    if st.session_state.sql:
        if st.session_state.editing_sql is None:
            # st.code(st.session_state.sql, language="sql")
            col1, col2, col3 = st.columns([2, 2, 9])
            with col1:
                st.button(
                    "Edit SQL",
                    key="edit_sql",
                    on_click=display_code_editor,
                    args=(st.session_state.sql,),
                )
            with col2:
                if st.button("Save Example", key="save_sql"):
                    with st.spinner("Adding example..."):
                        run_async_function(process_save_example, st.session_state.question, st.session_state.sql,)
            with col3:
                pass
        else:
            editing_result = st_monaco(st.session_state.editing_sql, "200px", "sql")
            if st.button("Execute SQL", key="execute_sql"):
                run_async_function(process_question, st.session_state.question, editing_result,)
            if st.button("Cancel", key="cancel_edit_sql"):
                st.session_state.editing_sql = None
                st.rerun()

    if st.session_state.df is not None:
        if st.session_state.df.empty:
            st.warning("No results returned. Try to ask question once again.")
        else:
            st.write("Results:")
            st.dataframe(st.session_state.df, width='stretch', hide_index=True)
