import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import init_session_state, get_ai_response, backup_data
from utils.code_validator import CodeValidator
from components.overview import display_data_overview
from components.cleaning import display_cleaning
from components.visualization import display_visualizations
from components.analysis import display_advanced_analytics
from components.time_series import display_time_series_analysis

st.set_page_config(
    page_title="Data Spectra",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("✨ Data Spectra")
st.write("**By: Arnab** | An Interactive EDA Tool")
st.markdown("---")

with st.container(border=True):
    st.header("Get Started: Upload Your Dataset")
    st.markdown("""
        Welcome to DataSpectra! To begin your analysis, please upload a CSV file. 
        This tool will help you clean, visualize, and analyze your data to uncover hidden insights.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/256/8242/8242984.png", width=200)

    with col2:
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV file here or click to browse.", 
            type=["csv"],
            label_visibility="collapsed"
        )
        if uploaded_file is None:
            st.info("📂 Waiting for you to upload a CSV file...")
        else:
            st.success("✅ File uploaded successfully! Your analysis is ready below.")


if uploaded_file is not None:
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.clear()
        init_session_state(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name
        st.rerun()

    if st.session_state.get("original_data") is not None and not st.session_state.original_data.empty:
        
        st.subheader("💬 Ask Your Data",
                     help="Use this query feature for data manipulation and analysis. For visualizations, use the AI feature in the Visualization Gallery.")
        st.markdown("Use Short & Precise Query in Natural Language for data operations.")
        
        with st.form(key='query_form', clear_on_submit=True):
            query_text = st.text_input("Enter your question:", key="data_query_input", value="")
            submitted = st.form_submit_button("Enter")
        
        if 'last_ai_result' not in st.session_state:
            st.session_state.last_ai_result = None
        if 'last_query' not in st.session_state:
            st.session_state.last_query = ""

        if submitted and query_text:
            cleaned_query = query_text.lower().strip()

            if cleaned_query in ["undo", "undo the last change", "undo last change"]:
                if st.session_state.data_history:
                    st.session_state.redo_history.append(st.session_state.processed_data.copy())
                    st.session_state.processed_data = st.session_state.data_history.pop()
                    st.success("Undo successful.")
                    st.session_state.last_ai_result = st.session_state.processed_data
                    st.session_state.last_query = "Undo operation"
                    st.rerun()
                else:
                    st.toast("Nothing to undo.")
            
            elif cleaned_query in ["redo", "redo the last change", "redo last change"]:
                if st.session_state.redo_history:
                    st.session_state.data_history.append(st.session_state.processed_data.copy())
                    st.session_state.processed_data = st.session_state.redo_history.pop()
                    st.success("Redo successful.")
                    st.session_state.last_ai_result = st.session_state.processed_data
                    st.session_state.last_query = "Redo operation"
                    st.rerun()
                else:
                    st.toast("Nothing to redo.")

            else:
                with st.spinner("🤖 Generating and executing code..."):
                    df = st.session_state.processed_data.copy() 
                    
                    prompt = f"""
                    You are a code-generation expert for data analysis in Python using the pandas library.
                    A user has a pandas DataFrame named `df`. The columns of the DataFrame are: {', '.join(df.columns)}.
                    The user has asked the following question: "{query_text}"

                    Your task is to generate a Python code snippet to answer this question.

                    **RULES:**
                    - Respond with ONLY the Python code. Do not include explanations, comments, or the word "python".
                    - The code can be multiple lines.
                    - The final output (e.g., a DataFrame, a Series, or a scalar value) MUST be assigned to a variable named `result`.
                    - DO NOT generate any plots or visualizations. The output must be a data object.
                    - DO NOT use the `print()` function.
                    - Do not use `inplace=True` in any operations.
                    """
                    
                    generated_code = get_ai_response(prompt)
                    
                    if generated_code:
                        is_safe, error_message = CodeValidator.validate(generated_code)
                        if is_safe:
                            try:
                                local_vars = {'df': df, 'pd': pd, 'np': np}
                                
                                exec(generated_code, {}, local_vars)
                                
                                result = local_vars.get('result', None)
                                
                                st.session_state.last_ai_result = result
                                st.session_state.last_query = query_text

                                if isinstance(result, pd.DataFrame) and not df.equals(result):
                                    backup_data()
                                    st.session_state.processed_data = result
                                    st.session_state.redo_history.clear()
                                    
                            except Exception as e:
                                st.session_state.last_ai_result = f"An error occurred: {e}"
                                st.session_state.last_query = query_text
                        else:
                            st.session_state.last_ai_result = f"Execution stopped: {error_message}"
                            st.session_state.last_query = query_text
                    else:
                        st.session_state.last_ai_result = "Warning: The AI model could not generate a response."
                        st.session_state.last_query = query_text
        
        if st.session_state.last_ai_result is not None and st.session_state.last_query:
            st.write("🔍 **Result:**")
            result = st.session_state.last_ai_result
            
            if isinstance(result, pd.DataFrame):
                st.info("The query resulted in a modified DataFrame. Here is a preview:")
                st.dataframe(result.head())
            elif isinstance(result, pd.Series):
                st.dataframe(result)
            elif isinstance(result, str):
                st.error(result)
            else:
                 st.write(result)


        display_data_overview(st.session_state.processed_data)
        display_cleaning()
        display_visualizations(st.session_state.processed_data)
        display_advanced_analytics(st.session_state.processed_data)
        display_time_series_analysis(st.session_state.processed_data)
    
    else:
        st.error("The uploaded file could not be processed. Please try another file. 😥")
