import streamlit as st
import pandas as pd
import numpy as np
from utils.helpers import get_ai_response

def display_data_overview(data):
    """Displays the data preview, shape, dtypes, and descriptive statistics."""
    st.subheader("About Dataset")
    
    with st.expander("Data Preview"):
        rows = st.slider(
            "Number of rows to display", 
            min_value=1, 
            max_value=len(data), 
            value=5, 
            key="slider_rows"
        )
        st.dataframe(data.head(rows))

    with st.expander("Dataset Information"):
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_columns = data.select_dtypes(include=['bool']).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write("##### Column Data Types")
            st.dataframe(pd.DataFrame({
                "Column": data.columns, 
                "Data Type": [str(dtype) for dtype in data.dtypes]
            }))
            st.write(f"**Total Rows:** {data.shape[0]} | **Total Columns:** {data.shape[1]}")

        with col2:
            st.write("**Numerical Columns**")
            st.write(numeric_columns if numeric_columns else "None")
            
            if boolean_columns:
                st.write("**Boolean Columns**")
                st.write(boolean_columns)

        with col3:
            st.write("**Categorical Columns**")
            st.write(categorical_columns if categorical_columns else "None")

            if datetime_columns:
                st.write("**Datetime Columns**")
                st.write(datetime_columns)

    with st.expander("Show Descriptive Statistics"):
        st.write(data.describe())
        if data.isna().any().any():
            st.info("🔔 Missing values were detected in the dataset.")

    st.subheader("AI-Powered Summary")
    if st.button("Generate AI Summary of Dataset", key="ai_summary_button"):
        with st.spinner("🤖 Generating AI summary... This may take a moment."):
            df_head_str = data.head().to_string()
            
            prompt = f"""
            You are an expert data analyst. Analyze the following dataset information and provide a concise, insightful summary for a non-technical user.

            **Dataset Information:**
            - Shape (rows, columns): {data.shape}
            - Column data types: 
            {data.dtypes.to_string()}
            - Missing values per column:
            {data.isnull().sum().to_string()}

            **First 5 rows of data:**
            {df_head_str}

            **Summary Guidelines:**
            1.  Start with a general overview of the dataset's size and content.
            2.  Highlight any potential data quality issues, such as missing values (and in which columns).
            3.  Point out the mix of data types (e.g., "The dataset contains a mix of numerical, categorical, and date-based information.").
            4.  Suggest one or two interesting initial questions that could be explored with this data.
            5.  Keep the summary to 3-4 paragraphs.
            """
            
            summary = get_ai_response(prompt)
            if summary:
                st.success("AI Summary")
                st.markdown(summary)
    st.markdown("---")