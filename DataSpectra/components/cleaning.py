import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import backup_data, get_ai_response
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a UTF-8 encoded CSV file."""
    return df.to_csv(index=False).encode('utf-8')

def visualize_missing_data(data):
    """Displays a summary and a bar chart of missing values."""
    st.info("Visualizing missing data:")
    missing_data_count = data.isnull().sum()
    missing_df = missing_data_count[missing_data_count > 0].reset_index()
    missing_df.columns = ['Column', 'Missing Count']
    
    if not missing_df.empty:
        missing_df['Missing Percent'] = (missing_df['Missing Count'] / len(data)) * 100
        st.dataframe(missing_df.sort_values(by='Missing Count', ascending=False), hide_index=True)
        
        fig = px.bar(
            missing_df, x='Column', y='Missing Count', title='Missing Values per Column',
            labels={'Missing Count': 'Number of Missing Values'},
            color='Missing Percent', color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("🎉 `No missing data to visualize!`")

# ... (rest of the cleaning functions remain the same) ...
def _render_binning_tab(data, numeric_cols):
    """Renders the UI for the Numerical Binning feature."""
    if not numeric_cols:
        st.info("No numerical columns available for binning.")
        return

    col_to_bin = st.selectbox("Select numerical column to bin", ["Select a column"] + numeric_cols, key="bin_select")
    if col_to_bin != "Select a column":
        bins_input = st.text_input("Enter bin edges (comma-separated)", key="bin_edges", placeholder="e.g., 0, 18, 40, 65, 100")
        labels_input = st.text_input("Enter labels for bins (comma-separated)", key="bin_labels", placeholder="e.g., Child, Adult, Senior")

        if st.button("Create Binned Feature", key="bin_button"):
            if bins_input and labels_input:
                try:
                    bins = [float(b.strip()) for b in bins_input.split(',')]
                    labels = [l.strip() for l in labels_input.split(',')]

                    if len(bins) - 1 == len(labels):
                        backup_data()
                        st.session_state.redo_history.clear()
                        new_col_name = f"{col_to_bin}_Group"
                        st.session_state.processed_data[new_col_name] = pd.cut(st.session_state.processed_data[col_to_bin], bins=bins, labels=labels, right=False)
                        st.success(f"Successfully created '{new_col_name}' column.")
                        st.rerun()
                    else:
                        st.error("The number of labels must be one less than the number of bin edges.")
                except Exception as e:
                    st.error(f"An error occurred during binning: {e}")
            else:
                st.warning("Please provide both bin edges and labels.")

def _render_combine_rare_tab(data, categorical_cols):
    """Renders the UI for the Combine Rare Categories feature."""
    if not categorical_cols:
        st.info("No categorical columns available for this operation.")
        return
        
    col_to_group = st.selectbox("Select categorical column to group", ["Select a column"] + categorical_cols, key="group_select")
    if col_to_group != "Select a column":
        threshold = st.slider("Minimum Frequency Threshold", 1, 50, 10, key="group_threshold")
        st.write(f"Categories appearing less than {threshold} times will be grouped into 'Other'.")

        if st.button("Combine Rare Categories", key="group_button"):
            try:
                backup_data()
                st.session_state.redo_history.clear()
                value_counts = st.session_state.processed_data[col_to_group].value_counts()
                to_replace = value_counts[value_counts < threshold].index
                if len(to_replace) > 0:
                    new_col_name = f"{col_to_group}_Grouped"
                    st.session_state.processed_data[new_col_name] = st.session_state.processed_data[col_to_group].replace(to_replace, 'Other')
                    st.success(f"Successfully created '{new_col_name}' column by grouping {len(to_replace)} rare categories.")
                    st.rerun()
                else:
                    st.info("No categories met the threshold for grouping.")
            except Exception as e:
                st.error(f"An error occurred during grouping: {e}")

def _render_polynomial_features_tab(data, numeric_cols):
    """Renders the UI for the Polynomial Features feature."""
    if not numeric_cols:
        st.info("No numerical columns available for polynomial features.")
        return

    poly_col = st.selectbox("Select a numeric column to create polynomial features from", ["Select a column"] + numeric_cols, key="poly_col")
    if poly_col != "Select a column":
        degree = st.slider("Select the polynomial degree", 2, 5, 2, key="poly_degree")
        if st.button("Create Polynomial Features", key="poly_button"):
            try:
                backup_data()
                st.session_state.redo_history.clear()
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(st.session_state.processed_data[[poly_col]])
                
                poly_df = pd.DataFrame(poly_features[:, 1:], columns=[f"{poly_col}^{i}" for i in range(2, degree + 1)], index=st.session_state.processed_data.index)
                
                st.session_state.processed_data = pd.concat([st.session_state.processed_data, poly_df], axis=1)
                st.success(f"Created polynomial features for '{poly_col}' up to degree {degree}.")
                st.rerun()
            except ValueError:
                st.error("Error: Could not create polynomial features. The selected column may contain missing values.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

def _render_combine_categorical_tab(data, categorical_cols):
    """Renders the UI for combining multiple categorical columns."""
    st.write("#### Combine Categorical Columns")
    st.write("Select two or more categorical columns to merge them into a single new column.")

    if len(categorical_cols) < 2:
        st.info("You need at least two categorical columns to use this feature.")
        return

    cols_to_combine = st.multiselect(
        "Select columns to combine",
        options=categorical_cols,
        key="cat_combine_select"
    )

    separator = st.text_input("Enter a separator", "_", key="cat_separator")
    new_col_name = st.text_input("Enter a name for the new combined column", "Combined_Category", key="cat_new_name")
    drop_originals = st.checkbox("Drop original columns after combining", value=True, key="cat_drop")

    if st.button("Combine Columns", key="cat_combine_button"):
        if len(cols_to_combine) < 2:
            st.warning("Please select at least two columns to combine.")
            return
        if not new_col_name:
            st.warning("Please provide a name for the new column.")
            return
        
        try:
            backup_data()
            st.session_state.redo_history.clear()
            df = st.session_state.processed_data
            
            df[new_col_name] = df[cols_to_combine].astype(str).agg(separator.join, axis=1)
            
            if drop_originals:
                df.drop(columns=cols_to_combine, inplace=True)
                st.success(f"Successfully created '{new_col_name}' and dropped original columns.")
            else:
                st.success(f"Successfully created '{new_col_name}'.")
            
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during column combination: {e}")

def _render_combine_numerical_tab(data, numeric_cols):
    """Renders the UI for combining two numerical columns with an operation."""
    st.write("#### Combine Numerical Columns")
    st.write("Select two numerical columns and an operation to create a new feature.")

    if len(numeric_cols) < 2:
        st.info("You need at least two numerical columns to use this feature.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        num_col1 = st.selectbox("Select the first column", ["Select a column"] + numeric_cols, key="num_combine_col1")
    
    if num_col1 != "Select a column":
        with col2:
            operation = st.selectbox("Select Operation", ["+", "-", "*", "/"], key="num_combine_op")
        
        options_col2 = [col for col in numeric_cols if col != num_col1]
        with col3:
            num_col2 = st.selectbox("Select the second column", ["Select a column"] + options_col2, key="num_combine_col2")

        if num_col2 != "Select a column":
            new_col_name_default = f"{num_col1}_{operation}_{num_col2}"
            new_col_name = st.text_input("Enter name for the new column", new_col_name_default, key="num_new_name")

            if st.button("Combine Numerical Columns", key="num_combine_button"):
                if not new_col_name:
                    st.warning("Please provide a name for the new column.")
                    return
                try:
                    backup_data()
                    st.session_state.redo_history.clear()
                    df = st.session_state.processed_data
                    
                    if operation == '+':
                        df[new_col_name] = df[num_col1] + df[num_col2]
                    elif operation == '-':
                        df[new_col_name] = df[num_col1] - df[num_col2]
                    elif operation == '*':
                        df[new_col_name] = df[num_col1] * df[num_col2]
                    elif operation == '/':
                        if (df[num_col2] == 0).any():
                            st.warning("Warning: The second column contains zeros. Division by zero will result in 'inf' values.")
                        df[new_col_name] = df[num_col1] / df[num_col2]

                    st.success(f"Successfully created new column '{new_col_name}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")

def _render_datetime_extraction_tab(data):
    """Renders the UI for extracting components from a datetime column."""
    st.write("#### Extract Datetime Components")
    st.write("Select a datetime column to extract its components (e.g., year, month, day) into new columns.")

    datetime_cols = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    if not datetime_cols:
        st.info("No datetime columns found in the dataset. You can create one using the 'Change Data Type' tab.")
        return

    col_to_extract = st.selectbox(
        "Select a datetime column",
        options=["Select a column"] + datetime_cols,
        key="dt_extract_select"
    )

    if col_to_extract != "Select a column":
        extraction_options = [
            'Year', 'Month', 'Day', 'Day of Week', 'Day Name',
            'Hour', 'Minute', 'Second', 'Quarter', 'Week of Year'
        ]
        parts_to_extract = st.multiselect(
            "Select components to extract",
            options=extraction_options,
            key="dt_parts_select"
        )

        drop_original = st.checkbox("Drop original datetime column after extraction", value=False, key="dt_extract_drop")

        if st.button("Extract Components", key="dt_extract_button"):
            if not parts_to_extract:
                st.warning("Please select at least one component to extract.")
                return

            try:
                backup_data()
                st.session_state.redo_history.clear()
                df = st.session_state.processed_data

                for part in parts_to_extract:
                    new_col_name = f"{col_to_extract}_{part.lower().replace(' ', '_')}"
                    if part == 'Year':
                        df[new_col_name] = df[col_to_extract].dt.year
                    elif part == 'Month':
                        df[new_col_name] = df[col_to_extract].dt.month
                    elif part == 'Day':
                        df[new_col_name] = df[col_to_extract].dt.day
                    elif part == 'Day of Week':
                        df[new_col_name] = df[col_to_extract].dt.dayofweek
                    elif part == 'Day Name':
                        df[new_col_name] = df[col_to_extract].dt.day_name()
                    elif part == 'Hour':
                        df[new_col_name] = df[col_to_extract].dt.hour
                    elif part == 'Minute':
                        df[new_col_name] = df[col_to_extract].dt.minute
                    elif part == 'Second':
                        df[new_col_name] = df[col_to_extract].dt.second
                    elif part == 'Quarter':
                        df[new_col_name] = df[col_to_extract].dt.quarter
                    elif part == 'Week of Year':
                        df[new_col_name] = df[col_to_extract].dt.isocalendar().week

                if drop_original:
                    df.drop(columns=[col_to_extract], inplace=True)
                    st.success(f"Successfully extracted components and dropped '{col_to_extract}'.")
                else:
                    st.success(f"Successfully extracted components from '{col_to_extract}'.")

                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during extraction: {e}")
                
def handle_rename_delete():
    st.subheader("Rename or Delete Columns")
    data = st.session_state.processed_data
    if not data.columns.tolist():
        st.warning("No columns available to modify.")
        return

    options = ["Select a column"] + data.columns.tolist()
    col_to_modify = st.selectbox("Select column", options, key="rename_select_col")

    if col_to_modify != "Select a column":
        rename_col = st.text_input("New name for the selected column", value=col_to_modify, key="rename_input_col")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Rename Column", key="rename_button", type="secondary"):
                if rename_col and rename_col != col_to_modify:
                    backup_data()
                    st.session_state.redo_history.clear() 
                    st.session_state.processed_data.rename(columns={col_to_modify: rename_col}, inplace=True)
                    st.success(f"Renamed '{col_to_modify}' to '{rename_col}'")
                    st.rerun()
        with btn_col2:
            if st.button("Delete Column", key="delete_button", type="primary"):
                backup_data()
                st.session_state.redo_history.clear() 
                st.session_state.processed_data.drop(columns=[col_to_modify], inplace=True)
                st.success(f"Deleted column '{col_to_modify}'")
                st.rerun()

def handle_dtype_change():
    st.subheader("Change Column Data Type")
    data = st.session_state.processed_data
    options = ["Select a column"] + data.columns.tolist()
    col_to_convert = st.selectbox("Select column", options, key="dtype_col_select")

    if col_to_convert != "Select a column":
        current_dtype = str(data[col_to_convert].dtype)
        st.write(f"Current Data Type of `{col_to_convert}`: `{current_dtype}`")
        new_dtype = st.selectbox("Convert to", ["int", "float", "str", "datetime"], key="dtype_target_select")

        if st.button("Convert Data Type", key="convert_button"):
            try:
                backup_data()
                st.session_state.redo_history.clear()
                
                original_non_nulls = data[col_to_convert].notna().sum()

                if new_dtype == "int":
                    st.session_state.processed_data[col_to_convert] = pd.to_numeric(data[col_to_convert], errors='coerce').astype('Int64')
                elif new_dtype == "float":
                    st.session_state.processed_data[col_to_convert] = pd.to_numeric(data[col_to_convert], errors='coerce').astype(float)
                elif new_dtype == "str":
                    st.session_state.processed_data[col_to_convert] = data[col_to_convert].astype(str)
                elif new_dtype == "datetime":
                    st.session_state.processed_data[col_to_convert] = pd.to_datetime(data[col_to_convert], format='mixed')
                
                st.success(f"Successfully converted '{col_to_convert}' to {new_dtype}.")

                final_non_nulls = st.session_state.processed_data[col_to_convert].notna().sum()
                coerced_count = original_non_nulls - final_non_nulls
                
                if coerced_count > 0:
                    st.warning(f"⚠️ {coerced_count} value(s) could not be converted and were set to missing (NaN/NaT).")

                st.rerun()
            except Exception as e:
                st.error(f"Failed to convert: {e}")
                if st.session_state.data_history:
                    st.session_state.processed_data = st.session_state.data_history.pop()

def handle_missing_values():
    st.subheader("Missing Value Imputation")
    data = st.session_state.processed_data
    cols_with_na = data.columns[data.isnull().any()].tolist()
    if not cols_with_na:
        st.success("✅ No missing values detected in the dataset!")
        return

    options = ["Select a column"] + cols_with_na
    col_to_handle_na = st.selectbox("Select column to handle missing values", options, key="missing_col_select")
    
    if col_to_handle_na != "Select a column":
        if pd.api.types.is_numeric_dtype(data[col_to_handle_na]):
            methods = ["Mean", "Median", "Mode", "Interpolation", "Forward Fill", "Backward Fill", "Drop Rows", "Drop Column"]
        else:
            methods = ["Mode", "Forward Fill", "Backward Fill", "Drop Rows", "Drop Column"]
            
        na_method = st.selectbox("Select method", methods, key="na_method_select")

        if st.button("Get AI Suggestion", key="ai_suggestion_button"):
            with st.spinner("🤖 Analyzing column and getting suggestion..."):
                column_data = data[col_to_handle_na].dropna().to_string()
                prompt = f"""
                You are a data cleaning expert. A user wants to impute missing values for a column named '{col_to_handle_na}'.
                The data type is {data[col_to_handle_na].dtype}.
                Here are some sample values from the column:
                {column_data}

                Based on this information, suggest the best imputation method (e.g., Mean, Median, Mode, Interpolation) and provide a one-sentence justification.
                For numerical data, consider the skewness. For categorical data, suggest Mode.
                Respond with only the suggestion and justification. For example: "Suggestion: Median. Justification: This is best for skewed numerical data."
                """
                suggestion = get_ai_response(prompt)
                if suggestion:
                    st.info(f"**AI Suggestion:** {suggestion}")

        if st.button("Apply Missing Value Handling", key="apply_na_button"):
            try:
                backup_data()
                st.session_state.redo_history.clear() 
                df = st.session_state.processed_data
                before_na = df[col_to_handle_na].isnull().sum()

                if na_method == "Mean":
                    df[col_to_handle_na] = df[col_to_handle_na].fillna(df[col_to_handle_na].mean())
                elif na_method == "Median":
                    df[col_to_handle_na] = df[col_to_handle_na].fillna(df[col_to_handle_na].median())
                elif na_method == "Mode":
                    mode_val = df[col_to_handle_na].mode()
                    if not mode_val.empty:
                        df[col_to_handle_na] = df[col_to_handle_na].fillna(mode_val[0])
                    else:
                        st.warning(f"Column '{col_to_handle_na}' has no mode. No values were imputed.")
                elif na_method == "Interpolation":
                    df[col_to_handle_na] = df[col_to_handle_na].interpolate(method='linear')
                elif na_method == "Forward Fill":
                    df[col_to_handle_na] = df[col_to_handle_na].fillna(method='ffill')
                elif na_method == "Backward Fill":
                    df[col_to_handle_na] = df[col_to_handle_na].fillna(method='bfill')
                elif na_method == "Drop Rows":
                    df.dropna(subset=[col_to_handle_na], inplace=True)
                elif na_method == "Drop Column":
                    df.drop(columns=[col_to_handle_na], inplace=True)
                
                after_na = df[col_to_handle_na].isnull().sum() if col_to_handle_na in df.columns else 0
                filled = before_na - after_na
                st.success(f"{filled} missing values in '{col_to_handle_na}' handled using '{na_method}'.")
                st.rerun()

            except Exception as e:
                st.error(f"Error applying method: {e}")
                st.session_state.processed_data = st.session_state.data_history.pop()

def handle_outliers():
    st.subheader("Outlier Detection and Handling")
    data = st.session_state.processed_data
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for outlier detection.")
        return

    options = ["Select a column"] + numeric_cols
    col_outlier = st.selectbox("Select a numeric column to analyze for outliers", options, key="outlier_col_select")

    if col_outlier != "Select a column":
        Q1 = data[col_outlier].quantile(0.25)
        Q3 = data[col_outlier].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[col_outlier] < lower_bound) | (data[col_outlier] > upper_bound)]
        
        st.write(f"**Interquartile Range (IQR):** `{IQR:.2f}`")
        st.write(f"**Lower Bound:** `{lower_bound:.2f}` | **Upper Bound:** `{upper_bound:.2f}`")
        
        if not outliers.empty:
            st.warning(f"Found **{len(outliers)}** potential outliers in `{col_outlier}`.")
        else:
            st.success(f"No outliers detected in `{col_outlier}` based on the IQR method.")

        fig = px.box(data, y=col_outlier, title=f"Box Plot for {col_outlier}", points="all")
        st.plotly_chart(fig, use_container_width=True)

        if not outliers.empty:
            handling_method = st.radio("Choose how to handle outliers:", ["None", "Remove Outliers", "Cap Outliers"], key="outlier_handling")

            if st.button("Apply Outlier Handling", key="apply_outlier_button"):
                if handling_method != "None":
                    backup_data()
                    st.session_state.redo_history.clear()
                    df = st.session_state.processed_data
                    
                    if handling_method == "Remove Outliers":
                        df.drop(outliers.index, inplace=True)
                        st.success(f"Removed {len(outliers)} outliers from `{col_outlier}`.")
                    elif handling_method == "Cap Outliers":
                        df[col_outlier] = np.where(df[col_outlier] < lower_bound, lower_bound, df[col_outlier])
                        df[col_outlier] = np.where(df[col_outlier] > upper_bound, upper_bound, df[col_outlier])
                        st.success(f"Capped outliers in `{col_outlier}` at the lower and upper bounds.")
                    
                    st.rerun()

def handle_categorical_encoding():
    st.subheader("Categorical Variable Encoding")
    data = st.session_state.processed_data
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        st.warning("No categorical columns available for encoding.")
        return

    options = ["Select a column"] + categorical_cols
    col_encode = st.selectbox("Select a categorical column to encode", options, key="encode_col_select")
    
    if col_encode != "Select a column":
        encoding_method = st.selectbox("Select an encoding method", ["One-Hot Encoding", "Label Encoding"], key="encode_method_select")
        
        st.write("#### Preview of Encoding")
        if encoding_method == "One-Hot Encoding":
            st.info("This method creates new columns for each category in the selected column.")
            preview_df = pd.get_dummies(data[[col_encode]], prefix=col_encode)
            st.dataframe(preview_df.head())
        elif encoding_method == "Label Encoding":
            st.info("This method converts each category into a unique integer.")
            le = LabelEncoder()
            try:
                preview_series = pd.Series(le.fit_transform(data[col_encode]), name=f"{col_encode}_encoded")
                st.dataframe(preview_series.head())
            except Exception:
                st.warning("Could not generate a preview for Label Encoding.")

        drop_original = st.checkbox("Drop original column after encoding", value=True, key="encoding_drop_col")

        if st.button("Apply Encoding", key="apply_encode_button"):
            backup_data()
            st.session_state.redo_history.clear()
            df = st.session_state.processed_data

            if encoding_method == "One-Hot Encoding":
                one_hot_df = pd.get_dummies(df[col_encode], prefix=col_encode)
                st.session_state.processed_data = pd.concat([df, one_hot_df], axis=1)
                if drop_original:
                    st.session_state.processed_data.drop(col_encode, axis=1, inplace=True)
                st.success(f"Applied One-Hot Encoding to `{col_encode}`.")

            elif encoding_method == "Label Encoding":
                le = LabelEncoder()
                if not drop_original:
                    new_col_name = f"{col_encode}_encoded"
                    df[new_col_name] = le.fit_transform(df[col_encode])
                    st.success(f"Applied Label Encoding to `{col_encode}` and created new column `{new_col_name}`.")
                else:
                    df[col_encode] = le.fit_transform(df[col_encode])
                    st.success(f"Applied Label Encoding to `{col_encode}`.")
            st.rerun()

def handle_feature_engineering():
    """Manages the UI for the Feature Engineering section and its sub-tabs."""
    st.subheader("Feature Engineering")
    st.markdown("`Please handle missing values in your columns before using Feature Engineering.`")
    
    data = st.session_state.processed_data
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Numerical Binning", "Combine Rare Categories", "Combine Categorical", 
        "Combine Numerical", "Polynomial Features", "Extract Datetime Parts"
    ])

    with tab1:
        _render_binning_tab(data, numeric_cols)
    with tab2:
        _render_combine_rare_tab(data, categorical_cols)
    with tab3:
        _render_combine_categorical_tab(data, categorical_cols)
    with tab4:
        _render_combine_numerical_tab(data, numeric_cols)
    with tab5:
        _render_polynomial_features_tab(data, numeric_cols)
    with tab6:
        _render_datetime_extraction_tab(data)

def display_cleaning():
    """The main function to display the entire data cleaning and transformation section."""
    st.subheader("Perform Data Cleaning and Transformations")

    with st.expander("Expand to Clean and Transform Data"):
        
        tabs_col, undo_col, redo_col = st.columns([8, 1, 1]) 

        with undo_col:
            st.write("") 
            if st.button("↩️", key="undo_button_icon", help="Undo last change", use_container_width=True):
                if st.session_state.data_history:
                    st.session_state.redo_history.append(st.session_state.processed_data.copy())
                    st.session_state.processed_data = st.session_state.data_history.pop()
                    st.success("Undo successful.")
                    st.rerun()
                else:
                    st.toast("Nothing to undo.")
        with redo_col:
            st.write("")
            if st.button("↪️", key="redo_button_icon", help="Redo last change", use_container_width=True):
                if st.session_state.redo_history:
                    st.session_state.data_history.append(st.session_state.processed_data.copy())
                    st.session_state.processed_data = st.session_state.redo_history.pop()
                    st.success("Redo successful.")
                    st.rerun()
                else:
                    st.toast("Nothing to redo.")

        with tabs_col:
            tab_rename, tab_dtype, tab_missing, tab_outlier, tab_encode, tab_feature_eng = st.tabs([
                "Rename/Delete Columns", 
                "Change Data Type", 
                "Missing Values",
                "Outlier Handling",
                "Categorical Encoding",
                "Feature Engineering"
            ])

            with tab_rename:
                handle_rename_delete()
            with tab_dtype:
                handle_dtype_change()
            with tab_missing:
                handle_missing_values()
            with tab_outlier:
                handle_outliers()
            with tab_encode:
                handle_categorical_encoding()
            with tab_feature_eng:
                handle_feature_engineering()
    
    visualize_missing_data(st.session_state.processed_data)

    _, col2 = st.columns([3, 1]) 
    with col2:
        csv = convert_df_to_csv(st.session_state.processed_data)
        st.download_button(
            label="📥 Download Data",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv',
            use_container_width=True
        )
    st.markdown("---")
