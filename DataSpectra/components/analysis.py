import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
    roc_curve, auc, silhouette_score
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import warnings
from sklearn.exceptions import ConvergenceWarning

def display_advanced_analytics(data):
    st.subheader("Advanced Analytical Features")
    st.markdown("📌 Please discard all selections from **Local Filters** before Analysis ")

    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    with st.expander("Select Analysis Methods & Evaluation Strategy"):
        use_cv = st.checkbox(
            "Use Cross-Validation for Evaluation", 
            value=False
        )

        if use_cv:
            k_folds = st.slider(
                "Select Number of Folds (k)", 
                min_value=3, 
                max_value=10, 
                value=5, 
                step=1,
                help="Select the number of folds for cross-validation. The data will be split into 'k' parts."
            )
            test_size = None
        else:
            test_size = st.slider(
                "Select Test Set Size", 
                min_value=0.1, 
                max_value=0.4, 
                value=0.2, 
                step=0.05,
                help="Select the proportion of the dataset to hold out for testing the models."
            )
            k_folds = None
        st.markdown("📌 Don't select Primary Keys(eg. Customer_ID, Product Id etc..) as Feature")    

        analysis_methods = st.multiselect(
            "Choose analysis methods to perform:",
            [
                "Linear Regression", 
                "Logistic Regression", 
                "Random Forest Classification", 
                "Decision Tree Classifier",
                "Support Vector Machine (SVM)",
                "K-Nearest Neighbors (KNN)",
                "K-Means Clustering"
            ]
        )
        
        eval_params = {"use_cv": use_cv, "k_folds": k_folds, "test_size": test_size}

        if "Linear Regression" in analysis_methods:
            perform_linear_regression(data, numeric_columns, test_size)
        
        if "Logistic Regression" in analysis_methods:
            perform_logistic_regression(data, numeric_columns, categorical_columns, **eval_params)

        if "Random Forest Classification" in analysis_methods:
            perform_random_forest(data, numeric_columns, categorical_columns, **eval_params)
        
        if "Decision Tree Classifier" in analysis_methods:
            perform_decision_tree_classification(data, numeric_columns, categorical_columns, **eval_params)

        if "Support Vector Machine (SVM)" in analysis_methods:
            perform_svm_classification(data, numeric_columns, categorical_columns, **eval_params)

        if "K-Nearest Neighbors (KNN)" in analysis_methods:
            perform_knn_classification(data, numeric_columns, categorical_columns, **eval_params)

        if "K-Means Clustering" in analysis_methods:
            perform_kmeans_clustering(data, numeric_columns)

def perform_linear_regression(data, numeric_columns, test_size):
    st.write("---")
    st.write("#### Linear Regression Report")

    if len(numeric_columns) < 2:
        st.warning("Linear Regression requires at least two numeric columns.")
        return

    y_col_selected = st.session_state.get("lr_y", "Select a column")
    x_cols_selected = st.session_state.get("lr_x", [])

    valid_y_options = ["Select a column"] + [col for col in numeric_columns if col not in x_cols_selected]
    valid_x_options = [col for col in numeric_columns if col != y_col_selected]

    if y_col_selected not in valid_y_options:
        y_col_selected = "Select a column"
    x_cols_selected = [col for col in x_cols_selected if col in valid_x_options]

    col1, col2 = st.columns(2)
    with col1:
        x_cols = st.multiselect(
            "Select Independent variable(s) (X)",
            options=valid_x_options,
            default=x_cols_selected,
            key="lr_x"
        )
    with col2:
        y_col = st.selectbox(
            "Select Dependent variable (Y)",
            options=valid_y_options,
            index=valid_y_options.index(y_col_selected),
            key="lr_y"
        )

    if x_cols and y_col != "Select a column":
        temp_data = data[x_cols + [y_col]]

        if temp_data.shape[0] < 10:
            st.error("Not enough data to perform regression. A minimum of 10 rows is required.")
            return

        try:
            X = temp_data[x_cols]
            y = temp_data[y_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("##### Model Performance on Test Set")
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("R-squared (R²)", f"{r2:.3f}")
            m_col2.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
            m_col3.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")

            equation_str = f"{model.intercept_:.4f}"
            for coef, name in zip(model.coef_, x_cols):
                equation_str += f" + {coef:.4f} * {name}" if coef >= 0 else f" - {abs(coef):.4f} * {name}"
            st.write(f"**Equation:** `{y_col} = {equation_str}`")

            st.markdown("---")

            if len(x_cols) == 1:
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.write("##### Regression Plot")
                    fig = px.scatter(temp_data, x=x_cols[0], y=y_col, trendline="ols", title=f'Trend: {y_col} vs {x_cols[0]}')
                    st.plotly_chart(fig, use_container_width=True)
                with plot_col2:
                    st.write("##### Residuals Plot")
                    residuals = y_test - y_pred
                    res_fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title="Residuals vs. Predicted Values")
                    res_fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(res_fig, use_container_width=True)
            else:
                st.write("##### Individual Variable Plots")
                cols = st.columns(2)
                for i, col_name in enumerate(x_cols):
                    with cols[i % 2]:
                        fig = px.scatter(temp_data, x=col_name, y=y_col, title=f'{y_col} vs {col_name}', trendline="ols", trendline_color_override="red")
                        st.plotly_chart(fig, use_container_width=True)

                st.write("##### Residuals Plot")
                residuals = y_test - y_pred
                res_fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title="Residuals vs. Predicted Values")
                res_fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(res_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during regression: {e}")


def perform_logistic_regression(data, numeric_columns, categorical_columns, use_cv, k_folds, test_size):
    st.write("---")
    st.write("#### Logistic Regression Report")

    if not categorical_columns:
        st.warning("Logistic Regression requires a categorical target variable.")
        return

    binary_targets = [col for col in categorical_columns if data[col].nunique() == 2]
    if not binary_targets:
        st.warning("Logistic Regression requires a binary target variable (exactly 2 unique classes).")
        return

    target_var = st.selectbox("Select Target Variable (Binary)", ["Select a column"] + binary_targets, key="log_reg_target")

    if target_var != "Select a column":
        feature_cols = st.multiselect(
            "Select Feature Columns (X)",
            options=[col for col in numeric_columns if col != target_var],
            key="log_reg_features"
        )

        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return
            
        max_iter_value = st.number_input(
            "Enter Max Iterations for the solver",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
            key="log_reg_max_iter",
            help="The maximum number of iterations for the model's solver to converge."
        )

        try:
            X = data[feature_cols]
            y = data[target_var]

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(random_state=42, class_weight='balanced', max_iter=max_iter_value))
            ])
            
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)

                if use_cv:
                    st.write(f"##### Model Performance using {k_folds}-Fold Cross-Validation")
                    with st.spinner("Running Cross-Validation... This may take a moment."):
                        acc_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='accuracy')
                        prec_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='precision_weighted')
                        rec_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='recall_weighted')
                        f1_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='f1_weighted')

                        y_pred = cross_val_predict(pipeline, X, y_encoded, cv=k_folds)
                        y_proba = cross_val_predict(pipeline, X, y_encoded, cv=k_folds, method='predict_proba')

                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Avg. Accuracy", f"{acc_scores.mean():.2%} (±{acc_scores.std()*2:.2%})")
                    m_col2.metric("Avg. Precision", f"{prec_scores.mean():.2%} (±{prec_scores.std()*2:.2%})")
                    m_col3.metric("Avg. Recall", f"{rec_scores.mean():.2%} (±{rec_scores.std()*2:.2%})")
                    m_col4.metric("Avg. F1-Score", f"{f1_scores.mean():.2%} (±{f1_scores.std()*2:.2%})")
                    
                    pipeline.fit(X, y_encoded)
                    y_test = y_encoded
                
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                    )
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    y_proba = pipeline.predict_proba(X_test)

                    st.write("##### Model Performance on Test Set")
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Accuracy", f"{acc:.2%}")
                    m_col2.metric("Precision", f"{prec:.2%}")
                    m_col3.metric("Recall", f"{rec:.2%}")
                    m_col4.metric("F1-Score", f"{f1:.2%}")

                for warning_message in caught_warnings:
                    if issubclass(warning_message.category, ConvergenceWarning):
                        st.warning(
                            "The model failed to converge within the specified number of iterations. "
                            "The results shown may be unreliable. Consider increasing the 'Max Iterations' value."
                        )
                        break

            intercept = pipeline.named_steps['model'].intercept_[0]
            coeffs = pipeline.named_steps['model'].coef_[0]
            equation_str = f"{intercept:.4f}"
            for coef, name in zip(coeffs, X.columns):
                equation_str += f" + {coef:.4f} * {name}" if coef >= 0 else f" - {abs(coef):.4f} * {name}"

            st.write(f"**Model Equation (Log-Odds):** `z = {equation_str}`")
            st.write("**Model Equation (Sigmoid):** `P(Y=1) = 1 / (1 + exp(-z))`")

            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Pred: {c}" for c in class_names],
                    y=[f"Actual: {c}" for c in class_names],
                    colorscale='Blues', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(
                            x=f"Pred: {class_names[j]}",
                            y=f"Actual: {class_names[i]}",
                            text=str(cm[i, j]),
                            showarrow=False,
                            font={"color": "white" if cm[i, j] < cm.max() / 1.5 else "black"}
                        )
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with plot_col2:
                st.write("##### Feature Coefficients")
                coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coeffs})
                coeff_df = coeff_df.sort_values(by="Coefficient", ascending=False)
                coeff_fig = px.bar(
                    coeff_df,
                    x="Feature",
                    y="Coefficient",
                    color="Feature",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title="Feature Influence on Outcome"
                )
                st.plotly_chart(coeff_fig, use_container_width=True)

            st.write("##### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_fig = go.Figure(data=go.Scatter(
                x=fpr, y=tpr, mode='lines',
                line=dict(color=px.colors.sequential.Blues[-1], width=2),
                name=f"AUC = {roc_auc:.2f}"
            ))
            roc_fig.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
            roc_fig.update_layout(
                title_text='Logistic Regression ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(roc_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

def perform_random_forest(data, numeric_columns, categorical_columns, use_cv, k_folds, test_size):
    st.write("---")
    st.write("#### Random Forest Classification Report")

    if not categorical_columns:
        st.warning("Random Forest Classification requires a categorical target variable.")
        return

    target_var = st.selectbox("Select Target Variable (Categorical)", ["Select a column"] + categorical_columns, key="rf_target")

    if target_var != "Select a column":
        feature_cols = st.multiselect(
            "Select Feature Columns (X)",
            options=[col for col in numeric_columns if col != target_var],
            key="rf_features"
        )
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return

        try:
            X = data[feature_cols]
            y = data[target_var]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_
            
            criterion = st.selectbox("Select Criterion", ['gini', 'entropy'], key="rf_criterion", help="The function to measure the quality of a split.")
            n_estimators = st.slider("Number of Trees in Forest", 10, 500, 100, 10, key="rf_n_estimators")
            
            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=42)
            
            if use_cv:
                st.write(f"##### Model Performance using {k_folds}-Fold Cross-Validation")
                with st.spinner("Running Cross-Validation... This may take a moment."):
                    acc_scores = cross_val_score(model, X, y_encoded, cv=k_folds, scoring='accuracy')
                    prec_scores = cross_val_score(model, X, y_encoded, cv=k_folds, scoring='precision_weighted')
                    rec_scores = cross_val_score(model, X, y_encoded, cv=k_folds, scoring='recall_weighted')
                    f1_scores = cross_val_score(model, X, y_encoded, cv=k_folds, scoring='f1_weighted')
                    
                    y_pred = cross_val_predict(model, X, y_encoded, cv=k_folds)
                    y_proba = cross_val_predict(model, X, y_encoded, cv=k_folds, method='predict_proba')

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Avg. Accuracy", f"{acc_scores.mean():.2%} (±{acc_scores.std()*2:.2%})")
                m_col2.metric("Avg. Precision", f"{prec_scores.mean():.2%} (±{prec_scores.std()*2:.2%})")
                m_col3.metric("Avg. Recall", f"{rec_scores.mean():.2%} (±{rec_scores.std()*2:.2%})")
                m_col4.metric("Avg. F1-Score", f"{f1_scores.mean():.2%} (±{f1_scores.std()*2:.2%})")
                
                model.fit(X, y_encoded)
                y_test = y_encoded
                y_test_bin = label_binarize(y_encoded, classes=list(range(len(class_names))))
            
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)

                st.write("##### Model Performance on Test Set")
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Accuracy", f"{acc:.2%}")
                m_col2.metric("Precision", f"{prec:.2%}")
                m_col3.metric("Recall", f"{rec:.2%}")
                m_col4.metric("F1-Score", f"{f1:.2%}")
                y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))

            st.write("**Core Concept:** `An ensemble of Decision Trees that votes on the final prediction.`")

            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Pred: {c}" for c in class_names],
                    y=[f"Actual: {c}" for c in class_names],
                    colorscale='Cividis', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(
                            x=f"Pred: {class_names[j]}",
                            y=f"Actual: {class_names[i]}",
                            text=str(cm[i, j]),
                            showarrow=False,
                            font={"color": "white" if cm[i, j] < cm.max() / 1.5 else "black"}
                        )
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)

            with plot_col2:
                st.write("##### Feature Importance")
                importances = model.feature_importances_
                importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=False)
                imp_fig = px.bar(
                    importance_df,
                    x="Feature",
                    y="Importance",
                    color="Feature",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    title="Feature Importances"
                )
                st.plotly_chart(imp_fig, use_container_width=True)

            st.write("##### ROC Curve")
            if len(class_names) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                roc_fig = go.Figure(data=go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    line=dict(color=getattr(px.colors.sequential, 'Cividis')[-1], width=2),
                    name=f"AUC = {roc_auc:.2f}"
                ))
            else:
                fpr, tpr, roc_auc = dict(), dict(), dict()
                for i in range(len(class_names)):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                roc_fig = go.Figure()
                colors = px.colors.qualitative.Set1
                for i, class_name in enumerate(class_names):
                    roc_fig.add_trace(go.Scatter(
                        x=fpr[i], y=tpr[i], mode='lines',
                        name=f"Class {class_name} (AUC = {roc_auc[i]:.2f})",
                        line=dict(color=colors[i % len(colors)])
                    ))

            roc_fig.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
            roc_fig.update_layout(
                title_text='Random Forest ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(roc_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

def perform_decision_tree_classification(data, numeric_columns, categorical_columns, use_cv, k_folds, test_size): 
    st.write("---")
    st.write("#### Decision Tree Classification Report")

    if not categorical_columns:
        st.warning("Decision Tree Classification requires a categorical target variable.")
        return

    target_var = st.selectbox("Select Target Variable (Categorical)", ["Select a column"] + categorical_columns, key="dt_target")

    if target_var != "Select a column":
        feature_cols = st.multiselect(
            "Select Feature Columns (X)",
            options=[col for col in numeric_columns if col != target_var],
            key="dt_features"
        )
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return

        try:
            X = data[feature_cols]
            y = data[target_var]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_.astype(str)

            criterion = st.selectbox("Select Criterion", ['gini', 'entropy'], key="dt_criterion", help="The function to measure the quality of a split.")
            max_depth = st.slider("Select Max Depth for the Tree", 2, 20, 5, key="dt_max_depth")
            
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)

            if use_cv:
                st.write(f"##### Model Performance using {k_folds}-Fold Cross-Validation")
                with st.spinner("Running Cross-Validation..."):
                    acc_scores = cross_val_score(model, X, y_encoded, cv=k_folds, scoring='accuracy')
                    prec_scores = cross_val_score(model, X, y_encoded, cv=k_folds, scoring='precision_weighted')
                    rec_scores = cross_val_score(model, X, y_encoded, cv=k_folds, scoring='recall_weighted')
                    f1_scores = cross_val_score(model, X, y_encoded, cv=k_folds, scoring='f1_weighted')
                    
                    y_pred = cross_val_predict(model, X, y_encoded, cv=k_folds)
                    y_proba = cross_val_predict(model, X, y_encoded, cv=k_folds, method='predict_proba')

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Avg. Accuracy", f"{acc_scores.mean():.2%} (±{acc_scores.std()*2:.2%})")
                m_col2.metric("Avg. Precision", f"{prec_scores.mean():.2%} (±{prec_scores.std()*2:.2%})")
                m_col3.metric("Avg. Recall", f"{rec_scores.mean():.2%} (±{rec_scores.std()*2:.2%})")
                m_col4.metric("Avg. F1-Score", f"{f1_scores.mean():.2%} (±{f1_scores.std()*2:.2%})")
                
                model.fit(X, y_encoded)
                y_test = y_encoded
                y_test_bin = label_binarize(y_encoded, classes=list(range(len(class_names))))

            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)

                st.write("##### Model Performance on Test Set")
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Accuracy", f"{acc:.2%}")
                m_col2.metric("Precision", f"{prec:.2%}")
                m_col3.metric("Recall", f"{rec:.2%}")
                m_col4.metric("F1-Score", f"{f1:.2%}")
                y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))

            st.write("**Core Concept:** `A tree-like model of decisions based on feature values.`")

            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Pred: {c}" for c in class_names],
                    y=[f"Actual: {c}" for c in class_names],
                    colorscale='Greens', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(
                            x=f"Pred: {class_names[j]}",
                            y=f"Actual: {class_names[i]}",
                            text=str(cm[i, j]),
                            showarrow=False,
                            font={"color": "white" if cm[i, j] < cm.max() / 1.5 else "black"}
                        )
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with plot_col2:
                st.write("##### Feature Importance")
                importances = model.feature_importances_
                importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=False)
                imp_fig = px.bar(
                    importance_df,
                    x="Feature",
                    y="Importance",
                    color="Feature",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    title="Feature Importances"
                )
                st.plotly_chart(imp_fig, use_container_width=True)

            st.write("##### ROC Curve")
            if len(class_names) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                roc_fig = go.Figure(data=go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    line=dict(color=getattr(px.colors.sequential, 'Greens')[-1], width=2),
                    name=f"AUC = {roc_auc:.2f}"
                ))
            else:
                fpr, tpr, roc_auc = dict(), dict(), dict()
                for i in range(len(class_names)):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                roc_fig = go.Figure()
                colors = px.colors.qualitative.Set1
                for i, class_name in enumerate(class_names):
                    roc_fig.add_trace(go.Scatter(
                        x=fpr[i], y=tpr[i], mode='lines',
                        name=f"Class {class_name} (AUC = {roc_auc[i]:.2f})",
                        line=dict(color=colors[i % len(colors)])
                    ))

            roc_fig.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
            roc_fig.update_layout(
                title_text='Decision Tree ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(roc_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

def perform_svm_classification(data, numeric_columns, categorical_columns, use_cv, k_folds, test_size):
    st.write("---")
    st.write("#### Support Vector Machine (SVM) Report")

    if not categorical_columns:
        st.warning("SVM requires a categorical target variable.")
        return

    target_var = st.selectbox("Select Target Variable (Categorical)", ["Select a column"] + categorical_columns, key="svm_target")

    if target_var != "Select a column":
        feature_cols = st.multiselect(
            "Select Feature Columns (X)",
            options=[col for col in numeric_columns if col != target_var],
            key="svm_features"
        )
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return

        try:
            X = data[feature_cols]
            y = data[target_var]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_

            C = st.slider("Select Regularization Parameter (C)", 0.1, 10.0, 1.0, 0.1, key="svm_c")
            kernel = st.selectbox("Select Kernel Type", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(C=C, kernel=kernel, probability=True, random_state=42))
            ])

            if use_cv:
                st.write(f"##### Model Performance using {k_folds}-Fold Cross-Validation")
                with st.spinner("Running Cross-Validation... This may take a moment."):
                    acc_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='accuracy')
                    prec_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='precision_weighted')
                    rec_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='recall_weighted')
                    f1_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='f1_weighted')
                    
                    y_pred = cross_val_predict(pipeline, X, y_encoded, cv=k_folds)
                
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Avg. Accuracy", f"{acc_scores.mean():.2%} (±{acc_scores.std()*2:.2%})")
                m_col2.metric("Avg. Precision", f"{prec_scores.mean():.2%} (±{prec_scores.std()*2:.2%})")
                m_col3.metric("Avg. Recall", f"{rec_scores.mean():.2%} (±{rec_scores.std()*2:.2%})")
                m_col4.metric("Avg. F1-Score", f"{f1_scores.mean():.2%} (±{f1_scores.std()*2:.2%})")

                y_test = y_encoded
                y_test_bin = label_binarize(y_encoded, classes=list(range(len(class_names))))
                
                if len(class_names) > 2:
                    y_proba = cross_val_predict(pipeline, X, y_encoded, cv=k_folds, method='predict_proba')
                else:
                    pipeline.fit(X, y_encoded)
                    y_proba = pipeline.predict_proba(X)
            
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                st.write("##### Model Performance on Test Set")
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Accuracy", f"{acc:.2%}")
                m_col2.metric("Precision", f"{prec:.2%}")
                m_col3.metric("Recall", f"{rec:.2%}")
                m_col4.metric("F1-Score", f"{f1:.2%}")
                y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))

            st.write("**Model Equation (Hyperplane):** `w * x - b = 0`")
            st.write(f"**Kernel Trick Used:** `{kernel}`")

            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Pred: {c}" for c in class_names],
                    y=[f"Actual: {c}" for c in class_names],
                    colorscale='Purples', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(
                            x=f"Pred: {class_names[j]}",
                            y=f"Actual: {class_names[i]}",
                            text=str(cm[i, j]),
                            showarrow=False,
                            font={"color": "white" if cm[i, j] < cm.max() / 1.5 else "black"}
                        )
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)

            with plot_col2:
                st.write("##### ROC Curve")
                if not use_cv:
                     X_scaled = pipeline.named_steps['scaler'].transform(X_test)
                     y_proba = pipeline.named_steps['model'].predict_proba(X_scaled)

                if len(class_names) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    roc_fig = go.Figure(data=go.Scatter(
                        x=fpr, y=tpr, mode='lines',
                        line=dict(color='purple', width=2),
                        name=f"AUC = {roc_auc:.2f}"
                    ))
                else:
                    fpr, tpr, roc_auc = dict(), dict(), dict()
                    for i in range(len(class_names)):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    roc_fig = go.Figure()
                    colors = px.colors.qualitative.Set1
                    for i, class_name in enumerate(class_names):
                        roc_fig.add_trace(go.Scatter(
                            x=fpr[i], y=tpr[i], mode='lines',
                            name=f"Class {class_name} (AUC = {roc_auc[i]:.2f})",
                            line=dict(color=colors[i % len(colors)])
                        ))

                roc_fig.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
                roc_fig.update_layout(
                    title_text='SVM ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(roc_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
            
def perform_knn_classification(data, numeric_columns, categorical_columns, use_cv, k_folds, test_size):
    st.write("---")
    st.write("#### K-Nearest Neighbors (KNN) Report")

    if not categorical_columns:
        st.warning("KNN requires a categorical target variable.")
        return

    target_var = st.selectbox("Select Target Variable (Categorical)", ["Select a column"] + categorical_columns, key="knn_target")

    if target_var != "Select a column":
        feature_cols = st.multiselect(
            "Select Feature Columns (X)",
            options=[col for col in numeric_columns if col != target_var],
            key="knn_features"
        )
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return

        try:
            X = data[feature_cols]
            y = data[target_var]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_

            k_neighbors = st.slider("Select Number of Neighbors (k)", 1, 15, 5, key="knn_k")
            metric = st.selectbox("Select Distance Metric", ['euclidean', 'manhattan'], key="knn_metric", help="The distance metric to use for the tree.")
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier(n_neighbors=k_neighbors, metric=metric))
            ])

            if use_cv:
                st.write(f"##### Model Performance using {k_folds}-Fold Cross-Validation")
                with st.spinner("Running Cross-Validation..."):
                    acc_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='accuracy')
                    prec_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='precision_weighted')
                    rec_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='recall_weighted')
                    f1_scores = cross_val_score(pipeline, X, y_encoded, cv=k_folds, scoring='f1_weighted')
                    
                    y_pred = cross_val_predict(pipeline, X, y_encoded, cv=k_folds)
                    y_proba = cross_val_predict(pipeline, X, y_encoded, cv=k_folds, method='predict_proba')

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Avg. Accuracy", f"{acc_scores.mean():.2%} (±{acc_scores.std()*2:.2%})")
                m_col2.metric("Avg. Precision", f"{prec_scores.mean():.2%} (±{prec_scores.std()*2:.2%})")
                m_col3.metric("Avg. Recall", f"{rec_scores.mean():.2%} (±{rec_scores.std()*2:.2%})")
                m_col4.metric("Avg. F1-Score", f"{f1_scores.mean():.2%} (±{f1_scores.std()*2:.2%})")
                y_test = y_encoded
                y_test_bin = label_binarize(y_encoded, classes=list(range(len(class_names))))
            
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)

                st.write("##### Model Performance on Test Set")
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Accuracy", f"{acc:.2%}")
                m_col2.metric("Precision", f"{prec:.2%}")
                m_col3.metric("Recall", f"{rec:.2%}")
                m_col4.metric("F1-Score", f"{f1:.2%}")
                y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))

            st.write(f"**Core Concept ({metric.capitalize()} Distance):** `d(p, q) = ...`")

            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Pred: {c}" for c in class_names],
                    y=[f"Actual: {c}" for c in class_names],
                    colorscale='Oranges', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(
                            x=f"Pred: {class_names[j]}",
                            y=f"Actual: {class_names[i]}",
                            text=str(cm[i, j]),
                            showarrow=False,
                            font={"color": "white" if cm[i, j] < cm.max() / 1.5 else "black"}
                        )
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)

            with plot_col2:
                st.write("##### ROC Curve")
                if len(class_names) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    roc_fig = go.Figure(data=go.Scatter(
                        x=fpr, y=tpr, mode='lines',
                        line=dict(color='orange', width=2),
                        name=f'AUC = {roc_auc:.2f}'
                    ))
                else:
                    fpr, tpr, roc_auc = dict(), dict(), dict()
                    for i in range(len(class_names)):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    roc_fig = go.Figure()
                    colors = px.colors.qualitative.Set1
                    for i, class_name in enumerate(class_names):
                        roc_fig.add_trace(go.Scatter(
                            x=fpr[i], y=tpr[i], mode='lines',
                            name=f"Class {class_name} (AUC = {roc_auc[i]:.2f})",
                            line=dict(color=colors[i % len(colors)])
                        ))

                roc_fig.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
                roc_fig.update_layout(
                    title_text='KNN ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(roc_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

def perform_kmeans_clustering(data, numeric_columns):
    st.write("---")
    st.write("#### K-Means Clustering Report")

    if len(numeric_columns) < 2:
        st.warning("K-Means Clustering requires at least two numeric columns.")
        return

    kmeans_cols = st.multiselect("Select at least two columns for Clustering", numeric_columns, key="kmeans_cols")

    if len(kmeans_cols) >= 2:
        clustering_data = data[kmeans_cols]

        if clustering_data.shape[0] < 10:
            st.error("Not enough data for clustering. At least 10 rows are required.")
            return
        
        num_clusters = st.number_input("Enter Number of Clusters (k)", min_value=2, max_value=20, value=3, step=1, key="num_clusters")
        
        if clustering_data.shape[0] < num_clusters:
            st.error(f"Not enough data for the selected k. You need at least {num_clusters} rows.")
            return

        st.write("**Objective Function (WCSS):** `minimize sum(||x - µᵢ||²)`")

        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.write("##### Elbow Method for Optimal k")
            with st.spinner("Calculating optimal k..."):
                inertia = []
                k_range = range(1, 11)
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    kmeans.fit(clustering_data)
                    inertia.append(kmeans.inertia_)

                elbow_fig = go.Figure(data=go.Scatter(
                    x=list(k_range), y=inertia,
                    mode='lines+markers',
                    line=dict(color='#ff6347')
                ))
                elbow_fig.update_layout(
                    title="Inertia vs. Number of Clusters",
                    xaxis_title="Number of Clusters (k)",
                    yaxis_title="Inertia (WCSS)",
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(elbow_fig, use_container_width=True)
            
            try:
                kmeans_sil = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                labels_sil = kmeans_sil.fit_predict(clustering_data)
                silhouette = silhouette_score(clustering_data, labels_sil)
                st.metric(f"Silhouette Score (k={num_clusters})", f"{silhouette:.3f}")
            except Exception:
                st.warning("Silhouette Score could not be computed.")

        with v_col2:
            st.write("##### Cluster Visualization")
            try:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(clustering_data)
                centroids = kmeans.cluster_centers_

                plot_data = clustering_data.copy()
                plot_data["Cluster"] = labels.astype(str)

                cluster_fig = px.scatter(
                    plot_data, x=kmeans_cols[0], y=kmeans_cols[1],
                    color="Cluster",
                    title=f"K-Means Clustering (k={num_clusters})",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    opacity=0.85
                )

                cluster_fig.add_trace(go.Scatter(
                    x=centroids[:, 0], y=centroids[:, 1],
                    mode='markers',
                    marker=dict(color='black', size=12, symbol='x', line=dict(width=2)),
                    name='Centroids'
                ))

                cluster_fig.update_layout(margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(cluster_fig, use_container_width=True)

                st.write("##### Cluster Size Distribution")
                cluster_sizes = pd.Series(labels).value_counts().sort_index()
                dist_str = ", ".join([f"Cluster {i}: {count}" for i, count in cluster_sizes.items()])
                st.write(f"`{dist_str}`")

            except Exception as e:
                st.error(f"An error occurred during clustering: {e}")
