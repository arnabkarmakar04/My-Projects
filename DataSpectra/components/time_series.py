import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

def display_time_series_analysis(data):
    st.subheader("Time Series Analysis")

    datetime_cols = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

    if not datetime_cols or not numeric_cols:
        st.warning("Time Series Analysis requires at least one datetime column and one numeric column.")
        return

    with st.expander("Expand for Time Series Analysis"):
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox(
                "Select a datetime column for the index",
                options=["Select a column"] + datetime_cols,
                key="ts_date_col"
            )
        with col2:
            value_col = st.selectbox(
                "Select a numeric column to analyze",
                options=["Select a column"] + numeric_cols,
                key="ts_value_col"
            )

        if date_col != "Select a column" and value_col != "Select a column":
            ts_data = data[[date_col, value_col]].copy().dropna()
            ts_data = ts_data.set_index(date_col).sort_index()

            analysis_options = [
                "Time Series Plot",
                "Time-Based Aggregation",
                "Rolling Window Calculations",
                "Time Series Decomposition",
                "Forecasting"
            ]

            selected_analyses = st.multiselect(
                "Select the analyses to display:",
                options=analysis_options,
                default=[],
                key="ts_analysis_multiselect"
            )

            if "Time Series Plot" in selected_analyses:
                st.markdown("---")
                st.write("#### Time Series Plot")
                fig_ts = px.line(ts_data, x=ts_data.index, y=value_col, title=f'{value_col} over Time')
                st.plotly_chart(fig_ts, use_container_width=True)

            if "Time-Based Aggregation" in selected_analyses:
                st.markdown("---")
                st.write("#### Time-Based Aggregation")
                agg_col1, agg_col2 = st.columns(2)
                with agg_col1:
                    agg_freq = st.selectbox("Aggregation Frequency", ["None", "Daily (D)", "Weekly (W)", "Monthly (M)"], key="ts_agg_freq")
                with agg_col2:
                    agg_func = st.selectbox("Aggregation Function", ["mean", "sum", "median", "min", "max"], key="ts_agg_func")

                if agg_freq != "None":
                    try:
                        freq_map = {"Daily (D)": "D", "Weekly (W)": "W", "Monthly (M)": "M"}
                        agg_data = ts_data[value_col].resample(freq_map[agg_freq]).agg(agg_func).reset_index()
                        
                        rows_to_show = st.slider(
                            "Number of aggregated rows to display",
                            min_value=1,
                            max_value=len(agg_data),
                            value=min(5, len(agg_data)),
                            key="agg_rows_slider"
                        )
                        st.dataframe(agg_data.head(rows_to_show))
                        
                        fig_agg = px.line(agg_data, x=date_col, y=value_col, title=f'Aggregated {value_col} ({agg_freq})')
                        st.plotly_chart(fig_agg, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not perform aggregation: {e}")

            if "Rolling Window Calculations" in selected_analyses:
                st.markdown("---")
                st.write("#### Rolling Window Calculations")
                window_size = st.slider("Select Window Size", min_value=1, max_value=100, value=7, key="ts_window_size")
                
                rolling_mean = ts_data[value_col].rolling(window=window_size).mean()
                rolling_std = ts_data[value_col].rolling(window=window_size).std()
                
                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(x=ts_data.index, y=ts_data[value_col], mode='lines', name='Original'))
                fig_rolling.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode='lines', name=f'{window_size}-Day Rolling Mean'))
                fig_rolling.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, mode='lines', name=f'{window_size}-Day Rolling Std Dev'))
                fig_rolling.update_layout(title="Rolling Window Statistics")
                st.plotly_chart(fig_rolling, use_container_width=True)

            if "Time Series Decomposition" in selected_analyses:
                st.markdown("---")
                st.write("#### Time Series Decomposition")
                try:
                    decomposition = seasonal_decompose(ts_data[value_col].dropna(), model='additive', period=max(1, min(12, len(ts_data)//2)))
                    fig_decompose = go.Figure()
                    fig_decompose.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend, mode='lines', name='Trend'))
                    fig_decompose.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal, mode='lines', name='Seasonality'))
                    fig_decompose.add_trace(go.Scatter(x=ts_data.index, y=decomposition.resid, mode='lines', name='Residuals'))
                    fig_decompose.update_layout(title="Time Series Decomposition")
                    st.plotly_chart(fig_decompose, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not perform decomposition: {e}")

            if "Forecasting" in selected_analyses:
                st.markdown("---")
                st.write("#### Forecasting")
                model_choice = st.selectbox("Select Forecasting Model", ["None", "ARIMA", "Prophet"], key="ts_model_choice")
                
                if model_choice != "None":
                    periods = st.number_input("Number of periods to forecast", min_value=1, max_value=365, value=30)
                    if st.button("Generate Forecast"):
                        with st.spinner("Generating forecast..."):
                            try:
                                if model_choice == "ARIMA":
                                    model = ARIMA(ts_data[value_col], order=(5,1,0))
                                    model_fit = model.fit()
                                    forecast = model_fit.forecast(steps=periods)
                                    forecast_index = pd.to_datetime(pd.date_range(start=ts_data.index[-1], periods=periods + 1, freq='D')[1:])
                                    
                                    fig_forecast = go.Figure()
                                    fig_forecast.add_trace(go.Scatter(x=ts_data.index, y=ts_data[value_col], mode='lines', name='Observed'))
                                    fig_forecast.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast'))
                                    fig_forecast.update_layout(title="ARIMA Forecast")
                                    st.plotly_chart(fig_forecast, use_container_width=True)

                                elif model_choice == "Prophet":
                                    prophet_df = ts_data.reset_index()
                                    prophet_df.columns = ['ds', 'y']
                                    
                                    model = Prophet()
                                    model.fit(prophet_df)
                                    future = model.make_future_dataframe(periods=periods)
                                    forecast = model.predict(future)
                                    
                                    fig_forecast = go.Figure()
                                    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
                                    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='lightblue', name='Confidence Interval'))
                                    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line_color='lightblue', name='Confidence Interval'))
                                    fig_forecast.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='markers', name='Observed'))
                                    fig_forecast.update_layout(title="Prophet Forecast")
                                    st.plotly_chart(fig_forecast, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not generate forecast: {e}")
