import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.frontend.utils.api_client import BackendAPIClient

api_client = BackendAPIClient()

st.set_page_config(page_title="Dataset Manager - ML Orchestrator", page_icon="💾", layout="wide")

# Inject Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0c10;
    }
    .metric-box {
        background: rgba(31, 40, 56, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #66fcf1;
    }
    .metric-text {
        font-size: 0.85rem;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-title {
        color: #66fcf1;
        border-left: 4px solid #45a29e;
        padding-left: 10px;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("💾 Dataset Management")
st.write("Upload, preview, profile, and visualize CSV files interactively.")

# Layout: File Upload and Selector
col_up, col_sel = st.columns([1, 1])

with col_up:
    st.markdown('<div class="section-title">📤 Upload New CSV File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file to import", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Upload and Process"):
            with st.spinner("Processing CSV file and extracting statistical profiles..."):
                file_bytes = uploaded_file.read()
                res = api_client.upload_dataset(uploaded_file.name, file_bytes)
                
                if res:
                    st.success(f"🎉 Success! Dataset '{res.get('filename')}' uploaded and analyzed.")
                    st.rerun()
                else:
                    st.error("❌ Failed to upload dataset. Check backend connection.")

# Fetch datasets
datasets = api_client.get_datasets()

with col_sel:
    st.markdown('<div class="section-title">📂 Select Active Dataset</div>', unsafe_allow_html=True)
    if not datasets:
        st.info("No datasets currently uploaded. Please upload a CSV file above.")
        selected_dataset_id = None
    else:
        dataset_options = {d["id"]: f"{d['filename']} ({d['rows']} rows)" for d in datasets}
        
        # Check if active dataset selection is cached in session state
        default_sel_idx = 0
        if "active_dataset_id" in st.session_state and st.session_state["active_dataset_id"] in dataset_options:
            default_sel_idx = list(dataset_options.keys()).index(st.session_state["active_dataset_id"])
            
        selected_dataset_id = st.selectbox(
            "Select dataset to profile and visualize:",
            options=list(dataset_options.keys()),
            index=default_sel_idx,
            format_func=lambda x: dataset_options[x]
        )

st.markdown("---")

if selected_dataset_id:
    # Retrieve metadata
    meta = api_client.get_dataset(selected_dataset_id)
    
    if meta:
        # 1. Summary Cards
        card1, card2, card3, card4, card5 = st.columns(5)
        with card1:
            st.markdown(f'<div class="metric-box"><div class="metric-number">{meta["rows"]}</div><div class="metric-text">Total Rows</div></div>', unsafe_allow_html=True)
        with card2:
            st.markdown(f'<div class="metric-box"><div class="metric-number">{meta["columns_count"]}</div><div class="metric-text">Columns</div></div>', unsafe_allow_html=True)
        with card3:
            st.markdown(f'<div class="metric-box"><div class="metric-number">{len(meta["numerical_columns"])}</div><div class="metric-text">Numerical</div></div>', unsafe_allow_html=True)
        with card4:
            st.markdown(f'<div class="metric-box"><div class="metric-number">{len(meta["categorical_columns"])}</div><div class="metric-text">Categorical</div></div>', unsafe_allow_html=True)
        with card5:
            st.markdown(f'<div class="metric-box"><div class="metric-number">{meta["duplicate_rows"]}</div><div class="metric-text">Duplicates</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">🔍 Data Profiling & Statistics</div>', unsafe_allow_html=True)
        
        stat_tab1, stat_tab2, stat_tab3, stat_tab4, stat_tab5 = st.tabs([
            "📄 Row Preview", 
            "📋 Column Information", 
            "📈 Descriptive Statistics", 
            "🤖 AI Dataset Analyst", 
            "🛠️ Data Cleaning Agent"
        ])
        
        # Load dataset rows for preview and visualization
        # Request up to 2000 rows for performant frontend loading
        with st.spinner("Fetching dataset data rows..."):
            preview_data = api_client.get_dataset_preview(selected_dataset_id, limit=2000)
            
        if preview_data and preview_data.get("data"):
            df = pd.DataFrame(preview_data["data"])
        else:
            df = pd.DataFrame()
            st.error("Failed to load dataset rows from backend.")
            
        with stat_tab1:
            if not df.empty:
                st.write("**First 15 Rows Preview:**")
                st.dataframe(df.head(15), use_container_width=True)
            else:
                st.info("No row data to display.")
                
        with stat_tab2:
            col_info_list = []
            for col in meta["columns"]:
                col_info_list.append({
                    "Column Header": col["name"],
                    "Data Type": col["data_type"],
                    "Missing Values": col["null_count"],
                    "Missing %": f"{col['null_percentage']:.2f}%",
                    "Numerical": "✅" if col["is_numerical"] else "❌",
                    "Categorical": "✅" if col["is_categorical"] else "❌"
                })
            st.dataframe(pd.DataFrame(col_info_list), use_container_width=True, hide_index=True)
            
        with stat_tab3:
            # Display descriptive stats
            stats_df = pd.DataFrame(meta["summary_stats"])
            st.dataframe(stats_df, use_container_width=True)

        with stat_tab4:
            st.subheader("🤖 LangChain AI Dataset Analysis Agent")
            st.write("Trigger the analysis agent to inspect columns, infer ML targets, and discover outliers and preprocessing steps.")
            
            # Setup session state key for caching analysis reports
            state_key = f"ai_analysis_{selected_dataset_id}"
            
            if st.button("Run AI Dataset Analysis", type="primary", key="trigger_analysis"):
                with st.spinner("AI Dataset Agent is analyzing metadata and statistics..."):
                    analysis_res = api_client.analyze_dataset(selected_dataset_id)
                    if analysis_res:
                        st.session_state[state_key] = analysis_res
                        st.success("Analysis report completed successfully!")
                    else:
                        st.error("Failed to get response from AI Agent backend.")
                        
            if state_key in st.session_state:
                report = st.session_state[state_key]
                
                st.markdown("---")
                st.markdown(f"### 📋 Dataset Summary Report")
                st.info(report.get("dataset_summary", "No summary provided."))
                
                # Metric grid
                col_tr, col_pr = st.columns(2)
                with col_tr:
                    st.markdown(f"""
                    <div style="background:rgba(102,252,241,0.05); border:1px solid rgba(102,252,241,0.15); padding:1rem; border-radius:8px;">
                        <span style="font-size:0.85rem; color:#8b9bb4; text-transform:uppercase;">Suggested Target Variable</span>
                        <div style="font-size:1.4rem; font-weight:800; color:#66fcf1; margin-top:0.3rem;">{report.get("target_column_suggestion")}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_pr:
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); padding:1rem; border-radius:8px;">
                        <span style="font-size:0.85rem; color:#8b9bb4; text-transform:uppercase;">ML Task inferred</span>
                        <div style="font-size:1.4rem; font-weight:800; color:#c5c6c7; margin-top:0.3rem;">{report.get("ml_problem_type")}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.write("")
                st.markdown(f"**Target Column Rationale:**\n{report.get('target_rationale')}")
                
                st.markdown("#### 🚨 Anomalies & Health Assessment")
                anomaly_col1, anomaly_col2 = st.columns(2)
                with anomaly_col1:
                    st.markdown(f"**Missing Values:**\n{report.get('missing_values_assessment')}")
                    st.write("")
                    st.markdown(f"**Duplicates Check:**\n{report.get('duplicates_assessment')}")
                with anomaly_col2:
                    st.markdown(f"**Outlier Analysis:**\n{report.get('outliers_assessment')}")
                    st.write("")
                    st.markdown(f"**Class Imbalance:**\n{report.get('class_imbalance_assessment') or 'No class imbalance issues detected.'}")
                
                st.markdown("#### 🛠️ Recommended Preprocessing Steps Checklist")
                for i, rec in enumerate(report.get("preprocessing_recommendations", [])):
                    st.checkbox(rec, value=True, key=f"rec_check_{selected_dataset_id}_{i}")
                    
                with st.expander("🔍 Detailed Feature-by-Feature Column Profiles"):
                    col_analyses_data = []
                    for c_an in report.get("column_analyses", []):
                        col_analyses_data.append({
                            "Column Name": c_an.get("name"),
                            "Feature Type": c_an.get("inferred_type"),
                            "Semantics / Description": c_an.get("description"),
                            "Detected Anomalies": ", ".join(c_an.get("anomalies_detected", [])) or "None"
                        })
                    st.dataframe(pd.DataFrame(col_analyses_data), use_container_width=True, hide_index=True)

        with stat_tab5:
            st.subheader("🛠️ Data Cleaning Agent Pipeline")
            st.write("Clean the dataset by removing duplicate rows, imputing missing columns, standardizing numeric fields, and encoding categories.")
            
            clean_form_key = f"clean_form_{selected_dataset_id}"
            ai_rec_key = f"ai_analysis_{selected_dataset_id}"
            has_ai_recs = ai_rec_key in st.session_state
            
            if has_ai_recs:
                st.success("💡 Found active AI recommendations! Form controls below are pre-populated with suggestions.")
                ai_report = st.session_state[ai_rec_key]
                ai_recs_text = " ".join(ai_report.get("preprocessing_recommendations", [])).lower()
            else:
                st.info("ℹ️ Run the 'AI Dataset Analyst' first to auto-populate recommended parameters, or configure manually below.")
                ai_recs_text = ""

            with st.form(clean_form_key):
                st.markdown("### 🎛️ Preprocessing Configuration")
                
                remove_dups = st.checkbox("Remove exact duplicate rows", value=True)
                
                # Missing Value Imputation
                st.markdown("#### 🩹 Missing Values Imputation")
                col_nulls = [c for c in meta["columns"] if c["null_count"] > 0]
                imputation_strategies = {}
                
                if not col_nulls:
                    st.write("No missing values detected in dataset.")
                else:
                    for c in col_nulls:
                        cname = c["name"]
                        default_index = 0  # "none"
                        
                        # Match recommendations keyword to set default index
                        if "median" in ai_recs_text and cname.lower() in ai_recs_text:
                            default_index = 2  # median
                        elif "mean" in ai_recs_text and cname.lower() in ai_recs_text:
                            default_index = 1  # mean
                        elif "mode" in ai_recs_text and cname.lower() in ai_recs_text:
                            default_index = 3  # mode
                        elif "drop" in ai_recs_text and cname.lower() in ai_recs_text:
                            default_index = 5  # drop
                            
                        imputation_strategies[cname] = st.selectbox(
                            f"Imputation for '{cname}' ({c['null_count']} missing):",
                            options=["none", "mean", "median", "mode", "constant", "drop"],
                            index=default_index,
                            key=f"imp_sel_{cname}"
                        )
                
                # Numerical attributes scaling
                st.markdown("#### ⚖️ Numerical Columns Scaling")
                num_cols = meta["numerical_columns"]
                scaling_strategies = {}
                
                if not num_cols:
                    st.write("No numerical columns available.")
                else:
                    scale_all_option = st.selectbox(
                        "Quick scaler preset for ALL numerical columns:",
                        ["none", "standard", "minmax", "robust"],
                        index=0
                    )
                    
                    with st.expander("Configure scaling per numerical column"):
                        for col in num_cols:
                            default_index = 0
                            if scale_all_option != "none":
                                default_index = ["none", "standard", "minmax", "robust"].index(scale_all_option)
                            elif "scale" in ai_recs_text or "standardize" in ai_recs_text:
                                default_index = 1  # standard scale
                                
                            scaling_strategies[col] = st.selectbox(
                                f"Scaler for '{col}'",
                                options=["none", "standard", "minmax", "robust"],
                                index=default_index,
                                key=f"scale_sel_{col}"
                            )
                
                # Categorical column encoding
                st.markdown("#### 🔠 Categorical Column Encoding")
                cat_cols = meta["categorical_columns"]
                encoding_strategies = {}
                
                if not cat_cols:
                    st.write("No categorical columns available.")
                else:
                    for col in cat_cols:
                        default_index = 0
                        
                        if "one-hot" in ai_recs_text and col.lower() in ai_recs_text:
                            default_index = 1  # one_hot
                        elif "label" in ai_recs_text and col.lower() in ai_recs_text:
                            default_index = 2  # label
                        elif "drop" in ai_recs_text and col.lower() in ai_recs_text:
                            default_index = 3  # drop
                            
                        encoding_strategies[col] = st.selectbox(
                            f"Encoder for '{col}'",
                            options=["none", "one_hot", "label", "drop"],
                            index=default_index,
                            key=f"encode_sel_{col}"
                        )
                
                submit_clean = st.form_submit_button("Execute Preprocessing Pipeline", type="primary")

            if submit_clean:
                payload = {
                    "remove_duplicates": remove_dups,
                    "imputation_strategies": imputation_strategies,
                    "scaling_strategies": scaling_strategies,
                    "encoding_strategies": encoding_strategies
                }
                
                with st.spinner("Processing dataset transformations..."):
                    clean_res = api_client.clean_dataset(selected_dataset_id, payload)
                    if clean_res and clean_res.get("success"):
                        st.session_state[f"clean_result_{selected_dataset_id}"] = clean_res
                        st.success("🎉 Preprocessing successfully executed! Metrics report loaded below.")
                    else:
                        st.error("Failed to run preprocessing. Check backend metrics.")
            
            clean_res_key = f"clean_result_{selected_dataset_id}"
            if clean_res_key in st.session_state:
                res = st.session_state[clean_res_key]
                comp = res["comparison"]
                
                st.markdown("---")
                st.subheader("📊 Preprocessing Pipeline Comparison Report")
                
                # Render comparative KPIs
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                
                def render_comp_card(label, original, cleaned):
                    change_color = "#00ff88" if cleaned <= original else "#ff3344"
                    if label == "Feature count":
                        change_color = "#66fcf1"
                    st.markdown(f"""
                    <div style="background:rgba(31, 40, 56, 0.4); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:1rem; text-align:center;">
                        <span style="font-size:0.8rem; color:#8b9bb4; text-transform:uppercase;">{label}</span>
                        <div style="font-size:1.8rem; font-weight:800; color:#c5c6c7; margin-top:0.3rem;">
                            <span style="color:#8b9bb4; font-size:1.2rem;">{original}</span> ➔ <span style="color:{change_color}">{cleaned}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c1:
                    render_comp_card("Row Count", comp["original_rows"], comp["cleaned_rows"])
                with col_c2:
                    render_comp_card("Feature count", comp["original_cols"], comp["cleaned_cols"])
                with col_c3:
                    render_comp_card("Total Missing", comp["original_missing"], comp["cleaned_missing"])
                with col_c4:
                    render_comp_card("Duplicate Rows", comp["original_duplicates"], comp["cleaned_duplicates"])
                    
                st.write("")
                
                # Active dataset switcher button
                st.markdown("#### 🔄 Make Cleaned Dataset Active")
                st.write("Click below to set this new preprocessed dataset as the primary active dataset for visualization and training.")
                if st.button("Set Cleaned Dataset Active", type="primary", key="set_active_cleaned"):
                    st.session_state["active_dataset_id"] = res["cleaned_dataset_id"]
                    st.success(f"Active dataset switched to: `{res['cleaned_filename']}`")
                    st.rerun()
                
                with st.expander("📋 Preprocessing Pipeline Execution Logs"):
                    for log_line in res.get("execution_logs", []):
                        st.text(f"⚙️ {log_line}")



        # 2. Visualizations
        st.markdown('<div class="section-title">📊 Interactive Visualizations (Plotly)</div>', unsafe_allow_html=True)
        
        if df.empty:
            st.warning("Cannot generate charts. No data loaded.")
        else:
            # Let's create visualization tabs
            vis_tabs = st.tabs([
                "📊 Histogram", 
                "📦 Box Plot", 
                "📈 Scatter Plot", 
                "🔥 Correlation Heatmap", 
                "📊 Bar Chart", 
                "🍩 Pie Chart"
            ])
            
            num_cols = meta["numerical_columns"]
            cat_cols = meta["categorical_columns"]
            
            # Tab 1: Histogram
            with vis_tabs[0]:
                st.subheader("Histogram - Numerical Distribution")
                if not num_cols:
                    st.info("No numerical columns found in the dataset to plot a histogram.")
                else:
                    h_col1, h_col2 = st.columns([1, 3])
                    with h_col1:
                        hist_x = st.selectbox("Select X Axis (Numeric Column)", num_cols, key="hist_x")
                        hist_color = st.selectbox("Color By (Optional Categorical Column)", [None] + cat_cols, key="hist_color")
                        hist_bins = st.slider("Number of Bins", 5, 100, 30, key="hist_bins")
                    with h_col2:
                        fig = px.histogram(
                            df, 
                            x=hist_x, 
                            color=hist_color, 
                            nbins=hist_bins,
                            title=f"Distribution of {hist_x}",
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Safe
                        )
                        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)

            # Tab 2: Box Plot
            with vis_tabs[1]:
                st.subheader("Box Plot - Outlier Analysis")
                if not num_cols:
                    st.info("No numerical columns found in the dataset to plot a box plot.")
                else:
                    b_col1, b_col2 = st.columns([1, 3])
                    with b_col1:
                        box_y = st.selectbox("Select Y Axis (Numeric Column)", num_cols, key="box_y")
                        box_x = st.selectbox("Group By (Optional Categorical Column)", [None] + cat_cols, key="box_x")
                    with b_col2:
                        fig = px.box(
                            df, 
                            y=box_y, 
                            x=box_x,
                            title=f"Box Plot for {box_y} grouped by {box_x or 'All data'}",
                            template="plotly_dark",
                            color=box_x if box_x else None,
                            color_discrete_sequence=px.colors.qualitative.Safe
                        )
                        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)

            # Tab 3: Scatter Plot
            with vis_tabs[2]:
                st.subheader("Scatter Plot - 2D Relationships")
                if len(num_cols) < 2:
                    st.info("Scatter plots require at least 2 numerical columns.")
                else:
                    s_col1, s_col2 = st.columns([1, 3])
                    with s_col1:
                        scat_x = st.selectbox("Select X Axis (Numeric Column)", num_cols, index=0, key="scat_x")
                        scat_y = st.selectbox("Select Y Axis (Numeric Column)", num_cols, index=min(1, len(num_cols)-1), key="scat_y")
                        scat_color = st.selectbox("Color By (Optional Column)", [None] + num_cols + cat_cols, key="scat_color")
                    with s_col2:
                        fig = px.scatter(
                            df, 
                            x=scat_x, 
                            y=scat_y, 
                            color=scat_color,
                            title=f"{scat_y} vs {scat_x}",
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Safe
                        )
                        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)

            # Tab 4: Correlation Heatmap
            with vis_tabs[3]:
                st.subheader("Correlation Heatmap - Linear Correlations")
                if len(num_cols) < 2:
                    st.info("Heatmaps require at least 2 numerical columns.")
                else:
                    # Select numerical subset and compute Pearson correlation
                    df_numeric = df[num_cols].select_dtypes(include=[np.number])
                    
                    if df_numeric.empty or df_numeric.shape[1] < 2:
                        st.info("Not enough numeric column variation to plot a heatmap.")
                    else:
                        corr_matrix = df_numeric.corr()
                        
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            zmin=-1,
                            zmax=1,
                            title="Pearson Correlation Coefficient Matrix",
                            template="plotly_dark"
                        )
                        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)

            # Tab 5: Bar Chart
            with vis_tabs[4]:
                st.subheader("Bar Chart - Categorical Distribution Counts")
                if not cat_cols:
                    st.info("No categorical columns found in the dataset to plot a bar chart.")
                else:
                    bar_col1, bar_col2 = st.columns([1, 3])
                    with bar_col1:
                        bar_cat = st.selectbox("Select Categorical Column", cat_cols, key="bar_cat")
                    with bar_col2:
                        # Group counts
                        counts = df[bar_cat].value_counts().reset_index()
                        counts.columns = [bar_cat, "Count"]
                        
                        fig = px.bar(
                            counts, 
                            x=bar_cat, 
                            y="Count",
                            title=f"Frequency Counts of {bar_cat}",
                            template="plotly_dark",
                            color=bar_cat,
                            color_discrete_sequence=px.colors.qualitative.Safe
                        )
                        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)

            # Tab 6: Pie Chart
            with vis_tabs[5]:
                st.subheader("Pie Chart - Categorical Distribution Proportions")
                if not cat_cols:
                    st.info("No categorical columns found in the dataset to plot a pie chart.")
                else:
                    pie_col1, pie_col2 = st.columns([1, 3])
                    with pie_col1:
                        pie_cat = st.selectbox("Select Categorical Column", cat_cols, key="pie_cat")
                    with pie_col2:
                        counts = df[pie_cat].value_counts().reset_index()
                        counts.columns = [pie_cat, "Count"]
                        
                        fig = px.pie(
                            counts, 
                            names=pie_cat, 
                            values="Count",
                            title=f"Proportion Distribution of {pie_cat}",
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Safe
                        )
                        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)

        # Deletion option
        st.write("")
        st.write("---")
        with st.expander("🚨 Delete Dataset Option"):
            st.warning("Deleting this dataset will remove the CSV file from backend storage.")
            if st.button("Delete Dataset Permanently", type="primary"):
                success = api_client.delete_dataset(selected_dataset_id)
                if success:
                    st.success("Dataset successfully deleted!")
                    st.rerun()
                else:
                    st.error("Failed to delete dataset.")
