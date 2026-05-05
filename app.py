"""
Streamlit App: Malware Detection using Explainable AI
Clean, professional UI with data validation, CV results, and overfitting warnings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
import time

from preprocessing import DataPreprocessor
from model_enhanced import EnsembleModels, ImprovedCNNModel
from shap_explainer import SHAPExplainer


# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Malware Detection XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {color:#1f77b4; font-size:2.2em; font-weight:bold; text-align:center; margin-bottom:5px;}
    .subtitle {color:#555; font-size:0.95em; text-align:center; margin-bottom:20px;}
    </style>
""", unsafe_allow_html=True)


# ============= SESSION STATE =============
def init_session_state():
    defaults = {
        'model': None, 'ensemble': None, 'preprocessor': None,
        'feature_names': None, 'X_test_scaled': None, 'y_test': None,
        'X_val_scaled': None, 'y_val': None,
        'metrics': None, 'model_history': None,
        'y_pred': None, 'y_pred_proba': None, 'X_train_scaled': None,
        'shap_explainer': None, 'training_complete': False,
        'last_training_time': None, 'comparison_df': None,
        'data_report': None, 'cv_results_df': None,
        'overfitting_df': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ============= UTILITY =============
def save_uploaded_files_to_temp(uploaded_files):
    paths = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp.write(f.getbuffer())
            paths.append(tmp.name)
    return paths

def cleanup_temp_files(paths):
    for p in paths:
        try: os.unlink(p)
        except: pass


# ============= SIDEBAR =============
with st.sidebar:
    st.header("Navigation")
    mode = st.radio("Select Mode:", ["Train Model", "Make Prediction", "Model Analysis"])


# ============= MAIN =============
def main():
    st.markdown('<div class="main-title">🛡️ Malware Detection using Explainable AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">RAM-based malware detection with interpretable deep learning</div>', unsafe_allow_html=True)
    st.divider()

    # ===== TRAIN =====
    if mode == "Train Model":
        st.header("Train Model")

        uploaded_files = st.file_uploader("Upload CSV file(s):", type=['csv'],
                                          accept_multiple_files=True)

        if uploaded_files and len(uploaded_files) > 0:
            st.success(f"✓ {len(uploaded_files)} file(s) selected")

            col1, col2, col3 = st.columns(3)
            with col1: epochs = st.number_input("Epochs:", 10, 200, 100)
            with col2: batch_size = st.selectbox("Batch Size:", [16, 32, 64])
            with col3: do_cv = st.checkbox("5-Fold CV", True)

            if st.button("🚀 Train All Models", use_container_width=True, type="primary"):
                try:
                    progress_bar = st.progress(0)
                    status = st.status("Initializing...", state="running")
                    temp_paths = save_uploaded_files_to_temp(uploaded_files)

                    try:
                        # --- Preprocessing ---
                        with status: st.write("Preprocessing data (70/15/15 split)...")
                        progress_bar.progress(10)

                        preprocessor = DataPreprocessor()
                        (X_train, X_val, X_test,
                         y_train, y_val, y_test,
                         feature_names, data_report) = preprocessor.preprocess(temp_paths)

                        progress_bar.progress(25)

                        # --- Train ---
                        with status: st.write("Training CNN, Random Forest, Logistic Regression...")
                        progress_bar.progress(30)

                        ensemble = EnsembleModels()
                        ensemble.train_all_models(
                            X_train, X_val, X_test,
                            y_train, y_val, y_test,
                            do_cv=do_cv,
                            feature_names=feature_names,
                            verbose=False,
                        )

                        progress_bar.progress(85)

                        # --- Store state ---
                        st.session_state['ensemble'] = ensemble
                        st.session_state['model'] = ensemble.cnn
                        st.session_state['preprocessor'] = preprocessor
                        st.session_state['feature_names'] = feature_names
                        st.session_state['X_test_scaled'] = X_test
                        st.session_state['y_test'] = y_test
                        st.session_state['X_val_scaled'] = X_val
                        st.session_state['y_val'] = y_val
                        st.session_state['X_train_scaled'] = X_train
                        st.session_state['metrics'] = ensemble.results['CNN']
                        st.session_state['model_history'] = ensemble.cnn.history
                        st.session_state['y_pred'] = ensemble.cnn.y_pred
                        st.session_state['y_pred_proba'] = ensemble.cnn.y_pred_proba
                        st.session_state['training_complete'] = True
                        st.session_state['comparison_df'] = ensemble.get_comparison_dataframe()
                        st.session_state['data_report'] = data_report
                        st.session_state['overfitting_df'] = ensemble.get_overfitting_report()
                        st.session_state['last_training_time'] = time.time()

                        if do_cv:
                            st.session_state['cv_results_df'] = ensemble.get_cv_dataframe()

                        # SHAP
                        try:
                            explainer = SHAPExplainer(ensemble.cnn.model, feature_names)
                            explainer.init_with_background_data(X_train)
                            st.session_state['shap_explainer'] = explainer
                        except Exception as e:
                            st.warning(f"SHAP init: {e}")
                            st.session_state['shap_explainer'] = None

                        progress_bar.progress(100)
                        status.update(label="✓ All Models Trained!", state="complete")

                    finally:
                        cleanup_temp_files(temp_paths)

                except Exception as e:
                    st.error(f"Training Error: {e}")
                    import traceback
                    st.error(traceback.format_exc())

        # --- Show results ---
        if st.session_state['training_complete'] and st.session_state['metrics']:
            st.divider()

            # Data quality report
            report = st.session_state.get('data_report')
            if report:
                with st.expander("📋 Data Quality Report", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Samples", report.get('num_samples', '?'))
                    c2.metric("Features", report.get('num_features', '?'))
                    c3.metric("Duplicates Removed", report.get('duplicates_removed', 0))
                    c4.metric("Class Imbalance", f"{report.get('class_imbalance_ratio', 1):.1f}:1")

                    cc = report.get('class_counts', {})
                    st.write(f"**Classes:** Benign = {cc.get('benign',0)}, Malware = {cc.get('malware',0)}")

                    nd_tt = report.get('train_test_near_duplicates', 0)
                    nd_tv = report.get('train_val_near_duplicates', 0)
                    if nd_tt > 0 or nd_tv > 0:
                        st.warning(f"⚠ Near-duplicates: train↔test={nd_tt}, train↔val={nd_tv}")
                    else:
                        st.success("✅ No near-duplicate leakage detected")

                    st.write(f"**Split:** Train={report.get('train_size','?')}, "
                             f"Val={report.get('val_size','?')}, Test={report.get('test_size','?')}")

            # Metrics
            metrics = st.session_state['metrics']
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Accuracy",  f"{metrics.get('accuracy',0):.4f}")
            c2.metric("Precision", f"{metrics.get('precision',0):.4f}")
            c3.metric("Recall",    f"{metrics.get('recall',0):.4f}")
            c4.metric("F1-Score",  f"{metrics.get('f1',0):.4f}")
            c5.metric("ROC-AUC",  f"{metrics.get('roc_auc',0):.4f}")

            # Overfitting warnings
            of_df = st.session_state.get('overfitting_df')
            if of_df is not None:
                overfit_models = of_df[of_df['Status'].str.contains('Overfitting')]
                if len(overfit_models) > 0:
                    st.warning(f"⚠️ **Overfitting detected** in: "
                               f"{', '.join(overfit_models['Model'].tolist())} "
                               f"(train-test accuracy gap > 5%)")
                with st.expander("🔍 Overfitting Analysis", expanded=False):
                    st.dataframe(of_df, use_container_width=True, hide_index=True)

            # CV results
            cv_df = st.session_state.get('cv_results_df')
            if cv_df is not None:
                with st.expander("📊 Cross-Validation Results (5-Fold)", expanded=False):
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)

            # Plots
            st.subheader("Visualizations")
            model_obj = st.session_state['model']
            c1, c2, c3 = st.columns(3)
            with c1:
                try:
                    ens = st.session_state.get('ensemble')
                    if ens:
                        fig = ens.plot_cnn_training_history()
                        if fig: st.pyplot(fig)
                except: pass
            with c2:
                try:
                    if model_obj.y_pred is not None:
                        fig = model_obj.plot_confusion_matrix()
                        if fig: st.pyplot(fig)
                except: pass
            with c3:
                try:
                    if model_obj.y_pred_proba is not None:
                        fig = model_obj.plot_roc_curve()
                        if fig: st.pyplot(fig)
                except: pass

    # ===== PREDICT =====
    elif mode == "Make Prediction":
        st.header("Make Prediction")

        if not st.session_state['training_complete'] or st.session_state['model'] is None:
            st.warning("No trained model available.")
            st.info("👈 Train a model in the **Train Model** tab first.")
        else:
            st.success(f"✓ Model loaded with {len(st.session_state['feature_names'])} features")
            st.divider()

            with st.form("prediction_form"):
                st.write("Enter normalized feature values (0.0 - 1.0)")
                input_values = {}
                cols = st.columns(3)
                for idx, feature in enumerate(st.session_state['feature_names']):
                    with cols[idx % 3]:
                        input_values[feature] = st.number_input(f"{feature}", value=0.5,
                                                                 min_value=0.0, max_value=1.0, step=0.01)
                predict_btn = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

            if predict_btn:
                try:
                    model_obj = st.session_state['model']
                    arr = np.array([input_values[f] for f in st.session_state['feature_names']]).reshape(1, -1)

                    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                        st.error("Invalid input values")
                    else:
                        with st.spinner("Computing prediction..."):
                            pred_proba = model_obj.model.predict(arr, verbose=0)
                            pred_class = 1 if pred_proba[0][0] > 0.5 else 0
                            confidence = max(pred_proba[0][0], 1 - pred_proba[0][0])

                            st.divider()
                            c1, c2 = st.columns(2)
                            with c1:
                                st.subheader("Prediction")
                                if pred_class == 0:
                                    st.success(f"✅ BENIGN\nConfidence: {confidence*100:.1f}%")
                                else:
                                    st.error(f"⚠️ MALWARE\nConfidence: {confidence*100:.1f}%")

                            with c2:
                                st.subheader("Why This Prediction?")
                                if st.button("📈 Explain with SHAP", use_container_width=True):
                                    explainer = st.session_state.get('shap_explainer')
                                    if explainer is None:
                                        st.error("SHAP explainer not available")
                                    else:
                                        with st.spinner("Computing SHAP..."):
                                            try:
                                                shap_exp = explainer.explain_instance(arr, num_samples=50)
                                                top_df = explainer.get_top_contributing_features(shap_exp, top_n=5)
                                                st.dataframe(top_df, use_container_width=True, hide_index=True)
                                                fig = explainer.plot_waterfall(shap_exp)
                                                st.pyplot(fig, use_container_width=True)
                                            except Exception as e:
                                                st.error(f"SHAP failed: {e}")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    # ===== ANALYSIS =====
    elif mode == "Model Analysis":
        st.header("Model Analysis")

        if not st.session_state['training_complete'] or st.session_state['metrics'] is None:
            st.warning("No trained model available.")
            st.info("👈 Train a model in the **Train Model** tab first.")
        else:
            ensemble = st.session_state.get('ensemble')

            # --- Data Quality ---
            report = st.session_state.get('data_report')
            if report:
                st.subheader("📋 Data Quality Report")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total Samples", report.get('num_samples','?'))
                c2.metric("Features", report.get('num_features','?'))
                c3.metric("Duplicates Removed", report.get('duplicates_removed',0))
                c4.metric("Imbalance Ratio", f"{report.get('class_imbalance_ratio',1):.1f}:1")

                ir = report.get('class_imbalance_ratio', 1)
                if ir > 3:
                    st.warning(f"⚠️ Significant class imbalance ({ir:.1f}:1). Class weights applied during training.")
                elif ir > 1.5:
                    st.info(f"ℹ️ Moderate class imbalance ({ir:.1f}:1). Class weights applied.")
                else:
                    st.success("✅ Balanced classes")
                st.divider()

            # --- 1. Comparison Table ---
            st.subheader("1️⃣ Model Performance Comparison")
            if st.session_state['comparison_df'] is not None:
                cdf = st.session_state['comparison_df']
                st.dataframe(cdf, use_container_width=True)
                best = cdf.index[0]
                st.info(f"✅ **Best Model:** {best} (Accuracy: {cdf.iloc[0]['accuracy']:.4f})")
            st.divider()

            # --- 2. Overfitting Analysis ---
            st.subheader("2️⃣ Overfitting Analysis")
            of_df = st.session_state.get('overfitting_df')
            if of_df is not None:
                st.dataframe(of_df, use_container_width=True, hide_index=True)
                overfit = of_df[of_df['Status'].str.contains('Overfitting')]
                if len(overfit) > 0:
                    st.error(f"⚠️ Models with overfitting: {', '.join(overfit['Model'].tolist())}")
                    st.write("**Recommendation:** These models show >5% gap between train and test accuracy. "
                             "Consider more regularization or more training data.")
                else:
                    st.success("✅ No significant overfitting detected across all models")
            st.divider()

            # --- 3. Cross-Validation ---
            st.subheader("3️⃣ Cross-Validation Results (5-Fold)")
            cv_df = st.session_state.get('cv_results_df')
            if cv_df is not None:
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
                st.info("💡 CV scores represent mean ± std across 5 stratified folds on training data only.")
            else:
                st.info("Enable '5-Fold CV' checkbox during training to see CV results.")
            st.divider()

            # --- 4. CNN Training History (smoothed) ---
            st.subheader("4️⃣ CNN Training History (Smoothed)")
            if ensemble:
                try:
                    fig = ensemble.plot_cnn_training_history()
                    if fig: st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not display: {e}")
            st.divider()

            # --- 5. ROC Curves ---
            st.subheader("5️⃣ ROC Curves — All Models")
            if ensemble:
                try: st.pyplot(ensemble.plot_roc_curves())
                except Exception as e: st.error(f"Error: {e}")
            st.divider()

            # --- 6. Precision-Recall ---
            st.subheader("6️⃣ Precision-Recall Curves")
            if ensemble:
                try: st.pyplot(ensemble.plot_precision_recall_curves())
                except Exception as e: st.error(f"Error: {e}")
            st.divider()

            # --- 7. Confusion Matrices ---
            st.subheader("7️⃣ Confusion Matrices")
            if ensemble:
                try: st.pyplot(ensemble.plot_confusion_matrices())
                except Exception as e: st.error(f"Error: {e}")
            st.divider()

            # --- 8. Metrics Bar Chart ---
            st.subheader("8️⃣ Metrics Comparison Chart")
            if ensemble:
                try: st.pyplot(ensemble.plot_metrics_comparison())
                except Exception as e: st.error(f"Error: {e}")
            st.divider()

            # --- 9. Learning Curves ---
            st.subheader("9️⃣ Learning Curves")
            if ensemble:
                try:
                    fig = ensemble.plot_learning_curves()
                    if fig: st.pyplot(fig)
                except Exception as e: st.warning(f"Learning curves: {e}")
            st.divider()

            # --- 10. RF Feature Importance ---
            if ensemble and ensemble.rf:
                st.subheader("🔟 Random Forest Feature Importance (Top 15)")
                try:
                    fig = ensemble.plot_rf_feature_importance(top_n=15)
                    if fig: st.pyplot(fig)
                except Exception as e: st.warning(f"Feature importance: {e}")
            st.divider()

            # --- 11. SHAP ---
            st.subheader("1️⃣1️⃣ CNN — SHAP Global Feature Importance")
            st.info("💡 SHAP shows which features are most important for CNN predictions")
            try:
                explainer = st.session_state.get('shap_explainer')
                X_test = st.session_state.get('X_test_scaled')
                if explainer and X_test is not None:
                    if st.button("📊 Compute SHAP (CNN)", use_container_width=True, type="primary"):
                        with st.spinner("Computing SHAP values (30-60s)..."):
                            try:
                                result = explainer.explain_batch(X_test, max_samples=10, num_samples=100)
                                c1, c2 = st.columns([1, 2])
                                with c1:
                                    st.write("**Top 10 Features**")
                                    st.dataframe(explainer.get_feature_importance(result, top_n=10),
                                                 use_container_width=True)
                                with c2:
                                    st.pyplot(explainer.plot_summary(result, max_display=10))
                                st.success("✅ SHAP complete")
                            except Exception as e:
                                st.error(f"SHAP Error: {e}")
                elif explainer is None:
                    st.warning("SHAP explainer not initialized")
            except Exception as e:
                st.error(f"Error: {e}")

            # --- 12. Model Insights ---
            st.divider()
            st.subheader("💡 Model Comparison Insights")
            if st.session_state['comparison_df'] is not None:
                cdf = st.session_state['comparison_df']
                best_name = cdf.index[0]
                worst_name = cdf.index[-1]
                best_acc = cdf.iloc[0]['accuracy']
                worst_acc = cdf.iloc[-1]['accuracy']

                st.markdown(f"""
                **Summary:**
                - **Best performing model:** {best_name} with {best_acc:.2%} accuracy
                - **Lowest performing model:** {worst_name} with {worst_acc:.2%} accuracy
                - **Performance spread:** {(best_acc - worst_acc)*100:.1f} percentage points between best and worst
                - Models use different architectures (CNN, tree-based, linear) providing complementary perspectives
                - All evaluations performed on **held-out test data** not seen during training
                """)


if __name__ == "__main__":
    main()
