# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from PIL import Image
from wordcloud import WordCloud

from utils import new_line
from config import set_page_config
from session_state import initial_state
from data_loading import load_data
import eda_module
from missing_values_handler import handle_missing_values
from CTGD import handle_categorical_data
from scaling_functions import display_scaling_options
from transformation_functions import display_transformation_options
from feature_engineering import extract_feature, transform_feature, select_feature, show_dataframe
from model_building import build_model


# Set configuration and initialize state
set_page_config()
initial_state()


# Progress Bar
def progress_bar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0002)
        my_bar.progress(percent_complete + 1)


# Logo 
col1, col2, col3 = st.columns([0.25,1,0.25])
col2.image("./assets/logo.png", use_column_width=True)
new_line(2)

# Description
st.markdown("""Welcome to Click Analytics! 🚀 
Dive right into the future of data with our user-friendly platform designed for everyone—no coding or machine learning experience required!
With just a few clicks, you can start preparing your data, training cutting-edge models, and uncovering valuable insights. 
Whether you're a data enthusiast or a seasoned analyst, Click Analytics empowers you to effortlessly create, analyze, and explore. 
What are you waiting for? Start building your very own analytics and models today and see what decisions you can empower with your data!!""", unsafe_allow_html=True)
st.divider()


# Dataframe selection
st.markdown("<h2 align='center'> <b> Getting Started", unsafe_allow_html=True)
new_line(1)
st.write("The first step is to upload your data. You can upload your data either by : **Upload File**, or **Write URL**. In all ways the data should be a csv file or Excel and should not exceed 200 MB.")
new_line(1)



# Uploading Way
uploading_way = st.session_state.uploading_way
col1, col2, col3 = st.columns(3,gap='large')

# Upload
def upload_click(): st.session_state.uploading_way = "upload"
col1.markdown("<h5 align='center'> Upload File", unsafe_allow_html=True)
col1.button("Upload File", key="upload_file", use_container_width=True, on_click=upload_click)
        
# URL
def url_click(): st.session_state.uploading_way = "url"
col3.markdown("<h5 align='center'> Write URL", unsafe_allow_html=True)
col3.button("Write URL", key="write_url", use_container_width=True, on_click=url_click)



# No Data
if st.session_state.df is None:

    # Upload
    if uploading_way == "upload":
        uploaded_file = st.file_uploader("Upload the Dataset", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            try:
                df = load_data(uploaded_file)
                st.session_state.df = df
            except Exception as e:
                st.error(f"Error loading the file: {e}")

    # URL
    elif uploading_way == "url":
        url = st.text_input("Enter URL")
        if url:
            df = load_data(url)
            st.session_state.df = df


# Sidebar       
with st.sidebar:
    st.image("./assets/logo.png",   use_column_width=True)
    
    
# Dataframe
if st.session_state.df is not None:

    # Re-initialize the variables from the state
    df = st.session_state.df
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    X_val = st.session_state.X_val
    y_val = st.session_state.y_val
    trained_model = st.session_state.trained_model
    is_train = st.session_state.is_train
    is_test = st.session_state.is_test
    is_val = st.session_state.is_val
    model = st.session_state.model
    show_eval = st.session_state.show_eval
    y_pred_train = st.session_state.y_pred_train
    y_pred_test = st.session_state.y_pred_test
    y_pred_val = st.session_state.y_pred_val
    metrics_df = st.session_state.metrics_df

    st.divider()
    new_line()

    # Call to the EDA module function
    eda_module.show_eda(df)


    # Missing Values
    handle_missing_values(df)


    # Encoding
    handle_categorical_data(st, df)


    # Scaling
    new_line()
    st.markdown("### ⚖️ Scaling", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Scaling"):
        new_line()

        # Scaling Methods
        display_scaling_options(st, df)


    # Data Transformation
    display_transformation_options(st, df)


    
    # Feature Engineering
    new_line()
    st.markdown("### ⚡ Feature Engineering", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Feature Engineering"):
        df = extract_feature(df)
        df = transform_feature(df)
        df = select_feature(df)
        show_dataframe(df)


    # Data Splitting
    st.markdown("### 🪚 Data Splitting", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Data Splitting"):

        new_line()
        train_size, val_size, test_size = 0,0,0
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Variable", df.columns.tolist(), key='target', help="Target Variable is the variable that you want to predict.")
            st.session_state['target_variable'] = target
        with col2:
            sets = st.selectbox("Select The Split Sets", ["Select", "Train and Test", "Train, Validation, and Test"], key='sets', help="Train Set is the data used to train the model. Validation Set is the data used to validate the model. Test Set is the data used to test the model. ")
            st.session_state['split_sets'] = sets

        if sets != "Select" and target:
            if sets == "Train, Validation, and Test" :
                new_line()
                col1, col2, col3 = st.columns(3)
                with col1:
                    train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                    train_size = round(train_size, 2)
                with col2:
                    val_size = st.number_input("Validation Size", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key='val_size')
                    val_size = round(val_size, 2)
                with col3:
                    test_size = st.number_input("Test Size", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key='test_size')
                    test_size = round(test_size, 2)

                if float(train_size + val_size + test_size) != 1.0:
                    new_line()
                    st.error(f"The sum of Train, Validation, and Test sizes must be equal to 1.0, your sum is: **train** + **validation** + **test** = **{train_size}** + **{val_size}** + **{test_size}** = **{sum([train_size, val_size, test_size])}**" )
                    new_line()

                else:
                    split_button = ""
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    with col2:
                        new_line()
                        split_button = st.button("Split Data", use_container_width=True)
                        
                        if split_button:
                            st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size= {val_size} / (1.0 - {train_size}),random_state=42)
\n """
                            from sklearn.model_selection import train_test_split
                            X_train, X_rem, y_train, y_rem = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                            X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size= val_size / (1.0 - train_size),random_state=42)
                            st.session_state['X_train'] = X_train
                            st.session_state['X_val'] = X_val
                            st.session_state['X_test'] = X_test
                            st.session_state['y_train'] = y_train
                            st.session_state['y_val'] = y_val
                            st.session_state['y_test'] = y_test

                    
                    col1, col2, col3 = st.columns(3)
                    if split_button:
                        st.success("Data Splitting Done!")
                        with col1:
                            st.write("Train Set")
                            st.write("X Train Shape: ", X_train.shape)
                            st.write("Y Train Shape: ", y_train.shape)

                            train = pd.concat([X_train, y_train], axis=1)
                            train_csv = train.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Train Set", train_csv, "train.csv", "text/csv", key='train3')

                        with col2:
                            st.write("Validation Set")
                            st.write("X Validation Shape: ", X_val.shape)
                            st.write("Y Validation Shape: ", y_val.shape)

                            val = pd.concat([X_val, y_val], axis=1)
                            val_csv = val.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Validation Set", val_csv, "validation.csv", key='val3')

                        with col3:
                            st.write("Test Set")
                            st.write("X Test Shape: ", X_test.shape)
                            st.write("Y Test Shape: ", y_test.shape)

                            test = pd.concat([X_test, y_test], axis=1)
                            test_csv = test.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Test Set", test_csv, "test.csv", key='test3')


            elif sets == "Train and Test":

                new_line()
                col1, col2 = st.columns(2)
                with col1:
                    train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                    train_size = round(train_size, 2)
                with col2:
                    test_size = st.number_input("Test Size", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key='val_size')
                    test_size = round(test_size, 2)

                if float(train_size + test_size) != 1.0:
                    new_line()
                    st.error(f"The sum of Train, Validation, and Test sizes must be equal to 1.0, your sum is: **train** + **test** = **{train_size}** + **{test_size}** = **{sum([train_size, test_size])}**" )
                    new_line()

                else:
                    split_button = ""
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    with col2:
                        new_line()
                        split_button = st.button("Split Data")

                        if split_button:
                            st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
\n """
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                            st.session_state['X_train'] = X_train
                            st.session_state['X_test'] = X_test
                            st.session_state['y_train'] = y_train
                            st.session_state['y_test'] = y_test

                    
                    
                    col1, col2 = st.columns(2)
                    if split_button:
                        st.success("Data Splitting Done!")
                        with col1:
                            st.write("Train Set")
                            st.write("X Train Shape: ", X_train.shape)
                            st.write("Y Train Shape: ", y_train.shape)

                            train = pd.concat([X_train, y_train], axis=1)
                            train_csv = train.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Train Set", train_csv, "train.csv", key='train2')

                        with col2:
                            st.write("Test Set")
                            st.write("X test Shape: ", X_test.shape)
                            st.write("Y test Shape: ", y_test.shape)

                            test = pd.concat([X_test, y_test], axis=1)
                            test_csv = test.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Test Set", test_csv, "test.csv", key='test2')


    # Building the model
    if 'X_train' in st.session_state and 'y_train' in st.session_state:
        build_model(st.session_state['X_train'], st.session_state['y_train'])


    # Evaluation
    if st.session_state['trained_model_bool']:
        st.markdown("### 📈 Evaluation")
        new_line()
        with st.expander("Model Evaluation"):
            # Load the model
            import joblib
            model = joblib.load('model.pkl')
            

            if str(model) not in st.session_state.lst_models_predctions:
                
                st.session_state.lst_models_predctions.append(str(model))
                st.session_state.lst_models.append(str(model))
                if str(model) not in st.session_state.models_with_eval.keys():
                    st.session_state.models_with_eval[str(model)] = []


                

                # Predictions
                if st.session_state["split_sets"] == "Train, Validation, and Test":
                        
                        st.session_state.all_the_process += f"""
# Predictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)
\n """
                        y_pred_train = model.predict(X_train)
                        st.session_state.y_pred_train = y_pred_train
                        y_pred_val = model.predict(X_val)
                        st.session_state.y_pred_val = y_pred_val
                        y_pred_test = model.predict(X_test)
                        st.session_state.y_pred_test = y_pred_test


                elif st.session_state["split_sets"] == "Train and Test":
                    
                    st.session_state.all_the_process += f"""
# Predictions 
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
\n """  
                    
                    y_pred_train = model.predict(X_train)
                    st.session_state.y_pred_train = y_pred_train
                    y_pred_test = model.predict(X_test)
                    st.session_state.y_pred_test = y_pred_test

            # Choose Evaluation Metric
            if st.session_state['problem_type'] == "Classification":
                evaluation_metric = st.multiselect("Evaluation Metric", ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"], key='evaluation_metric')

            elif st.session_state['problem_type'] == "Regression":
                evaluation_metric = st.multiselect("Evaluation Metric", ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R2 Score"], key='evaluation_metric')

            
            col1, col2, col3 = st.columns([1, 0.6, 1])
            
            st.session_state.show_eval = True
                
            
            if evaluation_metric != []:
                

                for metric in evaluation_metric:


                        if metric == "Accuracy":

                            # Check if Accuary is element of the list of that model
                            if "Accuracy" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("Accuracy")

                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - Accuracy 
from sklearn.metrics import accuracy_score
print("Accuracy Score on Train Set: ", accuracy_score(y_train, y_pred_train))
print("Accuracy Score on Validation Set: ", accuracy_score(y_val, y_pred_val))
print("Accuracy Score on Test Set: ", accuracy_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import accuracy_score
                                    train_acc = accuracy_score(y_train, y_pred_train)
                                    val_acc = accuracy_score(y_val, y_pred_val)
                                    test_acc = accuracy_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_acc, val_acc, test_acc]
                                    st.session_state['metrics_df'] = metrics_df


                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy Score on Train Set: ", accuracy_score(y_train, y_pred_train))
print("Accuracy Score on Test Set: ", accuracy_score(y_test, y_pred_test))
\n """

                                    from sklearn.metrics import accuracy_score
                                    train_acc = accuracy_score(y_train, y_pred_train)
                                    test_acc = accuracy_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_acc, test_acc]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "Precision":
                            
                            if "Precision" not in st.session_state.models_with_eval[str(model)]:
                                
                                st.session_state.models_with_eval[str(model)].append("Precision")

                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - Precision
from sklearn.metrics import precision_score
print("Precision Score on Train Set: ", precision_score(y_train, y_pred_train))
print("Precision Score on Validation Set: ", precision_score(y_val, y_pred_val))
print("Precision Score on Test Set: ", precision_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import precision_score
                                    train_prec = precision_score(y_train, y_pred_train)
                                    val_prec = precision_score(y_val, y_pred_val)
                                    test_prec = precision_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_prec, val_prec, test_prec]
                                    st.session_state['metrics_df'] = metrics_df
                                    
                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - Precision
from sklearn.metrics import precision_score
print("Precision Score on Train Set: ", precision_score(y_train, y_pred_train))
print("Precision Score on Test Set: ", precision_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import precision_score
                                    train_prec = precision_score(y_train, y_pred_train)
                                    test_prec = precision_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_prec, test_prec]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "Recall":

                            if "Recall" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("Recall")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - Recall
from sklearn.metrics import recall_score
print("Recall Score on Train Set: ", recall_score(y_train, y_pred_train))
print("Recall Score on Validation Set: ", recall_score(y_val, y_pred_val))
print("Recall Score on Test Set: ", recall_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import recall_score
                                    train_rec = recall_score(y_train, y_pred_train)
                                    val_rec = recall_score(y_val, y_pred_val)
                                    test_rec = recall_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_rec, val_rec, test_rec]
                                    st.session_state['metrics_df'] = metrics_df

                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - Recall
from sklearn.metrics import recall_score
print("Recall Score on Train Set: ", recall_score(y_train, y_pred_train))
print("Recall Score on Test Set: ", recall_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import recall_score
                                    train_rec = recall_score(y_train, y_pred_train)
                                    test_rec = recall_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_rec, test_rec]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "F1 Score":

                            if "F1 Score" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("F1 Score")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - F1 Score
from sklearn.metrics import f1_score
print("F1 Score on Train Set: ", f1_score(y_train, y_pred_train))
print("F1 Score on Validation Set: ", f1_score(y_val, y_pred_val))
print("F1 Score on Test Set: ", f1_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import f1_score
                                    train_f1 = f1_score(y_train, y_pred_train)
                                    val_f1 = f1_score(y_val, y_pred_val)
                                    test_f1 = f1_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_f1, val_f1, test_f1]
                                    st.session_state['metrics_df'] = metrics_df

                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - F1 Score
from sklearn.metrics import f1_score
print("F1 Score on Train Set: ", f1_score(y_train, y_pred_train))
print("F1 Score on Test Set: ", f1_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import f1_score
                                    train_f1 = f1_score(y_train, y_pred_train)
                                    test_f1 = f1_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_f1, test_f1]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "AUC Score":

                            if "AUC Score" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("AUC Score")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - AUC Score
from sklearn.metrics import roc_auc_score
print("AUC Score on Train Set: ", roc_auc_score(y_train, y_pred_train))
print("AUC Score on Validation Set: ", roc_auc_score(y_val, y_pred_val))
print("AUC Score on Test Set: ", roc_auc_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import roc_auc_score
                                    train_auc = roc_auc_score(y_train, y_pred_train)
                                    val_auc = roc_auc_score(y_val, y_pred_val)
                                    test_auc = roc_auc_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_auc, val_auc, test_auc]
                                    st.session_state['metrics_df'] = metrics_df

                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - AUC Score
from sklearn.metrics import roc_auc_score
print("AUC Score on Train Set: ", roc_auc_score(y_train, y_pred_train))
print("AUC Score on Test Set: ", roc_auc_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import roc_auc_score
                                    train_auc = roc_auc_score(y_train, y_pred_train)
                                    test_auc = roc_auc_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_auc, test_auc]
                                    st.session_state['metrics_df'] = metrics_df
                            

                        elif metric == "Mean Absolute Error (MAE)":

                            if "Mean Absolute Error (MAE)" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("Mean Absolute Error (MAE)")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - MAE
from sklearn.metrics import mean_absolute_error
print("MAE on Train Set: ", mean_absolute_error(y_train, y_pred_train))
print("MAE on Validation Set: ", mean_absolute_error(y_val, y_pred_val))
print("MAE on Test Set: ", mean_absolute_error(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import mean_absolute_error
                                    train_mae = mean_absolute_error(y_train, y_pred_train)
                                    val_mae = mean_absolute_error(y_val, y_pred_val)
                                    test_mae = mean_absolute_error(y_test, y_pred_test)

                                    metrics_df[metric] = [train_mae, val_mae, test_mae]
                                    st.session_state['metrics_df'] = metrics_df

                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - MAE
from sklearn.metrics import mean_absolute_error
print("MAE on Train Set: ", mean_absolute_error(y_train, y_pred_train))
print("MAE on Test Set: ", mean_absolute_error(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import mean_absolute_error
                                    train_mae = mean_absolute_error(y_train, y_pred_train)
                                    test_mae = mean_absolute_error(y_test, y_pred_test)

                                    metrics_df[metric] = [train_mae, test_mae]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "Mean Squared Error (MSE)":

                            if "Mean Squared Error (MSE)" not in st.session_state.models_with_eval[str(model)]:
                                
                                st.session_state.models_with_eval[str(model)].append("Mean Squared Error (MSE)")

                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - MSE
from sklearn.metrics import mean_squared_error
print("MSE on Train Set: ", mean_squared_error(y_train, y_pred_train))
print("MSE on Validation Set: ", mean_squared_error(y_val, y_pred_val))
print("MSE on Test Set: ", mean_squared_error(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import mean_squared_error
                                    train_mse = mean_squared_error(y_train, y_pred_train)
                                    val_mse = mean_squared_error(y_val, y_pred_val)
                                    test_mse = mean_squared_error(y_test, y_pred_test)

                                    metrics_df[metric] = [train_mse, val_mse, test_mse]
                                    st.session_state['metrics_df'] = metrics_df

                                else:

                                    st.session_state.all_the_process += f"""
# Evaluation - MSE
from sklearn.metrics import mean_squared_error
print("MSE on Train Set: ", mean_squared_error(y_train, y_pred_train))
print("MSE on Test Set: ", mean_squared_error(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import mean_squared_error
                                    train_mse = mean_squared_error(y_train, y_pred_train)
                                    test_mse = mean_squared_error(y_test, y_pred_test)

                                    metrics_df[metric] = [train_mse, test_mse]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "Root Mean Squared Error (RMSE)":

                            if "Root Mean Squared Error (RMSE)" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("Root Mean Squared Error (RMSE)")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - RMSE
from sklearn.metrics import mean_squared_error
print("RMSE on Train Set: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("RMSE on Validation Set: ", np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("RMSE on Test Set: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
\n """
                                    from sklearn.metrics import mean_squared_error
                                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                                    metrics_df[metric] = [train_rmse, val_rmse, test_rmse]
                                    st.session_state['metrics_df'] = metrics_df

                                else:

                                    st.session_state.all_the_process += f"""
# Evaluation - RMSE
from sklearn.metrics import mean_squared_error
print("RMSE on Train Set: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("RMSE on Test Set: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
\n """
                                    from sklearn.metrics import mean_squared_error
                                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                                    metrics_df[metric] = [train_rmse, test_rmse]
                                    st.session_state['metrics_df'] = metrics_df

                            
                        elif metric == "R2 Score":

                            if "R2 Score" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("R2 Score")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - R2 Score
from sklearn.metrics import r2_score
print("R2 Score on Train Set: ", r2_score(y_train, y_pred_train))
print("R2 Score on Validation Set: ", r2_score(y_val, y_pred_val))
print("R2 Score on Test Set: ", r2_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import r2_score
                                    train_r2 = r2_score(y_train, y_pred_train)
                                    val_r2 = r2_score(y_val, y_pred_val)
                                    test_r2 = r2_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_r2, val_r2, test_r2]
                                    st.session_state['metrics_df'] = metrics_df

                                else:

                                    st.session_state.all_the_process += f"""
# Evaluation - R2 Score
from sklearn.metrics import r2_score
print("R2 Score on Train Set: ", r2_score(y_train, y_pred_train))
print("R2 Score on Test Set: ", r2_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import r2_score
                                    train_r2 = r2_score(y_train, y_pred_train)
                                    test_r2 = r2_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_r2, test_r2]
                                    st.session_state['metrics_df'] = metrics_df



                # Show Evaluation Metric
                if show_eval:
                    new_line()
                    col1, col2, col3 = st.columns([0.5, 1, 0.5])
                    st.markdown("### Evaluation Metric")

                    if st.session_state["split_sets"] == "Train, Validation, and Test":
                        st.session_state['metrics_df'].index = ['Train', 'Validation', 'Test']
                        st.write(st.session_state['metrics_df'])

                    elif st.session_state["split_sets"] == "Train and Test":
                        st.session_state['metrics_df'].index = ['Train', 'Test']
                        st.write(st.session_state['metrics_df'])

                    


                    # Show Evaluation Metric Plot
                    new_line()
                    st.markdown("### Evaluation Metric Plot")
                    st.line_chart(st.session_state['metrics_df'])

                    # Show ROC Curve as plot
                    if "AUC Score" in evaluation_metric:
                        from sklearn.metrics import plot_roc_curve
                        st.markdown("### ROC Curve")
                        new_line()
                        
                        if st.session_state["split_sets"] == "Train, Validation, and Test":

                            # Show the ROC curve plot without any columns
                            col1, col2, col3 = st.columns([0.2, 1, 0.2])
                            fig, ax = plt.subplots()
                            plot_roc_curve(model, X_train, y_train, ax=ax)
                            plot_roc_curve(model, X_val, y_val, ax=ax)
                            plot_roc_curve(model, X_test, y_test, ax=ax)
                            ax.legend(['Train', 'Validation', 'Test'])
                            col2.pyplot(fig, legend=True)

                        elif st.session_state["split_sets"] == "Train and Test":

                            # Show the ROC curve plot without any columns
                            col1, col2, col3 = st.columns([0.2, 1, 0.2])
                            fig, ax = plt.subplots()
                            plot_roc_curve(model, X_train, y_train, ax=ax)
                            plot_roc_curve(model, X_test, y_test, ax=ax)
                            ax.legend(['Train', 'Test'])
                            col2.pyplot(fig, legend=True)

                            

                    # Show Confusion Matrix as plot
                    if st.session_state['problem_type'] == "Classification":
                        # from sklearn.metrics import plot_confusion_matrix
                        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
                        st.markdown("### Confusion Matrix")
                        new_line()

                        cm = confusion_matrix(y_test, y_pred_test)
                        col1, col2, col3 = st.columns([0.2,1,0.2])
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax)
                        col2.pyplot(fig)
                        
                        # Show the confusion matrix plot without any columns
                        # col1, col2, col3 = st.columns([0.2, 1, 0.2])
                        # fig, ax = plt.subplots()
                        # plot_confusion_matrix(model, X_test, y_test, ax=ax)
                        # col2.pyplot(fig)

                     
    st.divider()          
    col1, col2, col3= st.columns(3, gap='small')        

    if col1.button("🎬 Show df", use_container_width=True):
        new_line()
        st.subheader(" 🎬 Show The Dataframe")
        st.write("The dataframe is the dataframe that is used on this application to build the Machine Learning model. You can see the dataframe below 👇")
        new_line()
        st.dataframe(df, use_container_width=True)

    st.session_state.df.to_csv("df.csv", index=False)
    df_file = open("df.csv", "rb")
    df_bytes = df_file.read()
    if col2.download_button("📌 Download df", df_bytes, "df.csv", key='save_df', use_container_width=True):
        st.success("Downloaded Successfully!")


    if col3.button("⛔ Reset", use_container_width=True):
        new_line()
        st.subheader("⛔ Reset")
        st.write("Click the button below to reset the app and start over again")
        new_line()
        st.session_state.reset_1 = True

    if st.session_state.reset_1:
        col1, col2, col3 = st.columns(3)
        if col2.button("⛔ Reset", use_container_width=True, key='reset'):
            st.session_state.df = None
            st.session_state.clear()
            st.experimental_rerun()

