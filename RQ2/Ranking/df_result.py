import glob
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from PI_mterics import calculate_picp, calculate_pinc, calculate_MPIW, calculate_PINAW, calculate_PINRW, calculate_score
from dataframe import plot_dataframe, plot_dataframe2,plot_dataframe_,plot_dataframe2_
import numpy as np

BT0 = "Without Refit"
BT1 = "Fixed Backtesting"
BT2 = "Intermittent Refitting"
BT3 = "Refit Increasing Training Size"

name_model = ["LinearRegression","RandomForestRegressor","XGBoost","MLPRegressor","SVR","CatBoostRegressor","LightGBM"]#,"LinearForestRegressor"]
def CalculateMetrics(vmid, folder, name_columns):
    """
    Calculate various performance metrics for predictive models based on specified folder and VM ID.

    Parameters:
    ----------
    vmid : str
        The ID of the virtual machine for which the metrics are being calculated.
        
    folder : str
        The folder indicating the type of backtesting method used. This should be one of BT0, BT1, BT2, or BT3.
        
    name_columns : list of str
        The names of the columns in the data that will be used for metric calculations.
        
    Returns:
    -------
    tuple
        A tuple containing the following metrics:
        - error_mse_target : list
            Mean squared error between the target and predicted values.
        - error_mse_UB : list
            Mean squared error between the target and upper bound predictions.
        - picp_result : list
            Prediction interval coverage probability results.
        - pinc_result : list
            Prediction interval non-coverage results.
        - ACE : list
            Average coverage error results.
        - mpiw : list
            Mean prediction interval width results.
        - PINAW : list
            Prediction interval normalized average width results.
        - pinrw : list
            Prediction interval normalized width results.
        - score : list
            Custom scoring metrics.
        - runtime_train : ndarray
            The runtime of the training process.
        - runtime_predict : ndarray
            The runtime of the prediction process.
        - mae_target : list
            Mean absolute error between the target and predicted values.
        - mse_target : list
            Mean squared error between the target and predicted values.
        - rmse_target : list
            Root mean squared error between the target and predicted values.
        - mape_target : list
            Mean absolute percentage error between the target and predicted values.
        - r2_target : list
            R-squared values between the target and predicted values.
        - mae_UB : list
            Mean absolute error between the target and upper bound predictions.
        - mse_UB : list
            Mean squared error between the target and upper bound predictions.
        - rmse_UB : list
            Root mean squared error between the target and upper bound predictions.
        - mape_UB : list
            Mean absolute percentage error between the target and upper bound predictions.
        - r2_UB : list
            R-squared values between the target and upper bound predictions.
    
    Raises:
    ------
    ValueError
        If the `folder` parameter is not one of the expected values (BT0, BT1, BT2, BT3).
    
    Notes:
    -----
    The function loads data from specific file paths based on the folder type and computes various
    performance metrics by comparing predictions against the actual test data. Each metric serves
    a different purpose in assessing the model's performance.
    """

    if folder == BT0:  # without refit
        runtime_predict = np.load(f"./Results/{vmid}/{BT0}/runtime_UB__Prediction_using_without_refit_mae.npy", allow_pickle=True)
        runtime_train = np.load(f"./Results/{vmid}/{BT0}/runtime_UB__Train_using_without_refit_mae.npy", allow_pickle=True)
        prediction = glob.glob(f"./Results/{vmid}/{BT0}/predictions_*.xlsx")
        data_test = pd.read_excel(f"./Results/{vmid}/{folder}/data_test.xlsx")
    elif folder == BT1:  # forecaster_Fixed_mae
        runtime_predict = np.load(f"./Results/{vmid}/{BT1}/runtime_UB__Prediction_using_Fixed Backtesting_mae.npy", allow_pickle=True)
        runtime_train = np.load(f"./Results/{vmid}/{BT1}/runtime_UB__Train_using_Fixed Backtesting_mae.npy", allow_pickle=True)
        prediction = glob.glob(f"./Results/{vmid}/{BT1}/predictions_*.xlsx")
        data_test = pd.read_excel(f"./Results/{vmid}/{folder}/data_test.xlsx")
    elif folder == BT2:  # intermittent_refit
        runtime_predict = np.load(f"./Results/{vmid}/{BT2}/runtime_UB__Prediction_using_Intermittent_Refitting_mae.npy", allow_pickle=True)
        runtime_train = np.load(f"./Results/{vmid}/{BT2}/runtime_UB__Train_using_Intermittent_Refitting_mae.npy", allow_pickle=True)
        data_test = pd.read_excel(f"./Results/{vmid}/{folder}/data_test.xlsx")
        prediction = glob.glob(f"./Results/{vmid}/{BT2}/predictions_*.xlsx")
    elif folder == BT3:  # refit_increasing_trainingsize(fixed origin)
        runtime_predict = np.load(f"./Results/{vmid}/{BT3}/runtime_UB__Prediction_using_Refit_Increasing_Training Size_mae.npy", allow_pickle=True)
        runtime_train = np.load(f"./Results/{vmid}/{BT3}/runtime_UB__Train_using_Refit_Increasing_Training Size_mae.npy", allow_pickle=True)
        data_test = pd.read_excel(f"./Results/{vmid}/{folder}/data_test.xlsx")
        prediction = glob.glob(f"./Results/{vmid}/{BT3}/predictions_*.xlsx")
    else:
        # Handle unexpected folder value
        raise ValueError(f"Unexpected folder value: {folder}. Expected one of: {BT0}, {BT1}, {BT2}, {BT3}")

    # Initialize lists and proceed with calculations.
    predictions_models = []
    picp_result = []
    pinc_result = []
    ACE = []
    mpiw = []
    PINAW = []
    pinrw = []
    score = []
    error_mse_target = []
    error_mse_UB = []
    mae_target = []
    mse_target = []
    rmse_target = []
    mape_target = []
    r2_target = []

    mae_UB = []
    mse_UB = []
    rmse_UB = []
    mape_UB = []
    r2_UB = []

    for pth in prediction:
        if "data_test" in pth:
            continue
        else:  
            predict = pd.read_excel(pth)
            predictions_models.append(predict)
            predic_size = predict.shape[0]
            test_size = data_test.shape[0]
            if test_size > predic_size:
                data_test = data_test[:predic_size]
            else:
                data_test = data_test
            data_test_reset = data_test[name_columns].reset_index(drop=True)
            error_mse_target.append(mean_squared_error(data_test[name_columns], predict.iloc[:, 1]))  #######for prediction with prediction
            error_mse_UB.append(mean_squared_error(data_test[name_columns], predict.iloc[:, 3]))  #######for prediction with upper_bound
            picp_result.append(calculate_picp(data_test_reset, predict))
            pinc_result.append(calculate_pinc(predict, data_test_reset))
            ACE.append(calculate_picp(data_test_reset,predict) - calculate_pinc(predict, data_test_reset))
            mpiw.append(calculate_MPIW(predict))
            PINAW.append(calculate_PINAW(predict, R=2.0))
            pinrw.append(calculate_PINRW(predict, R=2.0))
            score.append(calculate_score(predict, data_test_reset, 0.2, 0.2))
            pred = predict.pred
            upper_bounds = predict.upper_bound
            mae_target.append(mean_absolute_error(data_test[name_columns], pred))
            mse_target.append(mean_squared_error(data_test[name_columns], pred))
            rmse_target.append(np.sqrt(mean_squared_error(data_test[name_columns], pred)))
            mape_target.append(mean_absolute_percentage_error(data_test[name_columns], pred))
            r2_target.append(r2_score(data_test[name_columns], pred))

            mae_UB.append(mean_absolute_error(data_test[name_columns], upper_bounds))
            mse_UB.append(mean_squared_error(data_test[name_columns], upper_bounds))
            rmse_UB.append(np.sqrt(mean_squared_error(data_test[name_columns], upper_bounds)))
            mape_UB.append(mean_absolute_percentage_error(data_test[name_columns], upper_bounds))
            r2_UB.append(r2_score(data_test[name_columns], upper_bounds))

    return (error_mse_target, error_mse_UB, picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime_train,runtime_predict, 
            mae_target, mse_target, rmse_target, mape_target, r2_target, mae_UB, mse_UB, rmse_UB, mape_UB, r2_UB)

def main(vmid, folder, name_columns):
    """
    Main function to calculate various metrics and generate styled dataframes for performance evaluation.

    Parameters:
    vmid (int or str): The identifier for the model or virtual machine.
    folder (str): The folder name that indicates the context of the data, 
                  e.g., "forecaster_in" or "forecaster_out".
    name_columns (list of str): A list containing the names of the columns to be used in the analysis.

    Returns:
    tuple: A tuple containing HTML representations of styled dataframes:
        - html1 (str): HTML representation of the first styled dataframe containing PICP, PINC, ACE, MPIW, PINAW, PINRW, score, and runtime metrics.
        - html1_ (str): HTML representation of an alternative styled dataframe with the same metrics.
        - html2 (str): HTML representation of the second styled dataframe containing MAE, MSE, RMSE, MAPE, R^2, and runtime metrics for the target.
        - html2_ (str): HTML representation of an alternative styled dataframe for the target metrics.
        - html3 (str): HTML representation of the third styled dataframe containing MAE, MSE, RMSE, MAPE, R^2, and runtime metrics for the upper bound (UB).
        - html3_ (str): HTML representation of an alternative styled dataframe for the upper bound metrics.
    
    This function calls several helper functions to compute metrics and generate styled dataframes,
    which are then converted to HTML format for presentation or further analysis.
    """
    # if folder in ["forecaster_in", "forecaster_out"]:
    error_mse_traget, error_mse_UB, picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime_train, runtime_predict, mae_target, mse_target, rmse_target, mape_target, r2_target,mae_UB, mse_UB, rmse_UB, mape_UB, r2_UB = CalculateMetrics(vmid, folder, name_columns)
    df_all1, styled_df1 = plot_dataframe(name_model,  picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime_train, runtime_predict)
    df_all1_, styled_df1_ = plot_dataframe_(name_model,  picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime_train, runtime_predict)
    
    df_all2, styled_df2 = plot_dataframe2(name_model, mae_target, mse_target, rmse_target, mape_target, r2_target, runtime_train, runtime_predict)
    df_all2_, styled_df2_ = plot_dataframe2_(name_model, mae_target, mse_target, rmse_target, mape_target, r2_target, runtime_train, runtime_predict)

    df_all3, styled_df3 = plot_dataframe2(name_model, mae_UB, mse_UB, rmse_UB, mape_UB, r2_UB, runtime_train, runtime_predict)
    df_all3_, styled_df3_ = plot_dataframe2_(name_model, mae_UB, mse_UB, rmse_UB, mape_UB, r2_UB, runtime_train, runtime_predict)
    
    html1 = styled_df1.to_html()
    html1_ = styled_df1_.to_html()
    html2 = styled_df2.to_html()
    html2_ = styled_df2_.to_html()

    html3 = styled_df3.to_html()
    html3_ = styled_df3_.to_html()
    
    return html1, html1_, html2, html2_, html3, html3_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot dataframe.")
    
    parser.add_argument('vmid', type=str, help='Name of the dataset (without .csv)')
    parser.add_argument('folder', type=str, help='Folder name (e.g., forecaster_in, forecaster_out, BT0, etc.)')
    parser.add_argument('name_columns', type=str, help='Name of the column to forecast')

    args = parser.parse_args()

    html1, html1_,html2,html2_,html3,html3_ = main(args.vmid, args.folder, args.name_columns)
    
    with open(f'./html/{args.folder}_PI_metric.html', 'w', encoding='utf-8') as f:
        f.write(html1)
    with open(f'./html/{args.folder}_PI_metric_.html', 'w', encoding='utf-8') as f:
        f.write(html1_)    
    
    with open(f'./html/{args.folder}_classic_metric_target.html', 'w', encoding='utf-8') as f:
        f.write(html2)
    with open(f'./html/{args.folder}_classic_metric_target_.html', 'w', encoding='utf-8') as f:
        f.write(html2_)

    with open(f'./html/{args.folder}_classic_metric_UB.html', 'w', encoding='utf-8') as f:
        f.write(html3)
    with open(f'./html/{args.folder}_classic_metric_UB_.html', 'w', encoding='utf-8') as f:
        f.write(html3_)    
    
