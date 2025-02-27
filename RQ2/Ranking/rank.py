import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataframe import plot_dataframe, plot_dataframe2
import os
import argparse
pd.set_option('display.max_colwidth', None)
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")
from great_tables import GT
from df_result import CalculateMetrics
name_columns = "avgcpu"
name_model = ["LinearRegression","RandomForestRegressor","XGBoost","MLPRegressor","SVR","CatBoostRegressor","LightGBM"]#,"LinearForestRegressor"]
# Function to process DataFrame: calculate runtime in seconds and ranks
import pandas as pd

def process_df(df, metric):
    """
    Processes the input DataFrame by converting the 'Runtime_predict [ms]' column 
    to total seconds and ranking the models based on a specified metric.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be processed.
                       It must contain a column named 'Runtime_predict [ms]' 
                       and a column corresponding to the specified metric.
    metric (str): The name of the metric column used for ranking. 
                  Lower values of this metric are considered better.

    Returns:
    pd.DataFrame: The processed DataFrame with two additional columns:
                  - 'Runtime_predict [ms]': Converted runtime in seconds.
                  - 'Rank': The rank of each model based on the specified metric, 
                             where a lower metric value results in a higher rank (closer to 1).
    """

    df['Runtime_predict [ms]'] = pd.to_timedelta(df['Runtime_predict [ms]']).dt.total_seconds()
    df['Rank'] = df[metric].rank()  # Rank models based on the specified metric
    return df


def create_performance_table(df_vm, metric, folder, Type):
    """
    Creates a performance ranking table for machine learning models based on specified metrics.

    This function processes a list of DataFrames containing performance metrics for various models,
    aggregates their ranks, and counts how often each model appears in the top 1 and top 3 rankings.
    Finally, it generates an HTML table summarizing this information.

    Parameters:
    ----------
    df_vm : list of pandas.DataFrame
        A list of DataFrames, each containing model performance data, including model names and ranks.

    metric : str
        The metric used to evaluate the models (e.g., accuracy, precision, etc.).

    folder : str
        The folder name used for saving the output HTML file.

    Type : str
        The type of analysis or experiment being performed, used in the table title.

    Returns:
    -------
    None
        The function saves the generated performance table as an HTML file in the specified folder.
    """

    data = []
    top3_counts = {}
    top1_counts = {}

    for df in df_vm:
        df = process_df(df, metric)
        
        df_sorted = df.sort_values(by='Rank', ascending=True)
        
        top3_models = df_sorted['Models'].head(3).tolist()
        top1_model = df_sorted['Models'].head(1).tolist()[0]
        
        # Count top 3 models
        for model in top3_models:
            if model in top3_counts:
                top3_counts[model] += 1
            else:
                top3_counts[model] = 1
        
        # Count top 1 model
        if top1_model in top1_counts:
            top1_counts[top1_model] += 1
        else:
            top1_counts[top1_model] = 1

        for model in df['Models'].unique():
            model_data = df[df['Models'] == model]
            data.append({
                'Model': model,
                'Avg_Rank': model_data['Rank'].mean(),
                'Median_Rank': model_data['Rank'].median(),
                'Top3_Count': top3_counts.get(model, 0),
                'Top1_Count': top1_counts.get(model, 0)
            })

    df = pd.DataFrame(data)

    # Aggregate data
    df_aggregated = df.groupby('Model', as_index=False).agg({
        'Avg_Rank': 'mean',
        'Median_Rank': 'median',
        'Top3_Count': 'max',
        'Top1_Count': 'max'
    })

    # Sort by Top3_Count in descending order
    df_aggregated = df_aggregated.sort_values(by='Top3_Count', ascending=False)

    # Create the GT table
    gt_table = (
        GT(df_aggregated)
        .tab_header(title=f"Performance Ranking of Models for {folder} (Metric: {metric}, Type: {Type})")
        .fmt_number(columns=["Avg_Rank", "Median_Rank"], decimals=2)
        .fmt_number(columns=["Top3_Count", "Top1_Count"], decimals=0)
        .cols_label(
            Model="Model",
            Avg_Rank="Average Rank",
            Median_Rank="Median Rank",
            Top3_Count="# in Top 3 Models",
            Top1_Count="# in Top 1 Model"
        )
    )

    # Save the table as HTML
    html_content = gt_table.as_raw_html()
    with open(f"./rank_plot/{folder}_performance_table_{metric}_{Type}.html", "w", encoding="utf-8") as file:
        file.write(html_content)

def create_performance_table___________(df_vm, metric, folder, Type):
    """
    Create a performance ranking table for models based on specified metrics.

    This function processes a list of DataFrames containing performance metrics for different models,
    aggregates the results, and generates an HTML table displaying the performance rankings.

    Parameters:
    df_vm (list of pd.DataFrame): A list of DataFrames, each containing performance metrics for various models.
    metric (str): The name of the metric used to evaluate model performance, e.g., 'Accuracy', 'F1 Score', etc.
    folder (str): The folder name to be used in the title of the output table and the file name for saving the HTML.
    Type (str): A string representing the type of analysis or model, which will be included in the table title.

    Output:
    This function does not return any values. Instead, it generates an HTML file named 
    '{folder}_performance_table_{metric}_{Type}.html' in the './rank_plot/' directory, containing
    a formatted performance table of models based on the specified metric.
    
    The generated table includes the following columns:
    - Model: The name of the model.
    - # Best: The count of the best performance metric for each model.
    - Average Rank: The average rank of the model based on the specified metric.
    - Median Rank: The median rank of the model based on the specified metric.
    - # in Top 3 Models: The count of times the model appears in the top 3 rankings.

    Example:
    create_performance_table___________(dataframes, 'Accuracy', 'Model_Comparison', 'Final')
    """

    data = []
    top3_counts = {} 

    for df in df_vm:
        df = process_df(df, metric)
        
        df_sorted = df.sort_values(by='Rank', ascending=True)
        
        top3_models = df_sorted['Models'].head(3).tolist()
        
        for model in top3_models:
            if model in top3_counts:
                top3_counts[model] += 1
            else:
                top3_counts[model] = 1

        for model in df['Models'].unique():
            model_data = df[df['Models'] == model]
            data.append({
                'Model': model,
                'Best_Count': model_data[metric].count(),
                'Avg_Rank': model_data['Rank'].mean(),
                'Median_Rank': model_data['Rank'].median(),
                'Top3_Count': top3_counts.get(model, 0)  
            })

    df = pd.DataFrame(data)

    df_aggregated = df.groupby('Model', as_index=False).agg({
        'Best_Count': 'sum',
        'Avg_Rank': 'mean',
        'Median_Rank': 'median',
        'Top3_Count': 'max' 
    })

    gt_table = (
        GT(df_aggregated)
        .tab_header(title=f"Performance Ranking of Models for {folder} (Metric: {metric}, Type: {Type})")
        .fmt_number(columns=["Avg_Rank", "Median_Rank"], decimals=2)
        .fmt_number(columns=["Best_Count", "Top3_Count"], decimals=0)
        .cols_label(
            Model="Model",
            Best_Count="# Best",
            Avg_Rank="Average Rank",
            Median_Rank="Median Rank",
            Top3_Count="# in Top 3 Models"
        )
    )

    html_content = gt_table.as_raw_html()
    with open(f"./rank_plot/{folder}_performance_table_{metric}_{Type}.html", "w", encoding="utf-8") as file:
        file.write(html_content)

def rank_model(df_vm, metric, folder, Type):
    """
    Ranks machine learning models based on specified metrics and visualizes the results.
    
    This function processes multiple DataFrames containing model performance metrics,
    calculates average ranks and runtimes, identifies top models, and generates
    visualizations to compare models based on their ranks and run times.
    
    Parameters:
    df_vm (list of pd.DataFrame): A list of DataFrames, each containing model performance metrics.
                                   Each DataFrame should include columns: 'Models', the specified 'metric',
                                   and 'Runtime_predict [ms]'.
    metric (str): The metric to rank the models by (e.g., 'MAE', 'RMSE').
    folder (str): The name of the folder for saving output plots.
    Type (str): The type of analysis or method being used (for labeling purposes in plots).
    
    Returns:
    None: The function generates and saves plots but does not return any values.
    
    Notes:
    - The function visualizes the average rank versus runtime in a scatter plot, 
      highlighting the optimal model with a star marker.
    - It also plots bar charts for the absolute frequencies of the top 1 and top 3 models.
    - The Y-axis of the scatter plot is on a logarithmic scale to better visualize runtime differences.
    - The function assumes the existence of a helper function `process_df` that processes individual DataFrames 
      to calculate ranks based on the specified metric.
    """
    
    DF = []
    concatenated_models_list = []
    for df in df_vm:
        DF.append(df[["Models", metric, "Runtime_predict [ms]"]])
        df_sorted = df.sort_values(by=metric, ascending=False)
        # Concatenate the sorted-by-MAE model names into one string, so that
        # you can group by it as a key later
        concatenated_models = ','.join(df_sorted['Models'].tolist())
        concatenated_models_list.append(concatenated_models)
        # Calculate frequencies for top N models in each DataFrame
    top_n = 3
    # Union everything into a single dataframe
    DF_T = []
    Top_MOdels = []
    Top_MOdel = []
    for dff in DF:
        DF_T.append(process_df(dff, metric))
        Top_MOdels.append(dff.nsmallest(top_n, metric)['Models'])
        Top_MOdel.append(dff.nsmallest(1, metric)['Models'])
    combined = pd.concat(DF_T)
    # print(combined)  
    # Compute average rank and runtime for each model
    agg_df = combined.groupby('Models', as_index=False).agg({
        'Rank': 'mean',
        'Runtime_predict [ms]': 'mean'
    })
    top_models = pd.concat(Top_MOdels)
    top_model = pd.concat(Top_MOdel)
    # Add absolute frequencies to agg_df
    agg_df['Frequency'] = agg_df['Models'].map(top_models.value_counts()).fillna(0)
    agg_df['Circle_Size'] = agg_df['Frequency'] * 300  # Scale circle sizes

    # Identify optimal model (lowest rank and runtime)
    optimal_model = agg_df.loc[agg_df['Rank'] == agg_df['Rank'].min()]

    # Tree-based models
    tree_based_models = ['Random Forest', 'XGBoost', 'CatBoostRegressor', 'LGBMRegressor']

    # Visualization
    plt.figure(figsize=(10, 7))
    sns.set(style="whitegrid")

    # Scatter plot with sizes based on frequencies
    for _, row in agg_df.iterrows():
        color = 'lightgreen' if row['Models'] in tree_based_models else 'blue'
        plt.scatter(row['Rank'], row['Runtime_predict [ms]'], 
                    s=row['Circle_Size'], alpha=0.5, edgecolors='k', color=color)

    # Add star marker for the optimal model
    plt.scatter(1, 1, 
                s=200, color='gold', marker='*', edgecolors='k', label="Optimal Model")

    # Annotate each model
    for _, row in agg_df.iterrows():
        plt.text(row['Rank'], row['Runtime_predict [ms]'], row['Models'], fontsize=9, ha='right', va='center')

    # Logarithmic scale for the Y-axis
    plt.yscale('log')
    plt.yticks([10**0, 10**1, 10**2, 10**3, 10**4], ['$1$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    plt.xlabel('Average Rank (Lower is Better)', fontsize=12)
    plt.ylabel('Runtime_predict [ms] (Log Scale)', fontsize=12)
    plt.title('Visualization of Average Rank vs Runtime with Frequencies', fontsize=14)
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markeredgecolor='k', markersize=8, label='Tree-Based Models'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markeredgecolor='k', markersize=8, label='Other Models'),
        Line2D([0], [0], marker='*', color='gold', markeredgecolor='k', markersize=12, label='Optimal Model')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"./rank_plot/{folder}_Visualization_Average_Rank_{metric}_{Type}.png")
    
    absolute_frequency_top3 = top_models.value_counts()
    absolute_frequency_top1 = top_model.value_counts()
    all_models = absolute_frequency_top3.index.union(absolute_frequency_top1.index)
    plot_df = pd.DataFrame(index=all_models)
    plot_df['Top 3 Frequency'] = absolute_frequency_top3
    plot_df['Top 1 Frequency'] = absolute_frequency_top1
    plot_df = plot_df.fillna(0)

    # Create a list of all unique model names
    all_models = combined['Models'].unique()

    # Update the plot DataFrame to include all models, filling missing frequencies with 0
    plot_df = pd.DataFrame({'Models': all_models}).set_index('Models')
    plot_df['Top 3 Frequency'] = absolute_frequency_top3
    plot_df['Top 1 Frequency'] = absolute_frequency_top1
    plot_df = plot_df.fillna(0)  # Fill NaN values with 0
    plot_df = plot_df.sort_values(by='Top 3 Frequency', ascending=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.4
    x = np.arange(len(plot_df))

    # Bar for top 3 frequencies
    ax.bar(x - bar_width/2, plot_df['Top 3 Frequency'], bar_width, color='black', label='Top 3 Frequency')

    # Bar for top 1 frequencies
    ax.bar(x + bar_width/2, plot_df['Top 1 Frequency'], bar_width, color='gray', edgecolor='w', hatch='//', label='Top 1 Frequency')

    # Customizing the chart
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=45, ha='right')
    ax.set_ylabel('Absolute Frequency')
    ax.set_xlabel('Models')
    if folder == "Without Refit":
        folder_ = "Backtesting without refit"
    elif folder == "Fixed Backtesting":
        folder_ = "Backtesting with refit and fixed training size (rolling origin)"
    elif folder == "Intermittent Refitting":
        folder_ = "Backtesting with intermittent refit" 
    elif folder == "Refit Increasing Training Size":
        folder_ = "Backtesting with refit and increasing training size (fixed origin)"   
    plt.title(f"Absolute Frequency of Forecast Models in Rankings {len(name_model)} predictive models
 over {len(DF_T)} VMs CPU workload sample \n for method:{folder_}\n Based on metric:{metric}", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"./rank_plot/{folder}_combined_models_{metric}_{Type}.png")


def main(folder, name_columns, metric, Type):
    """
    Main function to process virtual machine results and calculate various metrics.

    This function iterates through the results of virtual machines stored in a specified folder,
    calculates metrics using the `CalculateMetrics` function, and generates dataframes using
    the `plot_dataframe` and `plot_dataframe2` functions. It also ranks models based on the 
    specified metric and type, and creates performance tables.

    Parameters:
    folder (str): The directory where the results are stored. 
                  It should not be ".ipynb_checkpoints".
    name_columns (list): A list of column names to be used in the data processing.
    metric (str): The performance metric to be used for ranking models. 
                  Valid options include 'PICP', 'PINC', 'ACE', 'MPIW', 'PINAW', 'PINRW', 'score', 'MSE', 
                  'RMSE', 'MAPE', and 'R2'.
    Type (str): The type of metric to be processed. 
                It can be "PI" for Prediction Intervals, "target" for target metrics, 
                or "upper" for upper bound metrics.

    Returns:
    None: This function does not return any value. It performs operations and generates outputs 
          such as rankings and performance tables directly.

    Note:
    The function handles three types of metrics: 'PI', 'target', and 'upper', and calls the appropriate
    ranking and table creation functions based on the specified metric and type.
    """
    
    df_vm_PI = []
    df_vm_target = []
    df_vm_upper = []
    
    ListDire = sorted(os.listdir("./Results/"))
    
    for vmid in ListDire:
        if folder == ".ipynb_checkpoints":
            pass
        else:  
            error_mse_traget, error_mse_UB, picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime_train, runtime_predict, mae_target, mse_target, rmse_target, mape_target, r2_target, mae_UB, mse_UB, rmse_UB, mape_UB, r2_UB = CalculateMetrics(vmid, folder, name_columns)
            
            df_all1, styled_df1 = plot_dataframe(name_model, picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime_train, runtime_predict)
            df_vm_PI.append(df_all1)
            
            df_all2, styled_df2 = plot_dataframe2(name_model, mae_target, mse_target, rmse_target, mape_target, r2_target, runtime_train, runtime_predict)
            df_vm_target.append(df_all2)
            
            df_all3, styled_df3 = plot_dataframe2(name_model, mae_UB, mse_UB, rmse_UB, mape_UB, r2_UB, runtime_train, runtime_predict)
            df_vm_upper.append(df_all3)
            
            if metric in ['PICP', 'PINC', 'ACE', 'MPIW', "PINAW", 'PINRW', 'score'] and Type == "PI":
                rank_model(df_vm_PI, metric, folder, "PI")
                create_performance_table(df_vm_PI, metric, folder, "PI")

            elif metric in [metric, 'MSE', 'RMSE', 'MAPE', "R2"] and Type == "target":    
                rank_model(df_vm_target, metric, folder, "target")
                create_performance_table(df_vm_target, metric, folder, "target")
                
            elif metric in [metric, 'MSE', 'RMSE', 'MAPE', "R2"] and Type == "upper":    
                rank_model(df_vm_upper, metric, folder, "upper")
                create_performance_table(df_vm_upper, metric, folder, "upper")
                
            else:
                print("it isn't True")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="plot dataframe.")
    
  parser.add_argument('folder', type=str, help='Folder name (e.g., forecaster_in, forecaster_out, BT0, etc.)')
  parser.add_argument('name_columns', type=str, help='Name of the column to forecast')
  parser.add_argument('metric', type=str, help='Name of the metric')
  parser.add_argument('Type', type=str, help='type of the measurement')



  args = parser.parse_args()
  main( args.folder, args.name_columns,args.metric,args.Type)
