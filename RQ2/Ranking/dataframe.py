import pandas as pd
import seaborn as sns


def plot_dataframe(name_model, picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime_train, runtime_predict):
    """
    Creates a DataFrame to summarize model performance metrics and applies conditional formatting for better visualization.

    Parameters:
    - name_model (list of str): Names of the models to be included in the DataFrame.
    - picp_result (list of float): PICP results for each model.
    - pinc_result (list of float): PINC results for each model.
    - ACE (list of float): ACE results for each model.
    - mpiw (list of float): MPIW results for each model.
    - PINAW (list of float): PINAW results for each model.
    - pinrw (list of float): PINRW results for each model.
    - score (list of float): Score for each model (can be negative).
    - runtime_train (list of float): Training runtime for each model in milliseconds.
    - runtime_predict (list of float): Prediction runtime for each model in milliseconds.

    Returns:
    - df_all (DataFrame): A DataFrame containing the summarized metrics for all models.
    - styled_df (Styler): A styled DataFrame with conditional formatting applied for visual clarity.
    """
    
    df_all = pd.DataFrame(columns=['Models', 'PICP', 'PINC', 'ACE', 'MPIW', 'PINAW', 'PINRW', 
                                   'score', 'Runtime_train [ms]', 'Runtime_predict [ms]'])
    
    for i in range(len(name_model)):
        numb1 = len(df_all) + 1
        df_all.loc[numb1] = [f"{name_model[i]}",
                             picp_result[i],
                             pinc_result[i],
                             ACE[i],
                             mpiw[i],
                             PINAW[i],
                             pinrw[i],
                             score[i],
                             runtime_train[i],
                             runtime_predict[i]]

    def color_negative_red(val):
        color = 'red' if val < 0 else 'black'
        return 'color: %s' % color

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: cyan; font-weight: bold' if v else '' for v in is_max]

    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: yellow; font-weight: bold' if v else '' for v in is_min]

    def highlight_mean_greater(s):
        is_max = s > s.mean()
        return ['background-color: green' if i else 'background-color: velvet' for i in is_max]

    # Apply formatting using Styler
    styled_df = df_all.style.\
    apply(lambda x: ['color: red' if v < 0 else 'color: black' for v in x], subset=['score']).\
    apply(highlight_max, subset=['MPIW', 'PINAW']).\
    apply(highlight_min, subset=['PICP', 'PINC', 'ACE']).\
    apply(highlight_min, subset=['ACE']).\
    apply(highlight_mean_greater, subset=['score']).\
    apply(highlight_min, subset=['Runtime_train [ms]']).\
    apply(highlight_min, subset=['Runtime_predict [ms]']).\
    apply(highlight_mean_greater, subset=['Runtime_train [ms]'])

    # Create a background gradient for enhanced visualization
    cm = sns.color_palette("coolwarm", as_cmap=True)
    styled_df2 = df_all.style.background_gradient(cmap=cm)
    
    return df_all, styled_df

import pandas as pd
import seaborn as sns

def plot_dataframe_(name_model, picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime_train, runtime_predict):
    """
    Create a DataFrame summarizing model performance metrics and apply styling for visualization.

    Parameters:
    ----------
    name_model : list of str
        A list containing the names of the models being evaluated.
        
    picp_result : list of float
        A list of PICP (Prediction Interval Coverage Probability) results for each model.
        
    pinc_result : list of float
        A list of PINC (Prediction Interval Non-Coverage) results for each model.
        
    ACE : list of float
        A list of Average Coverage Error values for each model.
        
    mpiw : list of float
        A list of Mean Prediction Interval Widths for each model.
        
    PINAW : list of float
        A list of Prediction Interval Non-Average Widths for each model.
        
    pinrw : list of float
        A list of Prediction Interval Non-Ratio Widths for each model.
        
    score : list of float
        A list of overall performance scores for each model.
        
    runtime_train : list of float
        A list of training runtimes (in milliseconds) for each model.
        
    runtime_predict : list of float
        A list of prediction runtimes (in milliseconds) for each model.

    Returns:
    -------
    df_all : DataFrame
        A pandas DataFrame containing the summarized performance metrics for all models.
        
    styled_df2 : Styler
        A pandas Styler object for the DataFrame, with a background gradient applied for better visualization.
    """
    
    df_all = pd.DataFrame(columns=['Models',  'PICP', 'PINC', 'ACE', 'MPIW', "PINAW", 'PINRW',
                                   'score', 'Runtime_train [ms]', 'Runtime_predict [ms]'])
    for i in range(len(name_model)):
        numb1 = len(df_all) + 1
        df_all.loc[numb1] = [f"{name_model[i]}",
                             picp_result[i],
                             pinc_result[i],
                             ACE[i],
                             mpiw[i],
                             PINAW[i],
                             pinrw[i],
                             score[i],
                             runtime_train[i],
                             runtime_predict[i]]

    # Apply formatting using Styler
    cm = sns.color_palette("coolwarm", as_cmap=True)

    styled_df2 = df_all.style.\
    background_gradient(cmap=cm)

    return df_all, styled_df2


def plot_dataframe2(name_model,mae,mse,rmse,mape,r2,runtime_train,runtime_predict):
    """
    Creates a DataFrame to summarize model evaluation metrics and applies styling for visualization.

    This function takes in various performance metrics for different models, constructs a DataFrame,
    and applies conditional formatting to highlight important values such as maximum, minimum,
    and mean comparisons. The styling helps in visually analyzing the performance of the models.

    Parameters:
    - name_model (list of str): A list containing the names of the models.
    - mae (list of float): A list containing Mean Absolute Error values corresponding to the models.
    - mse (list of float): A list containing Mean Squared Error values corresponding to the models.
    - rmse (list of float): A list containing Root Mean Squared Error values corresponding to the models.
    - mape (list of float): A list containing Mean Absolute Percentage Error values corresponding to the models.
    - r2 (list of float): A list containing R² scores corresponding to the models.
    - runtime_train (list of float): A list containing training runtimes (in milliseconds) for each model.
    - runtime_predict (list of float): A list containing prediction runtimes (in milliseconds) for each model.

    Returns:
    - df_all (DataFrame): A pandas DataFrame containing the model names and their corresponding metrics.
    - styled1_df2 (Styler): A pandas Styler object for the DataFrame with conditional formatting applied.
    """
    df_all = pd.DataFrame(columns=['Models','MAE', 'MSE', 'RMSE', 'MAPE', "R\u00b2 score",'Runtime_train [ms]', 'Runtime_predict [ms]'])
    for i in range(len(name_model)):
        numb1 = len(df_all) + 1
        df_all.loc[numb1] = [f"{name_model[i]}",
                               mae[i],
                               mse[i],
                               rmse[i],
                               mape[i],
                               r2[i],
                              runtime_train[i],
                              runtime_predict[i]
                               ]


    def color_negative_red(val):
        color = 'red' if val < 0 else 'black'
        return 'color: %s' % color

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: cyan; font-weight: bold' if v else '' for v in is_max]

    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: yellow; font-weight: bold' if v else '' for v in is_min]

    def highlight_mean_greater(s):
        is_max = s > s.mean()
        return ['background-color: green' if i else 'background-color: velvet' for i in is_max]

    # Apply formatting using Styler
    styled1_df2 = df_all.style.\
      map(color_negative_red, subset=['R\u00b2 score']).\
      apply(   highlight_max,           subset=['R\u00b2 score']).\
      apply(   highlight_min,           subset=['MAE', 'MSE','RMSE','MAPE']).\
      apply(   highlight_min,           subset=['Runtime_train [ms]']).\
      apply(   highlight_min,           subset=['Runtime_predict [ms]']).\
      apply(   highlight_mean_greater,  subset=['Runtime_train [ms]'])

    return df_all, styled1_df2


def plot_dataframe2_(name_model, mae, mse, rmse, mape, r2, runtime_train, runtime_predict):
    """
    Create a DataFrame to summarize model performance metrics and return a styled version of the DataFrame.

    Parameters:
    ----------
    name_model : list of str
        A list containing the names of the models being evaluated.
        
    mae : list of float
        A list containing the Mean Absolute Error (MAE) values for each model.
        
    mse : list of float
        A list containing the Mean Squared Error (MSE) values for each model.
        
    rmse : list of float
        A list containing the Root Mean Squared Error (RMSE) values for each model.
        
    mape : list of float
        A list containing the Mean Absolute Percentage Error (MAPE) values for each model.
        
    r2 : list of float
        A list containing the R² (R-squared) scores for each model.
        
    runtime_train : list of float
        A list containing the training runtime in milliseconds for each model.
        
    runtime_predict : list of float
        A list containing the prediction runtime in milliseconds for each model.

    Returns:
    -------
    df_all : pandas.DataFrame
        A DataFrame containing the performance metrics for each model.
        
    styled2_df2 : pandas.io.formats.style.Styler
        A styled version of the DataFrame with a background gradient for better visualization.

    Notes:
    -----
    This function uses the seaborn color palette to create a gradient background for the styled DataFrame.
    The DataFrame is constructed by iterating over the input lists, and the index is adjusted to start from 1.
    """
    df_all = pd.DataFrame(columns=['Models', 'MAE', 'MSE', 'RMSE', 'MAPE', "R\u00b2 score", 'Runtime_train [ms]', 'Runtime_predict [ms]'])
    for i in range(len(name_model)):
        numb1 = len(df_all) + 1
        df_all.loc[numb1] = [f"{name_model[i]}",
                               mae[i],
                               mse[i],
                               rmse[i],
                               mape[i],
                               r2[i],
                               runtime_train[i],
                               runtime_predict[i]
                               ]
    # Apply formatting using Styler2
    cm = sns.color_palette("coolwarm", as_cmap=True)

    styled2_df2 = df_all.style.\
    background_gradient(cmap=cm)

    return df_all, styled2_df2


def plot_dataframe3(name_model, picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime):
    """
    Creates a styled Pandas DataFrame summarizing model performance metrics.

    This function takes various performance metrics of different models, 
    compiles them into a DataFrame, and applies conditional formatting 
    to visually differentiate metrics based on certain criteria.

    Parameters:
    ----------
    name_model : list of str
        A list containing the names of the models.
    
    picp_result : list of float
        A list containing the PICP results for each model.
    
    pinc_result : list of float
        A list containing the PINC results for each model.
    
    ACE : list of float
        A list containing the ACE values for each model.
    
    mpiw : list of float
        A list containing the MPIW values for each model.
    
    PINAW : list of float
        A list containing the PINAW values for each model.
    
    pinrw : list of float
        A list containing the PINRW values for each model.
    
    score : list of float
        A list containing the scores for each model.
    
    runtime : list of float
        A list containing the runtime (in milliseconds) for each model.

    Returns:
    -------
    tuple
        A tuple containing:
        - df_all : pd.DataFrame
            A DataFrame with all the model metrics.
        - styled_df : pd.io.formats.style.Styler
            A styled version of the DataFrame with conditional formatting applied.
    
    Notes:
    -----
    - The function highlights maximum values in 'MPIW' and 'PINAW' columns 
      with a cyan background and bold font.
    - The minimum values in 'PICP', 'PINC', 'ACE', and 'Runtime [ms]' 
      columns are highlighted with a yellow background and bold font.
    - Scores below zero are colored red, while non-negative scores are 
      colored black.
    - If the score is greater than the mean score, its background is 
      shaded green; otherwise, it is shaded velvet.
    - A color gradient is applied to the entire DataFrame using a coolwarm 
      colormap.
    """
    df_all = pd.DataFrame(columns=['Models',  'PICP', 'PINC', 'ACE', 'MPIW', "PINAW", 'PINRW',
                                   'score', 'Runtime [ms]'])
    
    for i in range(len(name_model)):
        numb1 = len(df_all) + 1
        df_all.loc[numb1] = [f"{name_model[i]}",
                             picp_result[i],
                             pinc_result[i],
                             ACE[i],
                             mpiw[i],
                             PINAW[i],
                             pinrw[i],
                             score[i],
                             runtime[i],
                             ]

    def color_negative_red(val):
        color = 'red' if val < 0 else 'black'
        return 'color: %s' % color

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: cyan; font-weight: bold' if v else '' for v in is_max]

    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: yellow; font-weight: bold' if v else '' for v in is_min]

    def highlight_mean_greater(s):
        is_max = s > s.mean()
        return ['background-color: green' if i else 'background-color: velvet' for i in is_max]

    # Apply formatting using Styler
    styled_df = df_all.style.\
    apply(lambda x: ['color: red' if v < 0 else 'color: black' for v in x], subset=['score']).\
    apply(highlight_max, subset=['MPIW', 'PINAW']).\
    apply(highlight_min, subset=['PICP', 'PINC', 'ACE']).\
    apply(highlight_min, subset=['ACE']).\
    apply(highlight_mean_greater, subset=['score']).\
    apply(highlight_min, subset=['Runtime [ms]'])

    # Apply formatting using Styler2
    cm = sns.color_palette("coolwarm", as_cmap=True)

    styled_df2 = df_all.style.\
    background_gradient(cmap=cm)
    
    return df_all, styled_df

def plot_dataframe3_(name_model, picp_result, pinc_result, ACE, mpiw, PINAW, pinrw, score, runtime):
    """
    Creates a DataFrame to summarize the performance metrics of various models and applies a gradient background for styling.

    Parameters:
    - name_model (list of str): A list containing the names of the models.
    - picp_result (list of float): A list of PICP results corresponding to each model.
    - pinc_result (list of float): A list of PINC results corresponding to each model.
    - ACE (list of float): A list of ACE values corresponding to each model.
    - mpiw (list of float): A list of MPIW values corresponding to each model.
    - PINAW (list of float): A list of PINAW values corresponding to each model.
    - pinrw (list of float): A list of PINRW values corresponding to each model.
    - score (list of float): A list of scores corresponding to each model.
    - runtime (list of float): A list of runtime values (in milliseconds) corresponding to each model.

    Returns:
    - df_all (DataFrame): A pandas DataFrame containing the performance metrics for each model.
    - styled_df2 (Styler): A pandas Styler object with a gradient background applied to the DataFrame for visual enhancement.

    Example:
    >>> df, styled = plot_dataframe3_(model_names, picp, pinc, ace, mpiw, pinaw, pinrw, scores, runtimes)
    >>> styled.to_html()  # to render the styled DataFrame in HTML format
    """
    df_all = pd.DataFrame(columns=['Models',  'PICP','PINC', 'ACE', 'MPIW', "PINAW", 'PINRW',
                                   'score','Runtime [ms]'])
    for i in range(len(name_model)):
        numb1 = len(df_all) + 1
        df_all.loc[numb1] = [f"{name_model[i]}",
                             picp_result[i],
                             pinc_result[i],
                             ACE[i],
                             mpiw[i],
                             PINAW[i],
                             pinrw[i],
                             score[i],
                             runtime[i],]

    # Apply formatting using Styler2
    cm = sns.color_palette("coolwarm", as_cmap=True)

    styled_df2 = df_all.style.\
    background_gradient(cmap=cm)

    return df_all, styled_df2


def plot_dataframe4(name_model, mae, mse, rmse, mape, r2, runtime):
    """
    Creates a DataFrame to summarize the performance metrics of different models and applies 
    custom styling to highlight specific values.

    Parameters:
    ----------
    name_model : list of str
        A list containing the names of the models being compared.
    
    mae : list of float
        A list containing the Mean Absolute Error (MAE) values for each model.
    
    mse : list of float
        A list containing the Mean Squared Error (MSE) values for each model.
    
    rmse : list of float
        A list containing the Root Mean Squared Error (RMSE) values for each model.
    
    mape : list of float
        A list containing the Mean Absolute Percentage Error (MAPE) values for each model.
    
    r2 : list of float
        A list containing the R² (coefficient of determination) scores for each model.
    
    runtime : list of float
        A list containing the runtime in milliseconds for each model's execution.

    Returns:
    -------
    df_all : pandas.DataFrame
        A DataFrame containing the performance metrics of the models with columns for model names,
        MAE, MSE, RMSE, MAPE, R² score, and runtime.
    
    styled1_df2 : pandas.io.formats.style.Styler
        A styled version of the DataFrame that includes conditional formatting:
        - R² scores less than 0 are colored red.
        - Maximum values in the R² score column are highlighted in cyan.
        - Minimum values in the MAE, MSE, RMSE, MAPE, and Runtime columns are highlighted in yellow.
        - Values greater than the mean in the specified columns are highlighted in green, while others are in velvet.

    Notes:
    -----
    This function is useful for visually comparing the performance of different models using 
    important regression metrics. The conditional formatting helps to quickly identify 
    the best and worst performing models in each metric.
    """
    df_all = pd.DataFrame(columns=['Models','MAE', 'MSE', 'RMSE', 'MAPE', "R\u00b2 score",'Runtime [ms]'])
    for i in range(len(name_model)):
        numb1 = len(df_all) + 1
        df_all.loc[numb1] = [f"{name_model[i]}",
                               mae[i],
                               mse[i],
                               rmse[i],
                               mape[i],
                               r2[i],
                               runtime[i],
                               ]

    def color_negative_red(val):
        color = 'red' if val < 0 else 'black'
        return 'color: %s' % color

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: cyan; font-weight: bold' if v else '' for v in is_max]

    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: yellow; font-weight: bold' if v else '' for v in is_min]

    def highlight_mean_greater(s):
        is_max = s > s.mean()
        return ['background-color: green' if i else 'background-color: velvet' for i in is_max]

    # Apply formatting using Styler
    styled1_df2 = df_all.style.\
      map(color_negative_red, subset=['R\u00b2 score']).\
      apply(highlight_max, subset=['R\u00b2 score']).\
      apply(highlight_min, subset=['MAE', 'MSE', 'RMSE', 'MAPE']).\
      apply(highlight_min, subset=['Runtime [ms]'])

    return df_all, styled1_df2



def plot_dataframe4_(name_model, mae, mse, rmse, mape, r2, runtime):
    """
    Create a DataFrame to summarize model performance metrics and return a styled DataFrame for visualization.

    This function takes in the performance metrics of different models, creates a pandas DataFrame,
    and styles it with a background gradient for better visualization. The metrics include Mean Absolute Error (MAE),
    Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE),
    R-squared (R²) score, and the runtime in milliseconds.

    Parameters:
    - name_model (list of str): A list containing the names of the models.
    - mae (list of float): A list containing the Mean Absolute Error values for each model.
    - mse (list of float): A list containing the Mean Squared Error values for each model.
    - rmse (list of float): A list containing the Root Mean Squared Error values for each model.
    - mape (list of float): A list containing the Mean Absolute Percentage Error values for each model.
    - r2 (list of float): A list containing the R-squared values for each model.
    - runtime (list of float): A list containing the runtime in milliseconds for each model.

    Returns:
    - df_all (DataFrame): A pandas DataFrame containing the model names and their corresponding performance metrics.
    - styled2_df2 (Styler): A styled pandas DataFrame with a background gradient for visualization purposes.
    """
    df_all = pd.DataFrame(columns=['Models','MAE', 'MSE', 'RMSE', 'MAPE', "R\u00b2 score",'Runtime [ms]'])
    for i in range(len(name_model)):
        numb1 = len(df_all) + 1
        df_all.loc[numb1] = [f"{name_model[i]}",
                               mae[i],
                               mse[i],
                               rmse[i],
                               mape[i],
                               r2[i],
                              runtime[i],

                               ]
    # Apply formatting using Styler2
    cm = sns.color_palette("coolwarm", as_cmap=True)

    styled2_df2 = df_all.style.\
    background_gradient(cmap=cm)

    return df_all, styled2_df2
