from sklearn.pipeline import Pipeline, FeatureUnion
from preprocessors2.equi_distance_metrics_preprocessor3 import EquiDistanceMetricsPreprocessor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Simple_forecast_UB import DataSplitter, ModelForecaster
import argparse
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import HTML
from UB__Prediction_using_BT import DataSplitter_BT_Fixed
from Simple_forecast_UB import SortDataFrame,CheckSequence,fill_nan_with_median,FillMissingSequenceTimes,AddVmid,ZScoreIQRDetector,SavitzkyGolayFilter

import numpy as np
name_model = ["LinearRegression","XGBRegressor","MLPRegressor","SVR","CatBoostRegressor","LGBMRegressor"]


def main(vmid, name_columns,anim,steps):
    data = pd.read_csv(f'./Vms_Condidates/{vmid}.csv')  # Fixed: added f-string
    start_date = pd.to_datetime('2019-01-01')
    data['timestamp'] = start_date + pd.to_timedelta(data['timestamp'], unit='s')
    # data = data[['timestamp', name_columns]]
    pre_processing_pipeline = Pipeline([
        ("check_Sequence", CheckSequence()),
        ("fill_missing_SequenceTimes", FillMissingSequenceTimes()),
    ])

    fill_missing_pipeline = Pipeline([
        ("fill_nan_with_median",fill_nan_with_median()),
        ("AddVmid" , AddVmid()),

    ])

    z_score_pipeline = Pipeline([
        ("Clip",ZScoreIQRDetector())
    ])

    denoising_pipeline = Pipeline([
        ("DE_noising",SavitzkyGolayFilter()),
    ])

    used_pipeline = Pipeline([
        ("sort_by_timestamp",  FeatureUnion([('SortDataFrame',       SortDataFrame("timestamp"))])),
        ('sub_preprocessing',  FeatureUnion([('pre_processing',      pre_processing_pipeline)])),
        ('sub_fill_nan',       FeatureUnion([('FillMissingValues',   fill_missing_pipeline)])),
        ('sub_ClipOvershoots', FeatureUnion([('ClipOvershoots',      z_score_pipeline)])),
        ('sub_denoising',      FeatureUnion([('denoising_Filter',    denoising_pipeline)])),
    ])
    sort_data_time = used_pipeline.get_params()['sort_by_timestamp__SortDataFrame']
    sorted_df = sort_data_time.transform(data)
    check_processing_data = used_pipeline.get_params()['sub_preprocessing__pre_processing__check_Sequence']
    sorted_df['timestamp'] = pd.to_datetime(sorted_df['timestamp'])  # Convert the column to datetime
    sorted_df.set_index('timestamp', inplace=True)
    fill_missing_times_processing = used_pipeline.get_params()['sub_preprocessing__pre_processing__fill_missing_SequenceTimes']
    fill_missing_times_df         = fill_missing_times_processing.fit_transform(sorted_df)
    filler = used_pipeline.get_params()['sub_fill_nan__FillMissingValues__fill_nan_with_median']
    filler.fit(fill_missing_times_df)
    df_filled = filler.transform(fill_missing_times_df)
    addVmiddata= used_pipeline.get_params()['sub_fill_nan__FillMissingValues__AddVmid']
    df_final=addVmiddata.transform(df_filled)
    Clipped = used_pipeline.get_params()['sub_ClipOvershoots__ClipOvershoots__Clip']
    Clipped.fit(name_columns,df_final)
    df_no_outliers=Clipped.transform(df_final)
    denoiser  = used_pipeline.get_params()['sub_denoising__denoising_Filter__DE_noising']
    denoiser.fit(df_no_outliers,name_columns)
    final_PrePrecessedDF=denoiser.transform(df_no_outliers)
    # print(final_PrePrecessedDF.isna().sum())   
    final_PrePrecessedDF.reset_index(inplace=True)
    final_PrePrecessedDF.rename(columns={'index': 'timestamp'}, inplace=True)
    data = final_PrePrecessedDF[['timestamp' , name_columns]]

    if anim==-1:
        step_size = steps
        data_train = data[:-step_size]
        data_test = data[-step_size:]  # unseen              
        # create plot
        plt.style.use("ggplot")  # <-- set overall look
        fig, ax = plt.subplots( figsize=(18,5))

        # plot data
        plt.plot(data['timestamp'], data[f'{name_columns}'], 'r-', linewidth=0.5,  label='data or y')
        Y_LIM       = 2*10**8 #data[f'{name_columns}'].max()

        ax.fill_between(data_train['timestamp'], 0,data[f'{name_columns}'].max(),where=data_train[f'{name_columns}'],facecolor='g', alpha=0.3)
        ax.fill_between(data_test['timestamp'] , 0,data[f'{name_columns}'].max(), where=data_test[f'{name_columns}'], facecolor='b', alpha=0.3)

        # make graph beautiful
        plt.plot([], [], 'g-', label="Train", linewidth=8, alpha=0.3)
        plt.plot([], [], 'b-', label="Test",  linewidth=8, alpha=0.3)

        step_size = 288
        selected_ticks = data['timestamp'][::step_size]
        plt.xticks(selected_ticks, rotation=90)
        #plt.xticks([data['date'].iloc[-1], 50, 100, 150, 200, 250, data['date'].iloc[-1]], visible=True, rotation="horizontal")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))


        Y_LIM       = 2*10**8 #data[f'{name_columns}'].max()
        TRAIN_WIDTH =   len(data_train['timestamp'][::step_size])
        TEST_WIDTH  =   len(data_test[ 'timestamp'][::step_size])

        plt.title(f'Data split:\n taraing-set {100*(len(data_train)/len(data)):.2f}% = {TRAIN_WIDTH} days and test-set {100*(len(data_test)/len(data)):.2f}% = {TEST_WIDTH} days')
        plt.ylabel(f'{name_columns} usage %', fontsize=15)
        plt.xlabel('Timestamp', fontsize=15)
        plt.grid(True)
        plt.legend(loc="upper left")
        #plt.ylim(2*10**6)
        #plt.legend(bbox_to_anchor=(1.1,.9), loc="upper right")
        fig.tight_layout(pad=1.2)
        plt.savefig("./plot_anim/-1.png")
    elif anim==0:   #Without Refit  
      step_size = steps
      data_train = data[:-step_size]
      data_test  = data[-step_size:] #unseen

      # create plot
      plt.style.use("ggplot")  # <-- set overall look
      fig, ax = plt.subplots( figsize=(10,5))
      # plot data
      plt.plot(list(data['timestamp']), data[name_columns], 'r-', linewidth=0.5,  label='data or y')

      # translate data to graph coordinates
      ax.margins(x=0, y=0)
      x_min, x_max = ax.get_xlim()
      y_min, y_max = ax.get_ylim()
      ax.margins(x=0.05, y=0.05)
      height = y_max-y_min

      # make graph beautiful
      plt.plot([], [], 'g-', label="Train", linewidth=8, alpha=0.3)
      plt.plot([], [], 'b-', label="Test",  linewidth=8, alpha=0.3)

      step_size = 287
      selected_ticks = data['timestamp'][::step_size]

      plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
      plt.xticks(selected_ticks, rotation=90)

      Y_LIM       = 2*10**8 #df[f'{name_columns}'].max()
      TRAIN_WIDTH =  288
      TEST_WIDTH  =  steps
      plt.title(f'Data split:\nsWithout Refit\ntaraing-set {100*(TRAIN_WIDTH/len(data)):.2f}% = {TRAIN_WIDTH//288} days and test-set {100*(len(data_test)/len(data))}% = {TEST_WIDTH//288} days')
      plt.ylabel(f' CPU usage %',   fontsize=15)
      plt.xlabel('Timestamp', fontsize=15)
      plt.grid(True)
      plt.legend(loc="upper left")
      fig.tight_layout(pad=1.2)
      def init():
          rects = [Rectangle((x_min, y_min), 5,  height, alpha=0.3, facecolor='green'),
                  Rectangle((x_min+5, y_min), 3,  height, alpha=0.3, facecolor='blue')]
          patches = []
          for rect in rects:
                  patches.append(ax.add_patch(rect))
          return patches
      def update(x_start):
          patches[1].xy = (x_start + 5, y_min)
          return patches
      # create "Train" and "Test" areas
      patches = init()
      ani = FuncAnimation(
          fig,
          update,
          frames= np.linspace(x_min, x_max-0.25*(x_max-x_min), 50),  # all starting points
          interval=150,
          blit=True)
      ani.save('./plot_anim/0.mp4', writer='ffmpeg', fps=15)      
    elif anim==1:   #Fixed Backtesting 
      plt.style.use("ggplot")  # <-- set overall look
      fig, ax = plt.subplots( figsize=(10,4))

      # plot data
      plt.plot(list(data['timestamp']), data[name_columns], 'r-', linewidth=0.5,  label='data or y')

      # translate data to graph coordinates
      ax.margins(x=0, y=0)
      x_min, x_max = ax.get_xlim()
      y_min, y_max = ax.get_ylim()
      ax.margins(x=0.05, y=0.05)
      height = y_max-y_min

      # make graph beautiful
      plt.plot([], [], 'g-', label="Train", linewidth=8, alpha=0.3)
      plt.plot([], [], 'b-', label="Test",  linewidth=8, alpha=0.3)

      step_size = 287
      selected_ticks = data['timestamp'][::step_size]

      plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
      plt.xticks(selected_ticks, rotation=90)

      Y_LIM       = 2*10**8 #df[f'{name_columns}'].max()
      TRAIN_WIDTH =  288*27
      TEST_WIDTH  =  steps
      data_splitter = DataSplitter_BT_Fixed(steps)
      data_train, data_test = data_splitter.split_data(data)

      plt.title(f'Data split:\nFixed Backtesting training-set {100*(len(data_train)/len(data)):.2f}% = {TRAIN_WIDTH//288} days and test-set {100*(len(data_test)/len(data)):.2f}% = {TEST_WIDTH//288} days')
      plt.ylabel(f' usage %',   fontsize=15)
      plt.xlabel('Timestamp', fontsize=15)
      plt.grid(True)
      plt.legend(loc="upper left")
      fig.tight_layout(pad=1.2)
      def init():
          rects = [Rectangle((x_min, y_min), 5,  height, alpha=0.3, facecolor='green'),
                  Rectangle((x_min+5, y_min), 3,  height, alpha=0.3, facecolor='blue')]
          patches = []
          for rect in rects:
                  patches.append(ax.add_patch(rect))
          return patches

      def update(x_start):
          patches[0].xy = (x_start, y_min)
          patches[1].xy = (x_start + 5, y_min)
          return patches
      # create "Train" and "Test" areas
      patches = init()
      ani = FuncAnimation(
          fig,
          update,
          frames= np.linspace(x_min, x_max-0.25*(x_max-x_min), 50),  # all starting points
          interval=150,
          blit=True)
      ani.save('./plot_anim/1.mp4', writer='ffmpeg', fps=15)
    elif anim == 2:  # Intermittent Refitting
      step_size = steps 
      data_train = data[:-step_size]
      data_test  = data[-step_size:] #unseen

      # create plot
      plt.style.use("ggplot")  # <-- set overall look
      fig, ax = plt.subplots( figsize=(10,5))


      # plot data
      plt.plot(list(data['timestamp']), data[name_columns], 'r-', linewidth=0.5,  label='data or y')

      # translate data to graph coordinates
      ax.margins(x=0, y=0)
      x_min, x_max = ax.get_xlim()
      y_min, y_max = ax.get_ylim()
      ax.margins(x=0.05, y=0.05)
      height = y_max-y_min
      # make graph beautiful
      plt.plot([], [], 'g-', label="Train", linewidth=8, alpha=0.3)
      plt.plot([], [], 'b-', label="Test",  linewidth=8, alpha=0.3)
      # plt.ylim(y_min, y_max)

      step_size = 287
      selected_ticks = data['timestamp'][::step_size]

      plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
      plt.xticks(selected_ticks, rotation=90)

      TRAIN_WIDTH =  288*27
      TEST_WIDTH  =  steps

      plt.title(f'Data split:\nIntermittent Refitting taraing-set {100*(len(data_train)/len(data)):.2f}% = {TRAIN_WIDTH//288} days and test-set {100*(len(data_test)/len(data)):.2f}% = {TEST_WIDTH//288} days')
      plt.ylabel(f' CPU usage %',   fontsize=15)
      plt.xlabel('Timestamp', fontsize=15)
      plt.grid(True)
      plt.legend(loc="upper left")
      fig.tight_layout(pad=1.2)
      
      def init():
          rects = [Rectangle((x_min, y_min), 5,  height, alpha=0.3, facecolor='green'),
                  Rectangle((x_min+5, y_min), 3,  height, alpha=0.3, facecolor='blue')]
          patches = []
          for rect in rects:
                  patches.append(ax.add_patch(rect))
          return patches

      # Initialize the counter for refitting the model
      refit_counter = 0
      REFIT_DELAY = 5 # steps
      previous_x_start = x_min
      def update(x_start):
          nonlocal   previous_x_start,refit_counter

          rect_width = 5
          patches[1].xy = (x_start+rect_width, y_min)
          width_diff = x_start - previous_x_start
          if width_diff > 0.45:
              if refit_counter == REFIT_DELAY:
                  rect_width_diff = 2.58
                  patches[0].set_width(patches[0].get_width() + rect_width_diff)
                  refit_counter = 0
              refit_counter += 1
          previous_x_start = x_start
          return patches

      # create "Train" and "Test" areas
      patches = init()
      ani = FuncAnimation(
          fig,
          update,
          frames= np.linspace(x_min, x_max-0.21*(x_max-x_min)+0.99, 50),  # all starting points
          interval=150,
          blit=True)
      ani.save('./plot_anim/2.mp4', writer='ffmpeg', fps=15)    
    else:           #Refit Increasing Training Size
      step_size = steps 
      data_train = data[:-step_size]
      data_test  = data[-step_size:] #unseen
      # create plot
      plt.style.use("ggplot")  # <-- set overall look
      fig, ax = plt.subplots( figsize=(10,5))
      # plot data
      plt.plot(list(data_train['timestamp']), data_train[name_columns], 'r-', linewidth=0.5,  label='data or y')
      plt.plot(list(data_test['timestamp']),  data_test[name_columns] , 'r-', linewidth=0.5,  label='unseen data')

      # translate data to graph coordinates
      ax.margins(x=0, y=0)
      x_min, x_max = ax.get_xlim()
      y_min, y_max = ax.get_ylim()
      ax.margins(x=0.05, y=0.05)
      height = y_max-y_min

      # make graph beautiful
      plt.plot([], [], 'g-', label="Train", linewidth=8, alpha=0.3)
      plt.plot([], [], 'b-', label="Test",  linewidth=8, alpha=0.3)

      step_size = 287
      selected_ticks = data['timestamp'][::step_size]

      plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
      plt.xticks(selected_ticks, rotation=90)

      Y_LIM       = 2*10**8 #df[f'{name_columns}'].max()
      TRAIN_WIDTH =  288*27
      TEST_WIDTH  =  steps
      plt.title(f'Data split:\nBacktesting with refit and increasing training size (fixed origin)\ntaraing-set {100*(len(data_train)/len(data)):.2f}% = {TRAIN_WIDTH//288} days and test-set {100*(len(data_test)/len(data)):.2f}% = {TEST_WIDTH//288} days')
      # plt.title(f'Backtesting with refit and increasing training size (fixed origin)')

      plt.ylabel(f' CPU usage %',   fontsize=15)
      plt.xlabel('date', fontsize=15)
      plt.grid(True)
      plt.legend(loc="upper left")
      fig.tight_layout(pad=1.2)


      def init():
          rects = [Rectangle((x_min,   y_min), 5,  height, alpha=0.3, facecolor='green'),
                  Rectangle((x_min+5, y_min), 1,  height, alpha=0.3, facecolor='blue')]
          patches = []
          for rect in rects:
                  patches.append(ax.add_patch(rect))
          return patches

      def update(x_start):
          patches[1].xy = (x_start + 5, y_min)
          # Add 5 to the width of the green rectangle each time
          green_width = 5 + (x_start - x_min) * 1
          patches[0].set_width(green_width)
          return patches

      # create "Train" and "Test" areas
      patches = init()
      ani = FuncAnimation(
          fig,
          update,
          frames= np.linspace(x_min, x_max-0.25*(x_max-x_min), 50),  # all starting points
          interval=150,
          blit=True)
      ani.save('./plot_anim/3.mp4', writer='ffmpeg', fps=15)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time series forecasting with models.")    
    parser.add_argument('vmid', type=str, help='Name of the dataset (without .csv)')
    parser.add_argument('name_columns', default="avgcpu", type=str, help='Name of the column to forecast')
    parser.add_argument('animation', type=int, help='Name of the animation for plot')
    parser.add_argument('steps', type=int, default=3*288, help='Number of time steps to forecast')

        
    args = parser.parse_args()
    main(args.vmid, args.name_columns, args.animation, args.steps)
