import os
import pandas as pd

STD = {'batch': 32,
  'lr': 0.01,
  'augment': True,
  'freeze': 0,
  'pretrained': True,
  'obs_no': -1,
  'md_z1_trainval': 1000,
  'md_z2_trainval': 1000,
  'md_test_no': 0,
  'img_sz': 640,
  'optimizer': 'SGD'}

def merge_results(directory, save_dir):
    # Create a global results dataframe and a global model ID counter
    global_results = pd.DataFrame()
    global_model_id = 0

    # Create a 'trends' directory to store all the trends data
    trends_dir = os.path.join(save_dir, 'trends')
    os.makedirs(trends_dir, exist_ok=True)

    # Iterate through all results_<number> directories
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path) and folder.startswith("results"):
            train_csv_path = os.path.join(folder_path, 'train.csv')
            
            # Check if the train.csv file exists
            if os.path.exists(train_csv_path):
                # Read the train.csv file
                train_data = pd.read_csv(train_csv_path)
                train_data = train_data[~train_data.isna().any(axis=1)]

                # Iterate through each row in train.csv to extract information
                for idx, row in train_data.iterrows():
                    # Define the model directory based on the 'id' column in train.csv
                    model_id = row['id']
                    model_dir = os.path.join(folder_path,"models", f'model_{model_id}')
                    results_csv_path = os.path.join(model_dir, 'results.csv')

                    # If results.csv exists, copy it to the trends folder
                    if os.path.exists(results_csv_path):
                        trend_df = pd.read_csv(results_csv_path)
                        trend_df = trend_df.drop_duplicates()
                        trend_df.to_csv(os.path.join(trends_dir, f'results_{global_model_id}.csv'), index=False)

                    # Update the row in train.csv to reflect the global model ID
                    row['id'] = global_model_id
                    global_results = pd.concat([global_results, pd.DataFrame([row])])

                    # Increment the global model ID
                    global_model_id += 1

    # Save the global results.csv
    global_results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    print(f"Data merged successfully! {global_model_id} models processed.")

def filter_results(results_path, model, filtered_param, std=STD):
    # load the global results file
    results_df = pd.read_csv(results_path)
    
    # selected filters
    filters = {'batch':True, 'lr':True, 'augment':True, 'freeze':True, 'pretrained':True, 'obs_no':True, 'md_z1_trainval':True, 'md_z2_trainval':True, 'md_test_no':True, 'optimizer':True, 'batch':True, 'img_sz':True}
    filters[filtered_param]

    # filter for desired param change
    variable_df = results_df[(results_df['lr'] == std['lr']) & 
                            (results_df['augment'] == std['augment']) &
                            (results_df['freeze'] == std['freeze']) &
                            (results_df['pretrained'] == std['pretrained']) &
                            (results_df['obs_no'] == std['obs_no']) &
                            (results_df['md_z1_trainval'] == std['md_z1_trainval']) &
                            (results_df['md_z2_trainval'] == std['md_z2_trainval']) &
                            (results_df['md_test_no'] == std['md_test_no']) &
                            (results_df['optimizer'] == std['optimizer']) &
                            (results_df['img_sz'] == std['img_sz']) &
                            (results_df['model'].str.contains(str(model), na=False))]

    print(variable_df)

# Run the script
# merge_results('ObjectDetection/Training/Results/hyper_tune_1','ObjectDetection/Training/Results')
filter_results("ObjectDetection/Training/Results/results.csv",5)
