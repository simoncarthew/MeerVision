import os
import argparse
import pandas as pd
import shutil
import re

def merge_results(parent_dir, merged_dir, delete_originals=False):
    # Check and find subfolders in the parent directory that match the expected format: "results_<batch>_<image_size>"
    pattern = re.compile(r"results_\d+_\d+")
    result_dirs = [os.path.join(parent_dir, folder) for folder in os.listdir(parent_dir) if pattern.match(folder) and os.path.isdir(os.path.join(parent_dir, folder))]
    
    if not result_dirs:
        print(f"No result directories found in {parent_dir} with the expected format 'results_<batch>_<image_size>'.")
        return

    # Create directories in merged folder for results, models, and trends
    os.makedirs(merged_dir, exist_ok=True)
    trends_dir = os.path.join(merged_dir, "trends")
    models_dir = os.path.join(merged_dir, "models")
    os.makedirs(trends_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Initialize a merged dataframe and a model counter
    merged_results = pd.DataFrame(columns=["model_no", "model_name", "batch_size", "learning_rate", 
                                           "pretrained", "img_size", "path", "test_acc", "inference"])
    global_model_no = 0

    # Loop through each result directory found in the parent directory
    for result_dir in result_dirs:
        # Identify the CSV file in the current result directory
        csv_path = os.path.join(result_dir, "results.csv")
        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found. Skipping directory {result_dir}.")
            continue

        # Load the results CSV
        current_results = pd.read_csv(csv_path)
        
        # For each model in the current results, update model_no and paths
        for _, row in current_results.iterrows():
            # Set the new model number
            old_model_no = row["model_no"]
            new_model_no = global_model_no
            global_model_no += 1

            # Update paths and model number in the row
            row["model_no"] = new_model_no
            row["path"] = f"{row['model_name']}_{new_model_no}"

            # Copy the trend and model files to the new location with updated names
            model_name = row["model_name"]
            old_trend_path = os.path.join(result_dir, f"trends/{model_name}_{int(old_model_no)}.json")
            new_trend_path = os.path.join(trends_dir, f"{model_name}_{new_model_no}.json")
            
            old_model_path = os.path.join(result_dir, f"models/{model_name}_{int(old_model_no)}.pth")
            new_model_path = os.path.join(models_dir, f"{model_name}_{new_model_no}.pth")

            shutil.copyfile(old_trend_path, new_trend_path)
            shutil.copyfile(old_model_path, new_model_path)

            # Add the updated row to the merged results
            merged_results.loc[len(merged_results)] = row

        # Optionally delete the original result directory
        if delete_originals:
            shutil.rmtree(result_dir)
            print(f"Deleted directory: {result_dir}")

    # Save the merged results CSV
    merged_csv_path = os.path.join(merged_dir, "results.csv")
    merged_results.to_csv(merged_csv_path, index=False)
    print(f"Merged results saved to {merged_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge training results from multiple directories in a parent folder.")
    parser.add_argument("--parent_dir", required=True, help="Parent directory containing the result folders to merge.")
    parser.add_argument("--merged_dir", required=True, help="Directory to save the merged results.")
    parser.add_argument("--delete_originals", action="store_true", help="Delete original result folders after merging.")

    args = parser.parse_args()

    # Call the merge function with provided arguments
    merge_results(args.parent_dir, args.merged_dir, args.delete_originals)

if __name__ == "__main__":
    main()