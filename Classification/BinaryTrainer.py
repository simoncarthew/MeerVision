import os
import pandas as pd
import sys
import json
import shutil
import torch
import gc
import argparse
from CNNS import CNNS

sys.path.append("Data")
from DataManager import DataManager

def main():
    # SET AND GET ARGUMENTS
    parser = argparse.ArgumentParser(description="Model Parameter Arguments")
    parser.add_argument("--batch", type=int, default=6, help="")
    parser.add_argument("--image_size", type=int, default=64, help="")
    parser.add_argument("--dont_ask", action="store_true", help="Dont ask if resulst already exists")

    # VARYING PARAMETERS
    model_names = ["resnet50","mobilenet_v2","vgg16","efficientnet_b0","shufflenet_v2"]
    # model_names = ["resnet50"]
    # learning_rates = [0.05,0.01,0.005]
    learning_rates = [0.01]
    pretraineds = [True,False]

    # STATIC PARAMETERS

    # basic
    raw_path = os.path.join("Data","Classification")
    test_path = os.path.join(raw_path,"Binary","cut_images","test")
    results_path = os.path.join("Classification","Results","Binary")
    num_workers = 1
    batch = parser.parse_args().batch
    img_size = (int(parser.parse_args().image_size),int(parser.parse_args().image_size))
    behaviour = False
    epochs = 1
    perc_val = 0.2

    # obs
    obs_no = 150
    obs_test_no = 25

    # md
    md_z1_trainval_no = 25
    md_z2_trainval_no = 25
    md_test_no = 0

    #snap
    snap_no=200
    snap_test_no = 25

    # MANAGE DIRECTORIES
    results_path = os.path.join(results_path,"results_" + str(batch) + "_" + str(parser.parse_args().image_size))
    if os.path.exists(results_path):
        if parser.parse_args().dont_ask:
            shutil.rmtree(results_path)
            os.mkdir(results_path)
            os.mkdir(os.path.join(results_path,"trends"))
            os.mkdir(os.path.join(results_path,"models"))
        elif input("Resulst already obtained. Would you like to create new? (y/n)") == "y":
            shutil.rmtree(results_path)
            os.mkdir(results_path)
            os.mkdir(os.path.join(results_path,"trends"))
            os.mkdir(os.path.join(results_path,"models"))
        else:
            exit()
    else:
        os.mkdir(results_path)
        os.mkdir(os.path.join(results_path,"trends"))
        os.mkdir(os.path.join(results_path,"models"))


    # RESULTS
    columns = ['model_no', 'model_name', 'batch_size', 'learning_rate', 'pretrained', 'img_size', 'path', 'test_acc', 'inference', 'file_size']
    results_df = pd.DataFrame(columns=columns)
    model_no = 0


    # MAIN LOOP
    print(f"Creating dataloaders: batch: {batch}, image size: {img_size}")
    dm = DataManager(perc_val=perc_val,debug=False)
    train_loader, val_loader, test_loader = dm.create_dataloaders(raw_path=raw_path,batch=batch,
                                                                num_workers=num_workers,obs_no=obs_no,obs_test_no=obs_test_no,
                                                                md_z1_trainval_no=md_z1_trainval_no,md_z2_trainval_no=md_z2_trainval_no,
                                                                md_test_no=md_test_no,snap_no=snap_no,snap_test_no=snap_test_no,img_size=img_size,behaviour=behaviour
                                                                )
    print(f"Created dataloaders: batch: {batch}, image size: {img_size}")
    for model_name in model_names:
        for learning_rate in learning_rates:
            for pretrained in pretraineds:
                print(f"Training Model: batch: {batch}, image size: {img_size}, model_name: {model_name}, learning_rate: {learning_rate}, pretrained: {pretrained}")
                # load and train the model iteration
                model = CNNS(model_name=model_name,num_classes=2,pretrained=pretrained)
                results = model.train(train_loader=train_loader,val_loader=val_loader,epochs=epochs,learning_rate=learning_rate,test_loader=test_loader,inference_path=test_path)

                # save the model
                model.save_model(os.path.join(results_path, "models",model_name + "_" + str(model_no) + ".pth"))

                # get the model size 
                file_size = os.path.getsize(os.path.join(results_path, "models",model_name + "_" + str(model_no) + ".pth"))

                # save the model results
                results_df.loc[len(results_df)] = [model_no, model_name, batch, learning_rate, pretrained, int(parser.parse_args().image_size), model_name + "_" + str(model_no), results["test_acc"], results["inference"], file_size]
                with open(os.path.join(results_path,"trends", model_name + "_" + str(model_no) + ".json"), "w") as f:
                    json.dump(results, f, indent=4)
                results_df.to_csv(os.path.join(results_path,"results.csv"),index=False)

                # increase model number
                model_no += 1

                # reset memory 
                del model
                torch.cuda.empty_cache()
                gc.collect()

if __name__ == "__main__":
    main()