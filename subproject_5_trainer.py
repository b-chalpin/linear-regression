import pandas as pd
import datetime
import json
import os
from playsound import playsound

from misc.utils import MyUtils
from subproject_5_cf_trainer import train_cf_model as CF_train
from subproject_5_gd_trainer import train_gd_model as GD_train

# training config
max_degree = 1
training_epochs = 100
eta_list = [0.01, 0.001, 0.0001]
lam_list = [200, 100, 0]

# toggles for train/not train GD and CF
train_gd = True
train_cf = True

# dataset config
normalize_neg1_pos1 = False
normalize_zero_one = True
num_samples = None # set to None for all samples

data_path = "./dataset/houseprice"

X_train = pd.read_csv(os.path.join(data_path, "x_train.csv")).to_numpy()[:num_samples]
y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).to_numpy()[:num_samples]
X_test = pd.read_csv(os.path.join(data_path, "x_test.csv")).to_numpy()
y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).to_numpy()

if normalize_neg1_pos1:
    X_train = MyUtils.normalize_neg1_pos1(X_train)
    y_train = MyUtils.normalize_neg1_pos1(y_train)
    X_test = MyUtils.normalize_neg1_pos1(X_test)
    y_test = MyUtils.normalize_neg1_pos1(y_test)
    
elif normalize_zero_one:
    X_train = MyUtils.normalize_0_1(X_train)
    y_train = MyUtils.normalize_0_1(y_train)
    X_test = MyUtils.normalize_0_1(X_test)
    y_test = MyUtils.normalize_0_1(y_test)

# helper to format output file
def save_training_output(training_output, CF):
    base_path = "./output"
    join_str = "-"

    if CF:
        output_path = f"{base_path}/cf/{datetime.datetime.now()}_CF_degree-{max_degree}_lam-{join_str.join([str(int) for int in lam_list])}.json"
    else:
        output_path = f"{base_path}/gd/{datetime.datetime.now()}_GD_degree-{max_degree}_epochs-{training_epochs}_eta-{join_str.join([str(int) for int in eta_list])}_lam-{join_str.join([str(int) for int in lam_list])}.json"

    # post-process filename
    output_path = output_path.replace(":", "-").replace(" ", "_")
    with open(output_path, "w") as file:
        json.dump(training_output, file)

    print(f"\ntraining output saved at {output_path}")

# run gd training
if train_gd:
    gd_output = GD_train(X_train, y_train, X_test, y_test, max_degree, training_epochs, eta_list, lam_list)
    save_training_output(gd_output, CF=False)

    # notification when done
    playsound('./misc/train_complete.mp3')

# run cf training
if train_cf:
    cf_output = CF_train(X_train, y_train, X_test, y_test, max_degree, lam_list)
    save_training_output(cf_output, CF=True)

    # notification when done
    playsound('./misc/train_complete.mp3')
