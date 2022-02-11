import numpy as np
import pandas as pd
import os
import json
import time
import datetime
from playsound import playsound

from misc.utils import MyUtils
from code_linear_regression.linear_regression import LinearRegression

# training config
max_degree = 3
training_epochs = 1000
eta_list = [0.01, 0.001, 0.0001]
lam_list = [200, 100, 50, 0] 

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
    
lr = LinearRegression()

results = [] # results will hold dict of (degree, epochs, eta, lam, train_mse, test_mse, y_hat)

for r in range(1, max_degree + 1):  # 1-based indexing
    print(f"degree {r}")
    
    print(f"\tepochs {training_epochs}")

    for eta_val in eta_list:
        print(f"\t\teta {eta_val}")

        for lam_val in lam_list:
            print(f"\t\t\tlam {lam_val}")

            start = time.time()
            train_mse, test_mse = lr.fit_metrics(X=X_train, y=y_train, X_test=X_test, y_test=y_test, epochs=training_epochs, eta=eta_val, degree=r, lam=lam_val)
            end = time.time()

            y_hat = lr.predict(X=X_test)
            
            min_train_mse = min(train_mse)
            min_test_mse = min(test_mse)

            result = {
                "degree": r,
                "epochs": training_epochs,
                "eta": eta_val,
                "lam": lam_val,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "min_train_mse": min_train_mse,
                "min_train_mse_epoch": train_mse.index(min_train_mse),
                "min_test_mse": min_test_mse,
                "min_test_mse_epoch": test_mse.index(min_test_mse),
                "y_hat": list(y_hat.flatten()), # json doesnt like the nd-array
                "train_time": (end - start) # trainng time in seconds
            } 

            results.append(result)

            print(f"\t\t\tlam {lam_val} done;")

        print(f"\t\teta {eta_val} done;")

    print(f"\tepochs {training_epochs} done;")
        
    print(f"degree {r} done;")
        
assert len(results) == max_degree * len(eta_list) * len(lam_list)
print(f"\nnumber of training runs: {len(results)}")

# add metadata
training_output = {
    "metadata": {
        "max_degree": max_degree,
        "training_epochs": training_epochs,
        "eta_list": eta_list,
        "lam_list": lam_list
    },
    "results": results
}

# store output
join_str = "-"
gd_output_filename = f"./output/{datetime.datetime.now()}_GD_degree-{max_degree}_epochs-{training_epochs}_eta-{join_str.join([str(int) for int in eta_list])}_lam-{join_str.join([str(int) for int in lam_list])}.json".replace(":", "-").replace(" ", "_")
with open(gd_output_filename, "w") as file:
    json.dump(training_output, file)

print(f"\ntraining output saved at {gd_output_filename}")

# notification when done
playsound('./misc/change_da_world.mp3')
