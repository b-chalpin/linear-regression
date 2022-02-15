# Linear Regression By Hand

## Description

Using `numpy` and native Python, implement a Linear Regression classification model. The model is able to be trained using a Closed-form method, or Gradient Descent method.

## Environment Installation

##### Using Conda

To create the virtual environment and install required Python packages, run the following commands in the terminal:

```
$conda env create -f environment.yml
$conda activate cscd496-prog4-bchalpin
```

##### Without Conda

If you do not have Conda installed, the packages may still be installed using the following command:

```
$pip install -r requirements.txt
```

## Linear Regression Training

For training, first open `subproject_5_trainer.py` in a code editor of your choice. Adjust the hyperparameter configuration however you like. These will look like

```
# training config
max_degree = 4
training_epochs = 100_000
epoch_step = 1000
eta_list = [0.01, 0.001, 0.0001]
lam_list = [0.01, 0.001, 0]
notify_when_done = False # option to notify user with audio when training is done

# toggles for train/not train GD and CF
train_gd = True
train_cf = True

# dataset config
normalize_neg1_pos1 = False
normalize_zero_one = True
num_samples = None # set to None for all samples
num_features = None # set to None for all features
```

Lastly, train the model by exeuting the following command.

```
$python subproject_5_trainer.py
```

## Visualization

Copy the file names of the training output from the previous section. Within `subproject_5.ipynb` replace the following code with your new output file names:

```
...
# change the filenames below accordingly. 
### note that when running both GF and CF visualization, hyperparameter configs MUST match
gd_output_filename = f"{gd_output_base_path}/2022-02-14_17-55-55.706677_GD_degree-4_epochs-100000_eta-0.01-0.001-0.0001_lam-0.01-0.001-0.json"
cf_output_filename = f"{cf_output_base_path}/2022-02-14_18-00-18.206677_CF_degree-4_lam-0.01-0.001-0.json"
...
```

### Author

Blake Chalpin [b-chalpin](https://github.com/b-chalpin)

