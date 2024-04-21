import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
#from ./training import PEARTrainer
from . import training
#from ./PEAR import PEARModel
from . import PEAR
import copy

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define your dataset
# Assume you have a dataset `train_data` and `train_labels` already prepared
# ...

# Define hyperparameters to tune



def do_validation(param_grid, dl_train, dl_valid):
    # Perform grid search
    best_test_loss = None
    best_model = None
    results = []
    counter = 0
    for params in ParameterGrid(param_grid):
        model_str = [(str(params[param]) + ' ') for param in params]
        model_str = ''.join(model_str)
        lr = params['lr']
        del params['lr']
        model = PEAR.PEARModel(**params)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, threshold=0.01, verbose=True)

        # Train and evaluate the model
        trainer = training.PEARTrainer(
            model=model,
            loss_fn=torch.nn.MSELoss(),
            optimizer=optimizer,
            device=params['device'],
            mask_lr=lr
        )

        res = trainer.fit(dl_train=dl_train, dl_test=dl_valid, num_epochs=50, checkpoints="checkpoints/" + model_str, early_stopping=7)
        def mean(li):
            return sum(li) / len(li)
        mean_train_loss = mean(res.train_loss)
        mean_test_loss = mean(res.test_loss)
        
        results.append(model_str + ": train loss: " + str(mean_train_loss)+ ", test loss: " + str(mean_test_loss))
        
        # Keep track of the best accuracy and parameters
        if best_test_loss == None or best_test_loss > mean_test_loss:
            best_test_loss = mean_test_loss
            best_model = model_str
        counter += 1
        print(f'finished {counter}')
    for string in results:
        print(string)
    print("best_model: " + best_model)
    
