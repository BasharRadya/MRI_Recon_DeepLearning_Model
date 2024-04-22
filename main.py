import argparse
import shutil
import random
import torch
import numpy as np
from models.PEAR import PEARModel
from matplotlib import pyplot as plt
from models.vanilla import VanillaModel
from utils.utils import create_data_loaders, freq_to_image
from models.validation import do_validation


from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def getDataloaderFromList(li, batch_size):
    custom_dataset = CustomDataset(li)
    shuffle = True
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def getFastDataloader(old, batch_size, filename):

    all_data = []
    for x in tqdm(old.dataset, desc="Processing dataset", unit=" samples"):
        all_data.append(x)

    dataloader = getDataloaderFromList(all_data, batch_size)
    dataset_path = './datasets/' + filename + '.pth';
    if not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    torch.save(all_data, dataset_path)
    return dataloader

def loadDataLoaderFromFile(filename, batch_size):
    all_data = None
    dataset_path = './datasets/' + filename + '.pth'
    all_data = torch.load(dataset_path)
    dataloader = getDataloaderFromList(all_data, batch_size)
    return dataloader
    

def generate_random_data(size):
    # Define a generator function
    def data_generator():
        for i in range(size):
            yield torch.randn(320,320,2),torch.randn(320,320)

    # Convert generator to a list of samples up to a given size
    def generator_to_list(generator_func, size):
        data_list = []
        for i, sample in enumerate(generator_func()):
            if i < size:
                data_list.append(sample)
            else:
                break
        return data_list

    # Create a custom Dataset class
    class CustomDataset(Dataset):
        def __init__(self, data_list):
            self.data_list = data_list
            
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, index):
            return self.data_list[index]

    # Specify the size of the dataset you want

    # Convert the generator to a list of samples
    data_list = generator_to_list(data_generator, size)

    # Create a CustomDataset instance
    dataset = CustomDataset(data_list)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    return dataloader




def main():
    import sys
    print("starting main function")
    sys.stdout.flush()
    args = create_arg_parser().parse_args() #get arguments from cmd/defaults

    #freeze seeds for result reproducability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    print("args:")
    print(args)
    print("starting making dataloaders")
    sys.stdout.flush()
    train_loader, validation_loader, test_loader = None, None, None
    if not args.load_dataset_fast_files:
        if args.generate_data:
            dataloader = generate_random_data(130)
            train_loader = dataloader
            validation_loader = dataloader 
            test_loader = dataloader
        else:    
            train_loader, validation_loader, test_loader = create_data_loaders(args) #get dataloaders

        print("starting making fast dataloaders")
        sys.stdout.flush()

        train_loader = getFastDataloader(train_loader, args.batch_size, filename='training')
        validation_loader = getFastDataloader(validation_loader, args.batch_size,filename='validation')
        test_loader = getFastDataloader(test_loader, args.batch_size,filename='testing')
    else:
        train_loader = loadDataLoaderFromFile('training', args.batch_size)
        validation_loader = loadDataLoaderFromFile('validation', args.batch_size)
        test_loader = loadDataLoaderFromFile('testing', args.batch_size)
    
    
    # total_samples = len(train_loader) * train_loader.batch_size
    # print(total_samples)
    # print("here")
    
    
    print("starting making grid")
    sys.stdout.flush()
    #model = VanillaModel(args.drop_rate, args.device, args.learn_mask).to(args.device) #Example instatiation - replace with your model
    # param_grid = {
    #     'drop_rate': np.arange(0.1, 0.75, 0.1),
    #     'device': ['cuda'],
    #     'learn_mask': [True, False],
    #     'block_len': [1, 2],
    #     'blocks_num': [3, 4],
    #     'bottleneck_block_len': [2, 3],
    #     'first_channel': [32, 64],
    #     'in_channel': [1],
    #     'k_size': [3, 4],
    #     'st': [2],
    #     'lr': [0.1, 0.001, 0.00005],
    # }

    param_grid = {
        #'drop_rate': np.arange(0.1, 0.75, 0.1),
        'drop_rate': [0.1],
        'device': ['cuda'],
        'learn_mask': [True, False],
        'block_len': [1],
        'blocks_num': [3, 4],
        'bottleneck_block_len': [2],
        'first_channel': [8],
        'in_channel': [1],
        'k_size': [3],
        'st': [2],
        'lr': [0.001],
    }
    print("starting validation")

    do_validation(param_grid=param_grid, dl_train=train_loader, dl_valid=validation_loader)
    print("finished")
def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for reproducability.')
    parser.add_argument('--data-path', type=str, default='/datasets/fastmri_knee/', help='path to MRI dataset.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of threads used for data handling.')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--report-interval', type=int, default=10, help='Report training stats once per this much iterations.')
    parser.add_argument('--drop-rate', type=float, default=0.8, help='Percentage of data to drop from each image (dropped in freq domain).')
    parser.add_argument('--learn-mask', action='store_true', default=False, help='Whether to learn subsampling mask')
    parser.add_argument('--results-root', type=str, default='results', help='result output dir.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learn rate for your reconstruction model.')
    parser.add_argument('--mask-lr', type=float, default=0.01, help='Learn rate for your mask (ignored if the learn-mask flag is off).')
    parser.add_argument('--val-test-split', type=float, default=0.3, help='Portion of test set (NOT of the entire dataset, since train-test split is pre-defined) to be used for validation.')
    parser.add_argument('--load-dataset-fast-files', type=int, default=0, help='Create files training.json testing.json validation.json for fast access')
    parser.add_argument('--generate-data', type=int, default=0, help='Generate data randomly for test runs')

    return parser
    
if __name__ == "__main__":    
    main()