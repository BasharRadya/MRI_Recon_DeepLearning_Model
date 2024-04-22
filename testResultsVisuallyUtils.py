import torch
import models.PEAR as PEAR
from main import loadDataLoaderFromFile
import os
def loadModel():
    params = {
        #'drop_rate': np.arange(0.1, 0.75, 0.1),
        'block_len': 1,
        'blocks_num': 4,
        'bottleneck_block_len': 2,
        'device': 'cuda',
        'drop_rate': 0.1,
        'first_channel': 64,
        'in_channel': 1,
        'k_size': 3,
        'learn_mask': False,
        'lr': 0.001,
        'st': 2,
    }
    model_str = [(str(params[param]) + ' ') for param in params]
    model_str = ''.join(model_str) + '.pt'
    checkpoint_filename = "checkpoints/" + model_str
    print(checkpoint_filename)
    del params['lr']
    model = PEAR.PEARModel(**params)
    saved_state = torch.load(checkpoint_filename, map_location="cuda")
    model.load_state_dict(saved_state["model_state"])
    return model

def _getFirstFromDataloader(loader):
    first = None
    for item in loader:
        first = item
        break
    return first

def runModelAndSaveImgs(model, num):
    validation_loader = loadDataLoaderFromFile('validation', num)
    model.train(False)
    with torch.no_grad():
        x, y = _getFirstFromDataloader(validation_loader)
        x = x.to("cuda")
        y = y.to("cuda")
        y = y.unsqueeze(1)
        model = model.to("cuda")
        orig_x = model.subsample(x)
        model_out = model(x)
        args = [("./results/label.pth", y),
            ("./results/reconstructed.pth", model_out),
            ("./results/input.pth", orig_x),
        ]
        def saveImgs(path, to_save):
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(to_save, path)
        for arg in args:
            saveImgs(*arg)

def loadDataFromFile():

    import os
    args = [
        "./results/label.pth",
        "./results/reconstructed.pth",
        "./results/input.pth",
    ]
    def loadImgs(path):    
        return torch.load(path)
    li = []
    for arg in args:
        li.append(loadImgs(arg))
    return li
