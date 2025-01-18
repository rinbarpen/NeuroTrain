from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)
