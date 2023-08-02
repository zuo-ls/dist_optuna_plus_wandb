import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import torchvision
import torch
import os

class TOYDataset(torch.utils.data.Dataset):
    def __init__(self,data,targets):
        super(TOYDataset, self).__init__()
        self.data = data
        self.targets = targets
    def __getitem__(self, index):
        return self.data[index].squeeze(),self.targets[index]

    def __len__(self):
        return len(self.targets)

def get_dataloaders(
    train_bs,
    valid_bs,
    test_bs,
    add_noise=False,
    noise_level=0.2,
    n_train=1000,
    n_valid=200,
    n_test=500,
    dataset_pth = None,
    download=True,
):
    """
    MNIST DATA for toy experiment
    """
    if dataset_pth is None:
        root_dir = os.path.abspath(os.path.join(__file__,'../../'))
        dataset_pth = os.path.join(root_dir,'data')
        if download:
            os.makedirs(dataset_pth,exist_ok=True)
        # else:
        #     dataset_pth = os.path.join(dataset_pth,'MNIST')
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_pth,
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=download,
    )
    test_dataset = torchvision.datasets.MNIST(root=dataset_pth, train=False)
    
    # train data
    train_data = train_dataset.data[:n_train]/255.
    train_label = train_dataset.targets[:n_train]
    if add_noise: # add noise to train data
        noise = torch.randn(train_data.size()) * noise_level
        train_data = train_data + noise

    train_dataset = TOYDataset(train_data, train_label)

    # valid data
    valid_data = test_dataset.data[-n_valid:]/255.
    valid_label = test_dataset.targets[-n_valid:]
    valid_dataset = TOYDataset(valid_data,valid_label)
    
    # test data
    test_data = test_dataset.data[:n_test]/255.
    test_label = test_dataset.targets[:n_test]
    test_dataset = TOYDataset(test_data, test_label)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=valid_bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=True)
    return train_loader,valid_loader,test_loader

