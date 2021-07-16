import os
import numpy as np
import torch


def create_tensor_from_dataset(dataset: torch.utils.data.Dataset, train: bool=True) -> torch.tensor:
    """
    Transform a dataset from Dataset class to torch tensor
    :param dataset: torch dataset
    :param train: for training phase (boolean), if false -> test phase
    :return: tensors for data and labels (data, labels) : tuple
    """
    list_imgs = []
    list_lbls = []

    for i in range(len(dataset)):
        img, lbl = dataset[i]
        list_imgs.append(img)
        list_lbls.append(lbl)
        # TODO : Fix \r trick
        # print(f"\rStacking {'train' if train else 'test'} : {(i+1) / len(dataset) * 100:.2f}%", end='')
    
    tensor_imgs = torch.stack(list_imgs)
    tensor_lbls = torch.tensor(list_lbls)
    print('')

    return tensor_imgs, tensor_lbls


def load_data(dataset_name: str, data_path: str='data/') -> dict:
    """
    This function take the name of the dataset and loads images and labels for train and test stages
    :param dataset_name: name of the dataset
    :param data_path: path for the data directory
    :return: mapping with train and test sets (images and labels) as a dictionnary
    """
    data_map = {
        'train': {
            'images': None,
            'labels': None
        },
        'test': {
            'images': None,
            'labels': None
        }
    }
    for stage in data_map.keys():
        for data_type in data_map[stage].keys():
            file_path = os.path.join(data_path, f'{dataset_name}_{data_type}_{stage}.npy')
            if os.path.isfile(file_path): 
                with open(file_path, mode='rb') as f:
                    try:
                        # This line can be problematic
                        array = np.lib.format.read_array(f, allow_pickle=True)
                        if type(array) is np.ndarray and len(array.shape) > 0:
                            data_map[stage][data_type] = array
                        else:
                            raise ValueError(f"Data were not loaded, check {file_path} to correct the problem.")
                    except:
                        raise RuntimeError(f"Data can't be load, {file_path} may have a problem.")
            else:
                raise FileExistsError(f"{file_path} doesn't exist !")
    return data_map