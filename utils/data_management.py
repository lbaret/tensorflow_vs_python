import os
import numpy as np
import torch
import config as cfg


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


class TFDataset:
    def __init__(self, dataset_name: str, data_path: str='data/'):
        self.dataset_name = dataset_name
        self.data_path = data_path

        self.dataset = self._load_data()
    
    def _load_data(self) -> dict:
        """
        This function takes the name of the dataset and loads images and labels for train and test stages
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
                file_path = os.path.join(self.data_path, f'{self.dataset_name}_{data_type}_{stage}.npy')
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
    
    # FIXME : Remake the dataset with the code written in `tensorflow.ipynb`
    def get_training_set(self) -> dict:
        """
        :return: Dictionnary of training set
        """
        return self.dataset['train']

    def get_test_set(self) -> dict:
        """
        :return: Dictionnary of testing set
        """
        return self.dataset['test']

    def get_image_size(self) -> tuple:
        """
        :return: Tuple of image size (C, H, W)
        """
        return tuple(self.dataset['train']['images'].shape[1:])
    
    def get_train_size(self) -> int:
        """
        :return: Size of the training set
        """
        return len(self.dataset['train']['images'])
    
    def get_test_size(self) -> int:
        """
        :return: Size of the testing set
        """
        return len(self.dataset['test']['images'])

# TODO : Redefine this class with the new class
class TFBatchLoader:
    def __init__(self, batch_size: int=64, shuffles_nb: int=10000):
        # Config
        self.batch_size = batch_size
        self.shuffles_nb = shuffles_nb

        # Datasets
        self.cifar = None
        self.mnist = None
        self.fashion = None

    def _init_datasets(self):
        """
        Set the three used datasets
        """
        pass

    def _load_dataset(self, name: str):
        """
        Load specified dataset
        :param name: the name of the dataset ('mnist', 'cifar', 'fashion')
        """
        self.cifar = load_data(cfg.CIFAR_DATA_NAME)
        self.mnist = load_data(cfg.MNIST_DATA_NAME)
        self.fashion = load_data(cfg.FASHION_DATA_NAME)