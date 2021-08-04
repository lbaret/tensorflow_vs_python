import os
import numpy as np
import torch
import config as cfg
import tensorflow as tf


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
        # FIXME : \r trick
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
    
    def get_training_set(self) -> tf.data.Dataset:
        """
        :return: Training dataset
        """
        images = self.dataset['train']['images']
        labels = self.dataset['train']['labels']
        # We reshape images to respect TF format (for convolutions)
        train_shape = images.shape
        return tf.data.Dataset.from_tensor_slices((
            images.reshape((train_shape[0], train_shape[2], train_shape[3], train_shape[1])),
            labels
        ))

    def get_test_set(self) -> tf.data.Dataset:
        """
        :return: Testing dataset
        """
        images = self.dataset['test']['images']
        labels = self.dataset['test']['labels']
        # We reshape images to respect TF format (for convolutions)
        test_shape = images.shape
        return tf.data.Dataset.from_tensor_slices((
            images.reshape((test_shape[0], test_shape[2], test_shape[3], test_shape[1])),
            labels
        ))

    def get_original_image_size(self) -> tuple:
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


class TFBatchLoader:
    def __init__(self, batch_size: int=64, shuffles_nb: int=10000):
        # Config
        self.batch_size = batch_size
        self.shuffles_nb = shuffles_nb

        # Datasets
        self.cifar = TFDataset('cifar')
        self.mnist = TFDataset('mnist')
        self.fashion = TFDataset('fashion')

        # Mapping (private)
        self._mapper = {
            'cifar': self.cifar,
            'mnist': self.mnist,
            'fashion': self.fashion
        }
    
    def get_training_loader(self, dataset_name: str, batch_size: int=None, shuffles_nb: int=None) -> tf.data.Dataset:
        """
        Get shuffled and batched training set
        :param dataset_name: Name of the dataset to return
        :param batch_size: Batch size
        :param shuffles_nb: Number of shuffle iterations
        :return: shuffled and batched set
        """
        if batch_size is None:
            batch_size = self.batch_size
        if shuffles_nb is None:
            shuffles_nb = self.shuffles_nb
        training_set = self._mapper[dataset_name].get_training_set()

        return training_set.shuffle(shuffles_nb).batch(batch_size)

    def get_testing_loader(self, dataset_name: str, batch_size: int=None, shuffles_nb: int=None) -> tf.data.Dataset:
        """
        Get shuffled and batched testing set
        :param dataset_name: Name of the dataset to return
        :param batch_size: Batch size
        :param shuffles_nb: Number of shuffle iterations
        :return: shuffled and batched set
        """
        if batch_size is None:
            batch_size = self.batch_size
        if shuffles_nb is None:
            shuffles_nb = self.shuffles_nb
        testing_set = self._mapper[dataset_name].get_testing_set()
        
        return testing_set.shuffle(shuffles_nb).batch(batch_size)