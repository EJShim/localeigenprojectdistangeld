import json
import os
import pickle
import tqdm
import trimesh
import torch

import numpy as np

from abc import abstractmethod
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Dataset, InMemoryDataset, Data

from swap_batch_transform import SwapFeatures


class DataGenerator:
    def __init__(self, model_dir, data_dir='./data'):
        self._model_dir = model_dir
        self._data_dir = data_dir

    def __call__(self, number_of_meshes, weight=1., overwrite_data=False):
        if not os.path.isdir(self._data_dir):
            os.mkdir(self._data_dir)

        if not os.listdir(self._data_dir) or overwrite_data:  # directory empty
            print("Generating Data from PCA")
            for i in tqdm.tqdm(range(number_of_meshes)):
                v = self.generate_random_vertices(weight)
                self.save_vertices(v, str(i))

    def save_vertices(self, vertices, name):
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        m = trimesh.Trimesh(vertices, process=False)
        m.export(os.path.join(self._data_dir, name + '.ply'))

    @abstractmethod
    def generate_random_vertices(self, weight):
        pass


class FaceGenerator(DataGenerator):
    def __init__(self, uhm_path, data_dir='./data'):
        super(FaceGenerator, self).__init__(uhm_path, data_dir)
        infile = open(self._model_dir, 'rb')
        uhm_dict = pickle.load(infile)
        infile.close()

        self._components = uhm_dict['Eigenvectors'].shape[1]
        self._mu = uhm_dict['Mean']
        self._eigenvectors = uhm_dict['Eigenvectors']
        self._eigenvalues = uhm_dict['EigenValues']

    def generate_random_vertices(self, weight):
        w = weight * np.random.normal(size=self._components) * \
            self._eigenvalues ** 0.5
        w = np.expand_dims(w, axis=1)
        vertices = self._mu + self._eigenvectors @ w
        return vertices.reshape(-1, 3)


class BodyGenerator(DataGenerator):
    """ To install star model see https://github.com/ahmedosman/STAR"""
    def __init__(self, data_dir='./data'):
        super(BodyGenerator, self).__init__(None, data_dir)
        from star.pytorch.star import STAR
        self._star = STAR(gender='neutral', num_betas=10)

    def generate_random_vertices(self, weight=3):
        poses = torch.zeros([1, 72])
        betas = torch.rand([1, self._star.num_betas]) * 2 * weight - weight

        trans = torch.zeros([1, 3])
        verts = self._star.forward(poses.cuda(), betas.cuda(), trans.cuda())[-1]
        # Normalize verts in -1, 1 wrt height
        y_min = torch.min(verts[:, 1])
        scale = 2 / (torch.max(verts[:, 1]) - y_min)
        verts[:, 1] -= y_min
        verts *= scale
        verts[:, 1] -= 1
        return verts

    def save_mean_mesh(self):
        t = trimesh.Trimesh(self._star.v_template.cpu().numpy(),
                            self._star.f, process=False)
        t.export(os.path.join(self._data_dir, 'template.ply'))


def get_data_loaders(config, template=None, in_memory=True):
    data_config = config['data']
    batch_size = config['optimization']['batch_size']

    dataset = MeshInMemoryDataset if in_memory else MeshDataset
    train_set = dataset(
        data_config['dataset_path'], dataset_type='train',
        precomputed_storage_path=data_config['precomputed_path'],
        normalize=data_config['normalize_data'], template=template)
    validation_set = dataset(
        data_config['dataset_path'], dataset_type='val',
        precomputed_storage_path=data_config['precomputed_path'],
        normalize=data_config['normalize_data'], template=template)
    test_set = dataset(
        data_config['dataset_path'], dataset_type='test',
        precomputed_storage_path=data_config['precomputed_path'],
        normalize=data_config['normalize_data'], template=template)
    normalization_dict = train_set.normalization_dict

    swapper = SwapFeatures(template) if data_config['swap_features'] else None

    train_loader = MeshLoader(train_set, batch_size, shuffle=True,
                              drop_last=True, feature_swapper=swapper,
                              num_workers=data_config['number_of_workers'])
    validation_loader = MeshLoader(validation_set, batch_size, shuffle=True,
                                   drop_last=True, feature_swapper=swapper,
                                   num_workers=data_config['number_of_workers'])
    test_loader = MeshLoader(test_set, batch_size, shuffle=False,
                             drop_last=True, feature_swapper=swapper,
                             num_workers=data_config['number_of_workers'])
    return train_loader, validation_loader, test_loader, normalization_dict


class MeshLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 feature_swapper=None, **kwargs):
        collater = MeshCollater(feature_swapper)
        super(MeshLoader, self).__init__(dataset, batch_size, shuffle,
                                         collate_fn=collater, **kwargs)


class MeshCollater:
    def __init__(self, feature_swapper=None):
        self._swapper = feature_swapper

    def __call__(self, data_list):
        return self.collate(data_list)

    def collate(self, data_list):
        if not isinstance(data_list[0], Data):
            raise TypeError(
                f"DataLoader found invalid type: {type(data_list[0])}. "
                f"Expected torch_geometric.data.Data instead")

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        batched_data = Data()
        for key in keys:
            attribute_list = [data[key] for data in data_list]
            batched_data[key] = default_collate(attribute_list)
        if self._swapper is not None:
            batched_data = self._swapper(batched_data)
        return batched_data


class MixinMeshDataset:
    def _initialize(self, root, precomputed_storage_path='precomputed',
                    dataset_type='train', normalize=True, template=None):
        self._root = root
        self._precomputed_storage_path = precomputed_storage_path
        if not os.path.isdir(precomputed_storage_path):
            os.mkdir(precomputed_storage_path)

        self._dataset_type = dataset_type
        self._normalize = normalize
        self._template = template

        self._train_names, self._test_names, self._val_names = self.split_data(
            os.path.join(precomputed_storage_path, 'data_split.json'))

        self._processed_files = [f + '.pt' for f in self.raw_file_names]

        normalization_dict = self.compute_mean_and_std()
        self._normalization_dict = normalization_dict
        self.mean = normalization_dict['mean']
        self.std = normalization_dict['std']

    @property
    def normalization_dict(self):
        return self._normalization_dict

    def find_filenames(self):
        root_l = len(self._root)
        files = []
        for dirpath, _, fnames in os.walk(self._root):
            for f in fnames:
                if f.endswith('.ply') or f.endswith('.obj'):
                    absolute_path = os.path.join(dirpath, f)
                    f = absolute_path[dirpath.index(self._root) + root_l + 1:]
                    files.append(f)
        return files

    def split_data(self, data_split_list_path):
        try:
            with open(data_split_list_path, 'r') as fp:
                data = json.load(fp)
            train_list = data['train']
            test_list = data['test']
            val_list = data['val']
        except FileNotFoundError:
            all_file_names = self.find_filenames()
            all_file_names.sort()

            train_list, test_list, val_list = [], [], []
            for i, fname in enumerate(all_file_names):
                if i % 100 <= 5:
                    test_list.append(fname)
                elif i % 100 <= 10:
                    val_list.append(fname)
                else:
                    train_list.append(fname)

            data = {'train': train_list, 'test': test_list, 'val': val_list}
            with open(data_split_list_path, 'w') as fp:
                json.dump(data, fp)
        return train_list, test_list, val_list

    def load_mesh(self, filename, show=False):
        mesh_path = os.path.join(self._root, filename)
        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                                  requires_grad=False)
        if show:
            tm = trimesh.Trimesh(vertices=mesh.vertices,
                                 faces=self._template.face.t().cpu().numpy())
            tm.show()
        return mesh_verts

    def compute_mean_and_std(self):
        normalization_dict_path = os.path.join(
            self._precomputed_storage_path, 'norm.pt')
        try:
            normalization_dict = torch.load(normalization_dict_path)
        except FileNotFoundError:
            assert self._dataset_type == 'train'
            train_verts = None
            for i, fname in tqdm.tqdm(enumerate(self._train_names)):
                mesh_verts = self.load_mesh(fname)
                if i == 0:
                    train_verts = torch.zeros(
                        [len(self._train_names), mesh_verts.shape[0], 3],
                        requires_grad=False)
                train_verts[i, ::] = mesh_verts

            mean = torch.mean(train_verts, dim=0)
            std = torch.std(train_verts, dim=0)
            std = torch.where(std > 0, std, torch.tensor(1e-8))
            normalization_dict = {'mean': mean, 'std': std}
            torch.save(normalization_dict, normalization_dict_path)
        return normalization_dict

    def save_mean_mesh(self):
        first_mesh_path = os.path.join(self._root, self._train_names[0])
        first_mesh = trimesh.load_mesh(first_mesh_path, process=False)
        first_mesh.vertices = self.mean.detach().cpu().numpy()
        first_mesh.export(
            os.path.join(self._precomputed_storage_path, 'mean.ply'))


class MeshDataset(MixinMeshDataset, Dataset):
    def __init__(self, root, precomputed_storage_path='precomputed',
                 dataset_type='train', normalize=True,
                 transform=None, pre_transform=None, template=None):

        self._initialize(root, precomputed_storage_path,
                         dataset_type, normalize, template)
        super(MeshDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        if self._dataset_type == 'train':
            file_names = self._train_names
        elif self._dataset_type == 'test':
            file_names = self._test_names
        elif self._dataset_type == 'val':
            file_names = self._val_names
        else:
            raise Exception("train, val and test are supported dataset types")
        return file_names

    @property
    def processed_file_names(self):
        return self._processed_files

    def download(self):
        pass

    def process(self):
        for i, fname in tqdm.tqdm(enumerate(self.raw_file_names)):
            mesh_verts = self.load_mesh(fname)

            if self._normalize:
                mesh_verts = (mesh_verts - self.mean) / self.std

            data = Data(x=mesh_verts)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            fname_split = fname.split(os.sep)
            if len(fname_split) > 1:
                fname = '_'.join(fname_split)
            torch.save(data, os.path.join(self.processed_dir,
                                          fname[:-4] + '.pt'))

    def get(self, idx):
        filename = self.raw_file_names[idx]
        return torch.load(os.path.join(self.processed_dir,
                                       filename[:-4] + '.pt'))

    def len(self):
        return len(self.processed_file_names)


class MeshInMemoryDataset(MixinMeshDataset, InMemoryDataset):
    def __init__(self, root, precomputed_storage_path='precomputed',
                 dataset_type='train', normalize=True,
                 transform=None, pre_transform=None, template=None):

        self._initialize(root, precomputed_storage_path,
                         dataset_type, normalize, template)
        super(MeshInMemoryDataset, self).__init__(root, transform,
                                                  pre_transform)

        if dataset_type == 'train':
            data_path = self.processed_paths[0]
        elif dataset_type == 'test':
            data_path = self.processed_paths[1]
        elif dataset_type == 'val':
            data_path = self.processed_paths[2]
        else:
            raise Exception("train, val and test are supported data types")

        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    @property
    def raw_file_names(self):
        return 'mesh_data.zip'

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt', 'val.pt']

    def download(self):
        pass

    def _process_set(self, files_list):        
        dataset = []
        for fname in tqdm.tqdm(files_list):
            mesh_verts = self.load_mesh(fname)

            if self._normalize:
                mesh_verts = (mesh_verts - self.mean) / self.std

            data = Data(x=mesh_verts)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            dataset.append(data)
        return dataset

    def process(self):
        train_data = self._process_set(self._train_names)
        torch.save(self.collate(train_data), self.processed_paths[0])
        test_data = self._process_set(self._test_names)
        torch.save(self.collate(test_data), self.processed_paths[1])
        val_data = self._process_set(self._val_names)
        torch.save(self.collate(val_data), self.processed_paths[2])


if __name__ == '__main__':
    import argparse
    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configurations/default.yaml',
                        help="Path to the configuration file.")
    opts = parser.parse_args()
    conf = utils.get_config(opts.config)

    tr_loader, val_loader, te_loader, norm_dict = get_data_loaders(conf, None)