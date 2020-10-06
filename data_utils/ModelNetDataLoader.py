import numpy as np
import warnings
import os
from torch.utils.data import Dataset
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
warnings.filterwarnings('ignore')



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, model_name='modelnet6', extension='.npy', npoint=1024, split='train', uniform=False, normal_channel=True, class_in_filename=False):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, f'{model_name}_shape_names.txt')
        self.extension = extension

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        if class_in_filename:
            shape_ids = {}
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, f'{model_name}_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, f'{model_name}_test.txt'))]

            assert (split == 'train' or split == 'test')
            shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
            # list of (shape_name, shape_txt_file_path) tuple
            self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + extension) for i
                             in range(len(shape_ids[split]))]

        else:
            self.datapath = [(os.path.split(filename)[0], os.path.join(self.root, filename))
                             for filename in [line.rstrip() for line in open(os.path.join(self.root, f'{model_name}_{split}.txt'))]]

        print('The size of %s data is %d' % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def read_file_by_extension(self, file_path: str) -> np.ndarray:
        """
        read file and convert to numpy according to extension
        :param file_path: path to 3d file
        :return: ndarry of shape (N, D)
        """
        if self.extension == ".txt":
            point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)
        elif self.extension == ".ply":
            pcd = o3d.io.read_point_cloud(file_path, format='ply')
            point_set = np.asanyarray(pcd.points).astype(np.float32)
        elif self.extension == ".npy":
            point_set = np.load(file_path)
        else:
            raise Exception(f"extension {self.extension} not supported")
        return point_set

    def _get_item(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        cls = np.array([cls]).astype(np.int32)
        point_set = self.read_file_by_extension(fn[1])
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('drive/My Drive/pointnet/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)