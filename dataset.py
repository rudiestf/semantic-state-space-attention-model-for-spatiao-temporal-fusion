from torch.utils.data import Dataset
import h5py
import numpy as np
import imageio.v2 as imageio
import tifffile as tiff
import torch


class StfDataset(Dataset):
    def __init__(self, h5_file: str, patch_size: int, n_channel: int,
                 factor: float = 1.0 / 10000., dataset_names=['modis_tar', 'modis_ref', 'landsat_ref', 'landsat_tar']):
        super().__init__()
        self.ps = patch_size
        self.nc = n_channel
        self.fac = factor
        with h5py.File(h5_file, 'r') as f:
            datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
            for ds in datasets:
                assert ds in dataset_names, 'input dataset_names does not match the h5 file'
            self.modis_tar = torch.from_numpy(np.float32(f[dataset_names[0]][:])).view(-1, self.nc, self.ps, self.ps) * self.fac
            self.modis_ref = torch.from_numpy(np.float32(f[dataset_names[1]][:])).view(-1, self.nc, self.ps, self.ps) * self.fac
            self.landsat_ref = torch.from_numpy(np.float32(f[dataset_names[2]][:])).view(-1, self.nc, self.ps, self.ps) * self.fac
            self.landsat_tar = torch.from_numpy(np.float32(f[dataset_names[3]][:])).view(-1, self.nc, self.ps, self.ps) * self.fac
    
    def __getitem__(self, index):
        m_tar = self.modis_tar[index].float()
        m_ref = self.modis_ref[index].float()
        l_ref = self.landsat_ref[index].float()
        l_tar = self.landsat_tar[index].float()
        return m_tar, m_ref, l_ref, l_tar
    
    def __len__(self):
        return len(self.modis_tar)
    
    
class StfTextDataset(Dataset):
    def __init__(self, h5_file: str, patch_size: int, n_channel: int, h5_text_file: str, text_dim: int = 768,
                 factor: float = 1.0 / 10000., dataset_names=['modis_tar', 'modis_ref', 'landsat_ref', 'landsat_tar']):
        super().__init__()
        self.ps = patch_size
        self.nc = n_channel
        self.fac = factor
        with h5py.File(h5_file, 'r') as f:
            datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
            for ds in datasets:
                assert ds in dataset_names, 'input dataset_names does not match the h5 file'
            self.modis_tar = torch.from_numpy(np.float32(f[dataset_names[0]][:])).view(-1, self.nc, self.ps, self.ps) * self.fac
            self.modis_ref = torch.from_numpy(np.float32(f[dataset_names[1]][:])).view(-1, self.nc, self.ps, self.ps) * self.fac
            self.landsat_ref = torch.from_numpy(np.float32(f[dataset_names[2]][:])).view(-1, self.nc, self.ps, self.ps) * self.fac
            self.landsat_tar = torch.from_numpy(np.float32(f[dataset_names[3]][:])).view(-1, self.nc, self.ps, self.ps) * self.fac
        with h5py.File(h5_text_file, 'r') as ft:
            self.text_f = torch.from_numpy(np.float32(ft['text_features'][:])).view(-1, text_dim)
    
    def __getitem__(self, index):
        m_tar = self.modis_tar[index].float()
        m_ref = self.modis_ref[index].float()
        l_ref = self.landsat_ref[index].float()
        l_tar = self.landsat_tar[index].float()
        text_f = self.text_f[index].float()
        return m_tar, m_ref, l_ref, l_tar, text_f
    
    def __len__(self):
        return len(self.modis_tar)


class CAFEDataset(Dataset):
    def __init__(self, modis_tar, modis_ref, landsat_ref, landsat_tar):
        super(CAFEDataset, self).__init__()
        self.modis_tar = modis_tar
        self.modis_ref = modis_ref
        self.landsat_ref = landsat_ref
        self.landsat_tar = landsat_tar

    def __getitem__(self, index):
        batch_modis_tar = self.modis_tar[index]
        batch_modis_ref = self.modis_ref[index]
        batch_landsat_ref = self.landsat_ref[index]
        batch_landsat_tar = self.landsat_tar[index]
        return batch_modis_tar.float(), batch_modis_ref.float(), batch_landsat_ref.float(), batch_landsat_tar.float()

    def __len__(self):
        return self.landsat_tar.shape[0]
    
    
class CAFEDatasetWithText(Dataset):
    def __init__(self, modis_tar, modis_ref, landsat_ref, landsat_tar, text_f):
        super(CAFEDataset, self).__init__()
        self.modis_tar = modis_tar
        self.modis_ref = modis_ref
        self.landsat_ref = landsat_ref
        self.landsat_tar = landsat_tar
        self.text_f = text_f

    def __getitem__(self, index):
        batch_modis_tar = self.modis_tar[index]
        batch_modis_ref = self.modis_ref[index]
        batch_landsat_ref = self.landsat_ref[index]
        batch_landsat_tar = self.landsat_tar[index]
        batch_text_f = self.text_f[index]
        return batch_modis_tar.float(), batch_modis_ref.float(), batch_landsat_ref.float(), batch_landsat_tar.float(), batch_text_f.float()

    def __len__(self):
        return self.landsat_tar.shape[0]
    
    
def load_h5file(
    filename: str,
    output_dirs: [] = [], # type: ignore
    ):
    with h5py.File(filename, 'r') as f:
        datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        total_imgs = []
        for ds_name in datasets:
            data = f[ds_name][:]          # (18496, 6, 40, 40)
            total_imgs.append(data)
        # all_imgs = np.concatenate(total_imgs, axis=0)  (73984, 6, 40, 40)
        # print(all_imgs.shape)
    if 0 == len(output_dirs):
        return total_imgs
    assert len(output_dirs) == len(total_imgs), 'inconsistent number of datasets'
    for idx_i, data_imgs in enumerate(total_imgs):
        print('processing dataset with key of %s' % datasets[idx_i])
        for idx_j, item in enumerate(data_imgs):
            out_img_name = output_dirs[idx_i] + 'image_%d.tif' % idx_j
            tiff.imwrite(out_img_name, item)
        print('finished for dataset %d' % (idx_i + 1))
    return total_imgs


def get_train_and_test_subsets(
        filename: str,
):    
    landsat1, modis1, modis2, landsat2 = load_h5file(
        filename,
        [],
    )
    total_num = len(landsat1)
    assert total_num == len(modis1) and total_num == len(modis2) and total_num == len(landsat2), 'inconsisitent dataset length!'
    half_num = total_num // 2
    train_set = (landsat1[0: half_num], modis1[0: half_num], modis2[0: half_num], landsat2[0: half_num])
    test_set = (landsat1[half_num:], modis1[half_num:], modis2[half_num:], landsat2[half_num:])
    return CAFEDataset(train_set[2], train_set[1], train_set[0], train_set[3]), CAFEDataset(test_set[2], test_set[1], test_set[0], test_set[3])
    
    
if '__main__' == __name__:
    input_path_name = '/mnt/e/data - source/stf data/test_set_cia.h5'
    # main_dir = '/mnt/e/data - source/stf data/train_cafe_datasets/'
    # out_dirs = [
    #     main_dir + 'ds1/',
    #     main_dir + 'ds2/',
    #     main_dir + 'ds3/',
    #     main_dir + 'ds4/',
    # ]
    # load_h5file(in_path_name, out_dirs)
    # train_ds, test_ds = get_train_and_test_subsets(input_path_name)
    # print(len(train_ds))
    # print(len(train_ds))
    
    ds = StfDataset(input_path_name, 100, 6)
    ddd = ds[122]
    print(len(ds))
    
