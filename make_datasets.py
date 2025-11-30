import numpy as np
import os
import tifffile
import h5py


def read_bil_image(filename, C, H, W, dtype=np.int16):
    try:
        with open(filename, 'rb') as f:
            f.seek(0, 0)
            result = np.empty((C, H, W), dtype=dtype)            
            for h in range(H):
                line_data = np.fromfile(f, dtype=dtype, count=C * W)
                result[:, h, :] = line_data.reshape(C, W)                
        negative_mask = np.any(result < 0, axis=0)
        result[:, negative_mask] = 0        
        return result
    
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {filename}")
    except IOError as e:
        raise IOError(f"读取文件时发生错误: {str(e)}")


def read_bsq_image(filename, C, H, W, dtype=np.int16):
    try:
        with open(filename, 'rb') as f:
            f.seek(0, 0)            
            result = np.empty((C, H, W), dtype=dtype)            
            for c in range(C):
                band_data = np.fromfile(f, dtype=dtype, count=H * W)                
                result[c, :, :] = band_data.reshape(H, W)                
        negative_mask = np.any(result < 0, axis=0)  
        result[:, negative_mask] = 0        
        return result
    
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {filename}")
    except IOError as e:
        raise IOError(f"读取文件时发生错误: {str(e)}")


def split_3d_matrix(matrix, P=100):
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 3:
        raise ValueError("输入必须是三维numpy矩阵（维度为CxHxW）")
    
    C, H, W = matrix.shape

    num_rows = (H + P - 1) // P
    num_cols = (W + P - 1) // P
    
    patches = []
    
    for i in range(num_rows):
        for j in range(num_cols):
            if i == num_rows - 1 and H % P != 0:
                h_start = H - P
            else:
                h_start = i * P
            h_end = h_start + P
            
            if j == num_cols - 1 and W % P != 0:
                w_start = W - P
            else:
                w_start = j * P
            w_end = w_start + P
            
            patch = matrix[:, h_start:h_end, w_start:w_end]
            patches.append(patch)
    
    return patches


def save_imagettes(matrix_list, bands=(3, 2, 1), output_path="."):
    os.makedirs(output_path, exist_ok=True)
    
    for band in bands:
        if not (0 <= band < 6):
            raise ValueError(f"无效的波段号 {band}，必须在0-5范围内")
        
    for idx, matrix in enumerate(matrix_list):
        if matrix.ndim != 3 or matrix.shape[0] != 6:
            raise ValueError(f"列表中第{idx}个矩阵维度不正确，应为6xPxP")
        
        selected_bands = matrix[bands, :, :]
        C, H, W = selected_bands.shape 

        normalized = np.empty_like(selected_bands, dtype=np.float32)

        for i in range(C):
            band_data = selected_bands[i, :, :]
            min_val = band_data.min()
            max_val = band_data.max()

            if max_val == min_val:
                normalized[i, :, :] = 0
            else:
                normalized[i, :, :] = ((band_data - min_val) / (max_val - min_val)) * 255

        uint8_img = normalized.astype(np.uint8)

        filename = os.path.join(output_path, f"imagette_{idx}.tif")

        tifffile.imwrite(filename, uint8_img.transpose(1, 2, 0))
    
    print(f"成功保存 {len(matrix_list)} 个图像到 {output_path}")


def filter_matrix_lists(list_a, list_b, list_c, list_d):

    assert len(list_a) == len(list_b) == len(list_c) == len(list_d), "四个输入列表的长度必须相同"
    
    result_a, result_b, result_c, result_d = [], [], [], []
    
    for idx in range(len(list_a)):
        mat_a = list_a[idx]
        mat_b = list_b[idx]
        mat_c = list_c[idx]
        mat_d = list_d[idx]
        
        store_flag = True
        
        for mat in [mat_a, mat_b, mat_c, mat_d]:
            C, H, W = mat.shape
            total_pixels = H * W
            
            zero_vector_count = np.sum(np.all(mat == 0, axis=0))
            
            if zero_vector_count > total_pixels / 3:
                store_flag = False
                break
        
        if store_flag:
            result_a.append(mat_a)
            result_b.append(mat_b)
            result_c.append(mat_c)
            result_d.append(mat_d)
    
    return result_a, result_b, result_c, result_d


def save_matrices_to_h5(list1, list2, list3, list4, file_path, 
                        dataset_names=['modis_ref', 'modis_tar', 'landsat_ref', 'landsat_tar']):
    if len(list1) != len(list2) or len(list1) != len(list3) or len(list1) != len(list4):
        raise ValueError("所有输入列表必须具有相同的长度")
    
    if len(list1) > 0:
        _, C, H, W = (1,) + list1[0].shape
    
    try:
        with h5py.File(file_path, 'w') as hf:
            for i, (lst, name) in enumerate(zip(
                [list1, list2, list3, list4], dataset_names
            )):
                if len(lst) == 0:
                    print(f"警告: 列表{name}为空，将不创建对应数据集")
                    continue
                
                data_array = np.stack(lst, axis=0)
                
                hf.create_dataset(name, data=data_array, dtype=data_array.dtype)
                print(f"已创建数据集 '{name}'，形状: {data_array.shape}")
        
        print(f"所有数据已成功保存到 {file_path}")
        
    except IOError as e:
        raise IOError(f"写入文件时发生错误: {str(e)}")
    
    
def cia_info():
    cia_date_list = [
        '2001_281_08oct',
        '2001_290_17oct',
        '2001_306_02nov',
        '2001_313_09nov',
        '2001_329_25nov',
        '2001_338_04dec',
        '2002_005_05jan',
        '2002_012_12jan',
        '2002_044_13feb',
        '2002_053_22feb',
        '2002_069_10mar',
        '2002_076_17mar',
        '2002_092_02apr',
        '2002_101_11apr',
        '2002_108_18apr',
        '2002_117_27apr',
        '2002_124_04may'
    ]
    cia_landsat_list = [
        'L71093084_08420011007_HRF_modtran_surf_ref_agd66',
        'L71093084_08420011016_HRF_modtran_surf_ref_agd66',
        'L71093084_08420011101_HRF_modtran_surf_ref_agd66',
        'L71093084_08420011108_HRF_modtran_surf_ref_agd66',
        'L71093084_08420011124_HRF_modtran_surf_ref_agd66',
        'L71093084_08420011203_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020104_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020111_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020212_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020221_HRF_modtran_surf_ref_agd66',
        'L72093084_08420020309_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020316_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020401_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020410_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020417_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020426_HRF_modtran_surf_ref_agd66',
        'L71093084_08420020503_HRF_modtran_surf_ref_agd66', 
    ]
    cia_landsat_sufix = '.bil'
    cia_landsat_dir = '/mnt/e/data - source/stf data/Landsat_and_MODIS_data_for_the_Coleambally_Irrigation_Area-W_ks3jCn-/data/CIA/Landsat/'
    cia_modis_list = [
        'MOD09GA_A2001281.sur_refl',
        'MOD09GA_A2001290.sur_refl',
        'MOD09GA_A2001306.sur_refl',
        'MOD09GA_A2001313.sur_refl',
        'MOD09GA_A2001329.sur_refl',
        'MOD09GA_A2001338.sur_refl',
        'MOD09GA_A2002005.sur_refl',
        'MOD09GA_A2002012.sur_refl',
        'MOD09GA_A2002044.sur_refl',
        'MOD09GA_A2002053.sur_refl',
        'MOD09GA_A2002069.sur_refl',
        'MOD09GA_A2002076.sur_refl',
        'MOD09GA_A2002092.sur_refl',
        'MOD09GA_A2002101.sur_refl',
        'MOD09GA_A2002108.sur_refl',
        'MOD09GA_A2002117.sur_refl',
        'MOD09GA_A2002124.sur_refl',
    ]
    cia_modis_sufix = '.int'
    cia_modis_dir = '/mnt/e/data - source/stf data/Landsat_and_MODIS_data_for_the_Coleambally_Irrigation_Area-W_ks3jCn-/data/CIA/MODIS/'
    imc, imh, imw = 6, 2040, 1720
    train_set_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    test_set_indexes = [9, 10, 11, 12, 13, 14, 15]
    result_path = '/mnt/e/data - source/stf data/Landsat_and_MODIS_data_for_the_Coleambally_Irrigation_Area-W_ks3jCn-/data/CIA/'
    return cia_date_list, cia_landsat_list, cia_landsat_sufix, cia_landsat_dir, cia_modis_list, cia_modis_sufix, cia_modis_dir, imc, imh, imw, train_set_indexes, test_set_indexes, result_path
    
    
def lgc_info():
    lgc_date_list = ['2004_107_Apr16', '2004_123_May02', '2004_187_Jul05', '2004_219_Aug06', '2004_235_Aug22', '2004_299_Oct25', '2004_331_Nov26', '2004_347_Dec12', '2004_363_Dec28', '2005_013_Jan13', '2005_029_Jan29', '2005_045_Feb14', '2005_061_Mar02', '2005_093_Apr03']
    lgc_landsat_list = [
        '20040416_TM',
        '20040502_TM',
        '20040705_TM',
        '20040806_TM',
        '20040822_TM',
        '20041025_TM',
        '20041126_TM',
        '20041212_TM',
        '20041228_TM',
        '20050113_TM',
        '20050129_TM',
        '20050214_TM',
        '20050302_TM',
        '20050403_TM',
    ]
    lgc_landsat_sufix = '.int'
    lgc_landsat_dir = '/mnt/e/data - source/stf data/Landsat_and_MODIS_data_for_the_Lower_Gwydir_Catchment_study_site-Umpuxcm0-/data/LGC/Landsat/'
    lgc_modis_list = [
        'MOD09GA_A2004107.sur_refl',
        'MOD09GA_A2004123.sur_refl',
        'MOD09GA_A2004187.sur_refl',
        'MOD09GA_A2004219.sur_refl',
        'MOD09GA_A2004235.sur_refl',
        'MOD09GA_A2004299.sur_refl',
        'MOD09GA_A2004331.sur_refl',
        'MOD09GA_A2004347.sur_refl',
        'MOD09GA_A2004363.sur_refl',
        'MOD09GA_A2005013.sur_refl',
        'MOD09GA_A2005029.sur_refl',
        'MOD09GA_A2005045.sur_refl',
        'MOD09GA_A2005061.sur_refl',
        'MOD09GA_A2005093.sur_refl',
    ]
    lgc_modis_sufix = '.int'
    lgc_modis_dir = '/mnt/e/data - source/stf data/Landsat_and_MODIS_data_for_the_Lower_Gwydir_Catchment_study_site-Umpuxcm0-/data/LGC/MODIS/'
    imc, imh, imw = 6, 2720, 3200
    train_set_indexes = [0, 1, 2, 3, 4, 5, 6]
    test_set_indexes = [7, 8, 9, 10, 11, 12]
    result_path = '/mnt/e/data - source/stf data/Landsat_and_MODIS_data_for_the_Lower_Gwydir_Catchment_study_site-Umpuxcm0-/data/LGC/'
    return lgc_date_list, lgc_landsat_list, lgc_landsat_sufix, lgc_landsat_dir, lgc_modis_list, lgc_modis_sufix, lgc_modis_dir, imc, imh, imw, train_set_indexes, test_set_indexes, result_path
    
    
def data_processing_for_cia_or_lgc(P=100, cia_or_lgc: bool = True, do_visualize: bool = False):
    if cia_or_lgc:
        date_list, landsat_list, landsat_sufix, landsat_dir, modis_list, modis_sufix, modis_dir, imc, imh, imw, train_set_indexes, test_set_indexes, result_path = cia_info()
    else:
        date_list, landsat_list, landsat_sufix, landsat_dir, modis_list, modis_sufix, modis_dir, imc, imh, imw, train_set_indexes, test_set_indexes, result_path = lgc_info()  
    
    landsat_imgs = []   
    for idx in range(len(date_list)):
        print('==> processing for landsat image of date ' + date_list[idx])
        cur_work_dir = landsat_dir + date_list[idx] + '/'
        if cia_or_lgc:
            data_mat = read_bil_image(cur_work_dir + landsat_list[idx] + landsat_sufix, imc, imh, imw)
        else:
            data_mat = read_bsq_image(cur_work_dir + landsat_list[idx] + landsat_sufix, imc, imh, imw)
        img_lst = split_3d_matrix(data_mat, P=P)
        landsat_imgs.append(img_lst)
        if do_visualize:
            divide_path = cur_work_dir + 'divide/'
            os.makedirs(divide_path, exist_ok=True)
            save_imagettes(img_lst, bands=(3, 2, 1), output_path=divide_path)
    
    modis_imgs = []
    for idx in range(len(date_list)):
        print('==> processing for modis image of date ' + date_list[idx])
        cur_work_dir = modis_dir + date_list[idx] + '/'
        data_mat = read_bsq_image(cur_work_dir + modis_list[idx] + modis_sufix, imc, imh, imw)
        img_lst = split_3d_matrix(data_mat, P=P)
        modis_imgs.append(img_lst)
        if do_visualize:
            divide_path = cur_work_dir + 'divide/'
            os.makedirs(divide_path, exist_ok=True)
            save_imagettes(img_lst, bands=(3, 2, 1), output_path=divide_path)
    
    # => make train dataset
    lst1, lst2, lst3, lst4 = [], [], [], []
    for idx in train_set_indexes:
        lt1, lt2, lt3, lt4 = filter_matrix_lists(modis_imgs[idx], modis_imgs[idx + 1], landsat_imgs[idx], landsat_imgs[idx + 1])
        if 0 == idx:
            lst1, lst2, lst3, lst4 = lt1, lt2, lt3, lt4
        else:
            lst1 += lt1
            lst2 += lt2
            lst3 += lt3
            lst4 += lt4
    save_matrices_to_h5(lst1, lst2, lst3, lst4, result_path + 'train_set.h5')
    
    # => make test dataset    
    lst1, lst2, lst3, lst4 = [], [], [], []
    for idx in test_set_indexes:
        lt1, lt2, lt3, lt4 = filter_matrix_lists(modis_imgs[idx], modis_imgs[idx + 1], landsat_imgs[idx], landsat_imgs[idx + 1])
        if 0 == idx:
            lst1, lst2, lst3, lst4 = lt1, lt2, lt3, lt4
        else:
            lst1 += lt1
            lst2 += lt2
            lst3 += lt3
            lst4 += lt4
    save_matrices_to_h5(lst1, lst2, lst3, lst4, result_path + 'test_set.h5')
            

if '__main__' == __name__:
    data_processing_for_cia_or_lgc(100, True, False)  # cia
    data_processing_for_cia_or_lgc(100, False, True)  # lgc
    
