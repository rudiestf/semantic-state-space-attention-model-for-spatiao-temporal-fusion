import os
import time
import glob
import re
import gc
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from dataset import *
from loss import *
from evaluation import *
from torch.utils.data import Subset

from s3amstf import S3AMSTF


def find_checkpoint(savedir):
    file_list = glob.glob(os.path.join(savedir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for m in file_list:
            result = re.findall(".*model_(.*).pth.*", m)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def main(
    # model parameters (choose one from the following)
    model = S3AMSTF(),
    model_path = 'S3AMSTF_LGC',  # modify according to model and dataset
    # dataset parameters
    train_data = '../train_set_lgc.h5',  # modify according to dataset
    test_data = '../test_set_lgc.h5',  # modify according to dataset
    train_text = '../train_set_lgc_text_nomic_embed_text.h5',  # modify according to dataset
    test_text = '../test_set_lgc_text_nomic_embed_text.h5',  # modify according to dataset
    patch_size = 100,
    nchannels = 6,
    # training parameters
    nepochs = 200,
    batch_size = 8,
    lr = 1e-4,
    cuda = torch.cuda.is_available(),
    # whether use subset, 0 means using the complete set
    random_subset_num = 0,
):
    savedir = 'model/' + model_path
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    log_header = [
        'epoch',
        'train_loss',
        'train_metrics...',
        'test_loss',
        'test_metrcis...'
    ]
    if not os.path.exists(os.path.join(savedir, 'log.csv')):
        with open(os.path.join(savedir, 'log.csv'), 'w') as f:
            f.write(','.join(log_header) + '\n')
            
    print("==> Generating data")
    train_set = StfTextDataset(train_data, patch_size, nchannels, train_text)
    test_set = StfTextDataset(test_data, patch_size, nchannels, test_text)
    if random_subset_num > 0:
        train_sub_idx = random.sample(range(0, len(train_set)), random_subset_num)
        test_sub_idx = random.sample(range(0, len(test_set)), random_subset_num)
        train_set = Subset(train_set, train_sub_idx)
        test_set = Subset(test_set, test_sub_idx)
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=batch_size, shuffle=False)

    print("==> Building model")
    model = model
    criterion = CELoss()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    print("==> Setting optimizer")
    optimizer = optim.AdamW(model.parameters(), lr=lr,)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # adjust learning rate

    initial_epoch = find_checkpoint(savedir=savedir)
    if initial_epoch > 0:
        print('==> Resuming by loading epoch %03d' % initial_epoch)
        state_dict = torch.load(os.path.join(savedir, 'model_%03d.pth' % initial_epoch))
        model.load_state_dict(state_dict)
        
    metrics = MetricAccumulator(nchannels, device=torch.device('cuda') if cuda else torch.device('cpu'))

    for epoch in range(initial_epoch, nepochs):
        gc.collect()

        # train
        start_time = time.time()
        metrics.reset()
        total_loss = 0.0
        model.train()
        for iteration, batch in enumerate(train_loader):
            modis_tar_batch, modis_ref_batch, landsat_ref_batch, landsat_tar_batch, text_batch = batch[0], batch[1], batch[2], batch[3], batch[4]
            if cuda:
                modis_tar_batch = modis_tar_batch.cuda()
                modis_ref_batch = modis_ref_batch.cuda()
                landsat_ref_batch = landsat_ref_batch.cuda()
                landsat_tar_batch = landsat_tar_batch.cuda()
                text_batch = text_batch.cuda()
            optimizer.zero_grad()
            out = model(modis_tar_batch, modis_ref_batch, landsat_ref_batch, text_batch)
            loss = criterion(out, landsat_tar_batch)
            total_loss += loss
            loss.backward()
            optimizer.step()
            batch_eval = metrics.update(out, landsat_tar_batch)
            print('epoch: %4d   %4d / %4d  loss = %2.6f, rmse = %.4f, cc = %.4f, ssim = %.4f, psnr = %.4f, sam = %.4f'
                  % (epoch + 1, iteration, len(train_set) / batch_size, loss.data,
                     batch_eval['rmse'][0], batch_eval['cc'][0], batch_eval['ssim'][0], batch_eval['psnr'][0], batch_eval['sam']))
        with torch.no_grad():
            epoch_eval = metrics.compute_epoch_metrics()
            print('train epoch: %4d loss = %2.6f, eval metrics: rmse = %.4f, cc = %.4f, ssim = %.4f, psnr = %.4f, sam = %.4f'
                  % (epoch + 1, total_loss / len(train_loader), epoch_eval['rmse'][0], epoch_eval['cc'][0], epoch_eval['ssim'][0], epoch_eval['psnr'][0], epoch_eval['sam']))
        torch.save(model.state_dict(), os.path.join(savedir, 'model_%03d.pth' % (epoch + 1)))
        print('train epcoh = %4d , time is %4.4f s\n' % (epoch + 1, time.time() - start_time))
        log = ['epoch=%d' % (epoch + 1)] + ['train_loss=%.6f' % loss.data.item()] + [metrics.metrics_to_string(epoch_eval, False)]
        scheduler.step()
        
        # test
        with torch.no_grad():
            metrics.reset()
            start_time = time.time()
            total_loss = 0.0
            model.eval()
            for iteration, batch in enumerate(test_loader):
                modis_tar_batch, modis_ref_batch, landsat_ref_batch, landsat_tar_batch, text_batch = batch[0], batch[1], batch[2], batch[3], batch[4]
                if cuda:
                    modis_tar_batch = modis_tar_batch.cuda()
                    modis_ref_batch = modis_ref_batch.cuda()
                    landsat_ref_batch = landsat_ref_batch.cuda()
                    landsat_tar_batch = landsat_tar_batch.cuda()
                    text_batch = text_batch.cuda()
                out = model(modis_tar_batch, modis_ref_batch, landsat_ref_batch, text_batch)
                loss = criterion(out, landsat_tar_batch)
                total_loss += loss
                batch_eval = metrics.update(out, landsat_tar_batch)
                print('epoch: %4d   %4d / %4d  loss = %2.6f, rmse = %.4f, cc = %.4f, ssim = %.4f, psnr = %.4f, sam = %.4f'
                    % (epoch + 1, iteration, len(test_set) / batch_size, loss.data,
                       batch_eval['rmse'][0], batch_eval['cc'][0], batch_eval['ssim'][0], batch_eval['psnr'][0], batch_eval['sam']))
                epoch_eval = metrics.compute_epoch_metrics()
            print('test epoch: %4d loss = %2.6f, eval metrics: rmse = %.4f, cc = %.4f, ssim = %.4f, psnr = %.4f, sam = %.4f'
                  % (epoch + 1, total_loss / len(test_loader), epoch_eval['rmse'][0], epoch_eval['cc'][0], epoch_eval['ssim'][0], epoch_eval['psnr'][0], epoch_eval['sam']))
            print('test epcoh = %4d , time is %4.4f s\n' % (epoch + 1, time.time() - start_time))            
        with open(os.path.join(savedir, 'log.csv'), 'a') as file:
            log += (['   test_loss: %.6f' % loss.data.item()] + [metrics.metrics_to_string(epoch_eval, True)])
            log = map(str, log)
            file.write(','.join(log) + '\n')


if __name__ == '__main__':
    main(
        # model parameters (choose one from the following)
        model = S3AMSTF(),
        model_path = 'S3AMSTF_CIA_ablate_p7',  # modify according to model and dataset
        # dataset parameters
        train_data = '../train_set_cia.h5',  # modify according to dataset
        test_data = '../test_set_cia.h5',  # modify according to dataset
        train_text = '../train_set_cia_text_nomic_embed_text_ablate_p7.h5',  # modify according to dataset
        test_text = '../test_set_cia_text_nomic_embed_text_ablate_p7.h5',  # modify according to dataset
        patch_size = 100,
        nchannels = 6,
        # training parameters
        nepochs = 200,
        batch_size = 8,
        lr = 1e-4,
        cuda = torch.cuda.is_available(),
        # whether use subset, 0 means using the complete set
        random_subset_num = 0,
    )
    main(
        # model parameters (choose one from the following)
        model = S3AMSTF(),
        model_path = 'S3AMSTF_LGC_ablate_p7',  # modify according to model and dataset
        # dataset parameters
        train_data = '../train_set_lgc.h5',  # modify according to dataset
        test_data = '../test_set_lgc.h5',  # modify according to dataset
        train_text = '../train_set_lgc_text_nomic_embed_text_ablate_p7.h5',  # modify according to dataset
        test_text = '../test_set_lgc_text_nomic_embed_text_ablate_p7.h5',  # modify according to dataset
        patch_size = 100,
        nchannels = 6,
        # training parameters
        nepochs = 200,
        batch_size = 8,
        lr = 1e-4,
        cuda = torch.cuda.is_available(),
        # whether use subset, 0 means using the complete set
        random_subset_num = 0,
    )
    
    # cfg_all = [
    #     # [True, True, True,],
    #     [True, False, False,],
    #     [False, True, False,],
    #     [False, False, True,],        
    #     [True, True, False,],
    #     [True, False, True,],
    #     [False, True, True,],              
    #     [False, False, False,],
    # ]
    # for cfg in cfg_all:
    #     main(
    #         model=S4AMSTF_ABLATION(cfg),
    #         model_path='S4AMSTF_ABLATION_%d_%d_%d_CIA' % (cfg[0], cfg[1], cfg[2]),
    #     )
