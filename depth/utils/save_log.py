import os
from glob import glob
import csv
import numpy as np

def write_csv(dataRoot, metrics_name, input_name, log_list):
    file_name = os.path.basename(input_name)[:-4]
    dir_name = input_name.split('\\')[-2]
    path = f'{dataRoot}/{metrics_name}/{dir_name}'
    
    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path, file_name+'.csv')

    csv_file = open(save_path,'w',newline='')
    csv_wr = csv.writer(csv_file)
    csv_wr.writerow(['stage', 'step_beta', 'cur_val', 'diff_val', 'clear_val', 'psnr', 'ssim'])
    csv_wr.writerows(log_list)
    
def write_csv_depth_err(dataRoot, input_name, log_list):
    file_name = os.path.basename(input_name)[:-4]
    dir_name = input_name.split('\\')[-2]
    path = f'{dataRoot}/statistic/{dir_name}'
    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path, file_name+'.csv')
    with open(save_path, 'w', newline='') as csv_file:
        csv_wr = csv.writer(csv_file)
        csv_wr.writerow(['stage', 'step_beta', 'psnr', 'ssim', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'])
        csv_wr.writerows(log_list)
        
def write_csv_netD_err(dataRoot, epoch, input_name, log_list):
    file_name = os.path.basename(input_name)[:-4]
    dir_name = input_name.split('\\')[-2]
    path = f'{dataRoot}/statistic/epoch_{epoch}/{dir_name}'
    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path, file_name+'.csv')
    with open(save_path, 'w', newline='') as csv_file:
        csv_wr = csv.writer(csv_file)
        csv_wr.writerow(['step', 'beta', 'psnr', 'ssim'])
        csv_wr.writerows(log_list)
    



def main(path):
    csv_list = glob(f'{path}/*.csv')
    
    psnr_sum=0
    for csv_name in csv_list:
        csv_file = open(csv_name, 'r', newline='')
        csv_reader = csv.reader(csv_file)
        
        data = list(csv_reader)
        index, data = data[0],data[1:]
        for i,row in enumerate(data):
            if float(row[3]) < 0 or i == 99:
                psnr_sum+=float(data[i-1][4])
                break
    
    print(psnr_sum/len(csv_list))

def main2(path):
    csv_list = glob(f'{path}/*.csv')
    psnr_sum = 0
    for csv_name in csv_list:
        csv_file = open(csv_name, 'r', newline='')
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)
        index, data = data[0],data[1:]
        data = np.array(data).astype(np.float64)
        psnr_sum+=np.max(data[:,4])
    print(psnr_sum/len(csv_list))


if __name__=='__main__':
    path = 'D:/data/NH_Haze/train/hazy'
    main(path)
    print()
    main2(path)