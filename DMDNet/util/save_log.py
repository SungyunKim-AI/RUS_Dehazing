import os
import glob
import csv
import numpy as np

def write_csv(input_name, log_list):
    csv_name = os.path.basename(input_name[0])[:-3] + 'csv'
    save_path = os.path.join('C:/Users/IIPL/Desktop/RUS_Dehazing/DMDNet/output_log/NIQUE', csv_name)
    csv_file = open(save_path,'w',newline='')
    csv_wr = csv.writer(csv_file)
    csv_wr.writerow(['stage','step_beta','cur_val','diff_val','psnr','ssim'])
    csv_wr.writerows(log_list)

def main(path):
    csv_list = glob.glob(f'{path}/*.csv')
    
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
    csv_list = glob.glob(f'{path}/*.csv')
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