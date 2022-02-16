import csv
from glob import glob
import os
from tqdm import tqdm

dataset='NYU'
csv_files = glob(f'output_{dataset}/*.csv')

ff = open(f'output_{dataset}_psnr.csv',"w", newline='')
fff = open(f'output_{dataset}_entropy.csv',"w", newline='')
wr = csv.writer(ff)
wrr = csv.writer(fff)
for csv_file in tqdm(csv_files):
    psnr_max_line=0
    entropy_max_line=0
    
    
    f = open(csv_file,'r')
    file_name = os.path.basename(csv_file)
    rdr = csv.reader(f)
    for line in rdr:
        step, mean_entropy, max_entropy, min_entropy, psnr, ssim = line
        if step == '0':
            continue
        if step == '1':
            psnr_max_line = line
            entropy_max_line = line
        if float(psnr) > float(psnr_max_line[4]):
            psnr_max_line = line
        if float(mean_entropy) > float(entropy_max_line[1]):
            entropy_max_line = line
    
    wr.writerow([file_name]+psnr_max_line)
    wrr.writerow([file_name]+entropy_max_line)
    f.close()
ff.close()
fff.close()
            
            
        
    

# f = open('example.csv','r')
# rdr = csv.reader(f)
 
# for line in rdr:
#     print(line)
 
# f.close()