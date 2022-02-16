import csv
from glob import glob
import os
from tqdm import tqdm

dataset='RESIDE'
csv_files = glob(f'output_{dataset}/*/*.csv')

f = open(f'output_{dataset}_entropy.csv',"w", newline='')
wr = csv.writer(f)
for csv_file in tqdm(csv_files):
    max_line=None
    
    f = open(csv_file,'r')
    file_name = os.path.basename(csv_file)
    rdr = csv.reader(f)
    for line in rdr:
        step, mean_entropy, psnr, ssim = line
        
        
        if float(step) < 3:
            max_line=line
            continue
        
        if float(mean_entropy) > float(max_line[1]):
            max_line = line
        
        # if float(mean_entropy) > 7.35:
        #     max_line = line
        #     break

    wr.writerow([file_name]+max_line)
    f.close()
f.close()
            
            
        
    

# f = open('example.csv','r')
# rdr = csv.reader(f)
 
# for line in rdr:
#     print(line)
 
# f.close()