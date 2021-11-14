import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv_all(dataRoot, header):
    all_df = {}
    for path in glob(dataRoot + '/*'):
        pathName = os.path.basename(path)
        for file in glob(path + '/*'):
            fileName = os.path.basename(file)[:-4]
            dfName = pathName + '_' + fileName
            df = pd.read_csv(file)
            df = pd.DataFrame(df, columns=header)
            df.set_index('stage', inplace=True)
            all_df[dfName] = df
    
    # return pd.concat(all_df, ignore_index=True)
    return all_df

def data_plot(dfName, df, labels):
    plt.title(dfName)
    for label in labels:
        df.plot(kind='line', x='stage', y=label, ax=plt.gca())
    plt.show()

if __name__ == '__main__':
    dataRoot = 'D:/programming/GitHub/RUS_Dehazing/Entropy_statistic'
    header = ['stage', 'step_beta', 'cur_etp', 'diff_etp', 'psnr', 'ssim']
    all_df_dict = read_csv_all(dataRoot, header)
    
    corr_etp_psnr, corr_etp_ssim, corr_psnr_ssim, corr_diff_psnr, corr_diff_ssim = [], [], [], [], []
    for dfName, df in all_df_dict.items():
        # Correlation (변수 별 상관 관계)
        corr_etp_psnr.append(df['cur_etp'].corr(df['psnr']))
        corr_etp_ssim.append(df['cur_etp'].corr(df['ssim']))
        corr_psnr_ssim.append(df['psnr'].corr(df['ssim']))
        corr_diff_psnr.append(df['diff_etp'].corr(df['psnr']))
        corr_diff_ssim.append(df['diff_etp'].corr(df['ssim']))
    
    
    corr_etp_psnr = np.std(np.array(corr_etp_psnr))
    corr_etp_ssim = np.std(np.array(corr_etp_ssim))
    corr_psnr_ssim = np.std(np.array(corr_psnr_ssim))
    corr_diff_psnr = np.std(np.array(corr_diff_psnr))
    corr_diff_ssim = np.std(np.array(corr_diff_ssim))
    
    print(corr_etp_psnr)
    print(corr_etp_ssim)
    print(corr_psnr_ssim)
    print(corr_diff_psnr)
    print(corr_diff_ssim)
    
    
        