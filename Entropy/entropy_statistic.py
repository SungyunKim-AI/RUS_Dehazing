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
            df = df.drop([df.index[0]])
            # df.set_index('stage', inplace=True)
            all_df[dfName] = df
    
    # return pd.concat(all_df, ignore_index=True)
    return all_df

def data_plot(dfName, df, labels):
    plt.title(dfName)
    for label in labels:
        df.plot(kind='line', y=label, ax=plt.gca())
    plt.show()
    
def getCorrelation(all_df_dict):
    corr_etp_psnr, corr_etp_ssim, corr_psnr_ssim, corr_diff_psnr, corr_diff_ssim = [], [], [], [], []
    for dfName, df in all_df_dict.items():
        # Correlation (변수 별 상관 관계)
        print(df.corr())
        exit(0)
        
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

# PSNR 최대값일 때 diff_etp값과 cur_etp값
def getMaxPSNR_SSIM(all_df_dict):
    psnr_stage, psnr_cur_etp, psnr_diff_etp = [], [], []
    ssim_stage, ssim_cur_etp, ssim_diff_etp = [], [], []
    for dfName, df in all_df_dict.items():
        psnr_stage.append(df.loc[df['psnr'].idxmax()]['stage'])
        psnr_cur_etp.append(df.loc[df['psnr'].idxmax()]['cur_etp'])
        psnr_diff_etp.append(df.loc[df['psnr'].idxmax()]['diff_etp'])
        
        ssim_stage.append(df.loc[df['ssim'].idxmax()]['stage'])
        ssim_cur_etp.append(df.loc[df['ssim'].idxmax()]['cur_etp'])
        ssim_diff_etp.append(df.loc[df['ssim'].idxmax()]['diff_etp'])
        
        
    df_stage = pd.DataFrame(psnr_stage)
    df_etp = pd.DataFrame(psnr_cur_etp)
    df_diff = pd.DataFrame(psnr_diff_etp)
    
    print("PSNR")
    print(f'stage : mean={df_stage.mean()[0]}, median={df_stage.median()[0]}, std={df_stage.std()[0]}')
    print(f'etp : mean={df_etp.mean()[0]}, median={df_etp.median()[0]}, std={df_etp.std()[0]}')
    print(f'diff : mean={df_diff.mean()[0]}, median={df_diff.median()[0]}, std={df_diff.std()[0]}')
    print()
    
    df_stage = pd.DataFrame(ssim_stage)
    df_etp = pd.DataFrame(ssim_cur_etp)
    df_diff = pd.DataFrame(ssim_diff_etp)
    
    print("SSIM")
    print(f'stage : mean={df_stage.mean()[0]}, median={df_stage.median()[0]}, std={df_stage.std()[0]}')
    print(f'etp : mean={df_etp.mean()[0]}, median={df_etp.median()[0]}, std={df_etp.std()[0]}')
    print(f'diff : mean={df_diff.mean()[0]}, median={df_diff.median()[0]}, std={df_diff.std()[0]}')
    

if __name__ == '__main__':
    # dataRoot = 'D:/programming/GitHub/RUS_Dehazing/Entropy_statistic'
    dataRoot = '/Users/sungyoon-kim/Downloads/Entropy_statistic'
    header = ['stage', 'step_beta', 'cur_etp', 'diff_etp', 'psnr', 'ssim']
    all_df_dict = read_csv_all(dataRoot, header)
    
    getCorrelation(all_df_dict)
    # getMaxPSNR_SSIM(all_df_dict)
    
    
    
        