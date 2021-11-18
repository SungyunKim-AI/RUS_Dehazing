import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv_all(dataRoot, metrics_name, header):
    all_df_dict = {}
    path = os.path.join(dataRoot, metrics_name)
    for file in glob(path + '/*'):
        fileName = os.path.basename(file)[:-4]
        df = pd.read_csv(file)
        df = pd.DataFrame(df, columns=header)
        df = df.drop([df.index[0]])
        # df.set_index('stage', inplace=True)
        all_df_dict[fileName] = df
    
    # return pd.concat(all_df, ignore_index=True)
    return all_df_dict

def data_plot(dfName, df, labels):
    plt.title(dfName)
    for label in labels:
        df.plot(kind='line', y=label, ax=plt.gca())
    plt.show()

def all_df_plot(all_df_dict):
    plt.title("Mean of All Data")
    cur_val, diff_val, psnr, ssim = [], [], [], []
    
    meanDF = pd.DataFrame()
    for row in range(1, 300):
        tempDF = pd.DataFrame()
        for dfName, df in all_df_dict.items():
            temp = pd.DataFrame({'cur_val':[df.loc[[row]]['cur_val']],
                                'diff_val':[df.loc[[row]]['diff_val']], 
                                'psnr':[df.loc[[row]]['psnr']], 
                                'ssim':[df.loc[[row]]['ssim']]})
            tempDF = pd.concat((tempDF, temp), ignore_index=True)

        meanTemp = pd.DataFrame({'cur_val':[tempDF['cur_val'].mean()],
                                'diff_val':[tempDF['diff_val'].mean()], 
                                'psnr':[tempDF['psnr'].mean()], 
                                'ssim':[ tempDF['ssim'].mean()]})
        
        meanDF = pd.concat((meanDF, meanTemp), ignore_index=True)
    
    data_plot('Mean of All data', meanDF, ['cur_val', 'diff_val', 'psnr', 'ssim'])
    
    
def getCorrelation(all_df_dict):
    corr_cur_psnr, corr_cur_ssim, corr_psnr_ssim, corr_diff_psnr, corr_diff_ssim = [], [], [], [], []
    for dfName, df in all_df_dict.items():
        # Correlation (변수 별 상관 관계)        
        corr_cur_psnr.append(df['cur_val'].corr(df['psnr']))
        corr_cur_ssim.append(df['cur_val'].corr(df['ssim']))
        corr_psnr_ssim.append(df['psnr'].corr(df['ssim']))
        corr_diff_psnr.append(df['diff_val'].corr(df['psnr']))
        corr_diff_ssim.append(df['diff_val'].corr(df['ssim']))
    
    
    corr_cur_psnr = np.std(np.array(corr_cur_psnr))
    corr_cur_ssim = np.std(np.array(corr_cur_ssim))
    corr_psnr_ssim = np.std(np.array(corr_psnr_ssim))
    corr_diff_psnr = np.std(np.array(corr_diff_psnr))
    corr_diff_ssim = np.std(np.array(corr_diff_ssim))
    
    print(corr_cur_psnr)
    print(corr_cur_ssim)
    print(corr_psnr_ssim)
    print(corr_diff_psnr)
    print(corr_diff_ssim)

# PSNR 최대값일 때 diff_val값과 cur_val값
def getMaxPSNR_SSIM(all_df_dict):
    psnr_stage, psnr_cur_val, psnr_diff_val = [], [], []
    ssim_stage, ssim_cur_val, ssim_diff_val = [], [], []
    for dfName, df in all_df_dict.items():
        psnr_stage.append(df.loc[df['psnr'].idxmax()]['stage'])
        psnr_cur_val.append(df.loc[df['psnr'].idxmax()]['cur_val'])
        psnr_diff_val.append(df.loc[df['psnr'].idxmax()]['diff_val'])
        
        ssim_stage.append(df.loc[df['ssim'].idxmax()]['stage'])
        ssim_cur_val.append(df.loc[df['ssim'].idxmax()]['cur_val'])
        ssim_diff_val.append(df.loc[df['ssim'].idxmax()]['diff_val'])
        
        
    df_stage = pd.DataFrame(psnr_stage)
    df_cur = pd.DataFrame(psnr_cur_val)
    df_diff = pd.DataFrame(psnr_diff_val)
    
    print("PSNR")
    print(f'stage : mean={df_stage.mean()[0]}, median={df_stage.median()[0]}, std={df_stage.std()[0]}')
    print(f'cur : mean={df_cur.mean()[0]}, median={df_cur.median()[0]}, std={df_cur.std()[0]}')
    print(f'diff : mean={df_diff.mean()[0]}, median={df_diff.median()[0]}, std={df_diff.std()[0]}')
    print()
    
    df_stage = pd.DataFrame(ssim_stage)
    df_cur = pd.DataFrame(ssim_cur_val)
    df_diff = pd.DataFrame(ssim_diff_val)
    
    print("SSIM")
    print(f'stage : mean={df_stage.mean()[0]}, median={df_stage.median()[0]}, std={df_stage.std()[0]}')
    print(f'cur : mean={df_cur.mean()[0]}, median={df_cur.median()[0]}, std={df_cur.std()[0]}')
    print(f'diff : mean={df_diff.mean()[0]}, median={df_diff.median()[0]}, std={df_diff.std()[0]}')

def getMean_Max_PSNR_SSIM(all_df_dict):
    max_psnr, max_ssim = [], []
    for dfName, df in all_df_dict.items():
        max_psnr.append(df.loc[df['psnr'].idxmax()]['psnr'])
        max_ssim.append(df.loc[df['ssim'].idxmax()]['ssim'])
        
    max_psnr = pd.DataFrame(max_psnr).mean()[0]
    max_ssim = pd.DataFrame(max_ssim).mean()[0]
    
    print(f'max_psnr={max_psnr}, max_ssim={max_ssim}')    
    
def getDiff_val(all_df_dict):
    idx_psnr, idx_ssim = [], []
    for dfName, df in all_df_dict.items():
        idx = df.index[(df['diff_val'] < 0)].tolist()[0]
        idx_psnr.append(df.loc[idx-1]['psnr'])
        idx_ssim.append(df.loc[idx-1]['ssim'])
        
    idx_psnr = pd.DataFrame(idx_psnr).mean()[0]
    idx_ssim = pd.DataFrame(idx_ssim).mean()[0]
    
    print(f'idx_psnr={idx_psnr}, idx_ssim={idx_ssim}')

def getMax_val(all_df_dict, label, val):
    idx_psnr, idx_ssim = [], []
    for dfName, df in all_df_dict.items():
        idx = df.index[(df[label] <= val)].tolist()[0]
        idx_psnr.append(df.loc[idx]['psnr'])
        idx_ssim.append(df.loc[idx]['ssim'])
        
    idx_psnr = pd.DataFrame(idx_psnr)
    idx_ssim = pd.DataFrame(idx_ssim)
    
    print(f'{val} : idx_psnr={idx_psnr.mean()[0]}({idx_psnr.std()[0]}), idx_ssim={idx_ssim.mean()[0]}({idx_ssim.std()[0]})')
    

if __name__ == '__main__':
    dataRoot = 'C:/Users/IIPL/Desktop/RUS_Dehazing/DMDNet/output_log'
    # dataRoot = '/Users/sungyoon-kim/Downloads/Entropy_statistic'
    header = ['stage', 'step_beta', 'cur_val', 'diff_val', 'psnr', 'ssim']
    all_df_dict = read_csv_all(dataRoot, 'NIQUE', header)
    
    # all_df_plot(all_df_dict)
    
    # getCorrelation(all_df_dict)
    # getMaxPSNR_SSIM(all_df_dict)
    
    getMean_Max_PSNR_SSIM(all_df_dict)
    # getDiff_0(all_df_dict)
    # getMax_val(all_df_dict, 'diff_val', 0.005)
    # getMax_val(all_df_dict, 'diff_val', 0.010)
    # getMax_val(all_df_dict, 'diff_val', 0.015)
    # getMax_val(all_df_dict, 'diff_val', 0.020)
    
    
    # getMax_val(all_df_dict, 'cur_val', 7.0)
    # getMax_val(all_df_dict, 'cur_val', 7.1)
    # getMax_val(all_df_dict, 'cur_val', 7.12)
    # getMax_val(all_df_dict, 'cur_val', 7.15)
    
    
    
    
    
        