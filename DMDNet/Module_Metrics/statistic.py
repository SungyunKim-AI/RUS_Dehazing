import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv_all(dataRoot, metrics_name, beta):
    header = ['stage', 'step_beta', 'cur_val', 'diff_val', 'psnr', 'ssim']
    all_df_dict = {}
    path = os.path.join(dataRoot, metrics_name)
    for folder in glob(path + '/*'):
        if folder.startswith('.'):
            continue
        
        if float(os.path.basename(folder).split('_')[-1]) not in beta:
            continue
        
        for file in glob(folder + '/*.csv'):
            fileName = os.path.basename(file)[:-4]
            df = pd.read_csv(file)
            df = pd.DataFrame(df, columns=header)
            df = df.drop([df.index[0]])
            # df.set_index('stage', inplace=True)
            all_df_dict[fileName] = df
    
    return all_df_dict

def data_plot(dfName, df, labels):
    plt.title(dfName)
    for label in labels:
        df.plot(kind='line', y=label, ax=plt.gca())
    plt.show()

def all_df_plot(all_df_dict):
    plt.title("Mean of All Data")
    meanDF = pd.DataFrame(columns=['cur_val', 'diff_val', 'psnr', 'ssim'])
    len_df = len(list(all_df_dict.values())[0])
    for idx in range(1, len_df):
        tempDF = pd.DataFrame(columns=['cur_val', 'diff_val', 'psnr', 'ssim'])
        for dfName, df in all_df_dict.items():
            row = df.iloc[idx]['cur_val':'ssim']
            tempDF = tempDF.append({'cur_val':row[0], 'diff_val':row[1], 
                                    'psnr':row[2], 'ssim':row[3]}, ignore_index=True)

        meanRow = tempDF.mean()
        meanDF = meanDF.append({'cur_val':meanRow[0],
                                'diff_val':meanRow[1], 
                                'psnr':meanRow[2], 
                                'ssim':meanRow[3]}, ignore_index=True)
    
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
    print(f'stage : mean={df_stage.mean()[0]:.3f}, median={df_stage.median()[0]:.3f}, std={df_stage.std()[0]:.3f}')
    print(f'cur : mean={df_cur.mean()[0]:.3f}, median={df_cur.median()[0]:.3f}, std={df_cur.std()[0]:.3f}')
    print(f'diff : mean={df_diff.mean()[0]:.3f}, median={df_diff.median()[0]:.3f}, std={df_diff.std()[0]:.3f}')
    print()
    
    df_stage = pd.DataFrame(ssim_stage)
    df_cur = pd.DataFrame(ssim_cur_val)
    df_diff = pd.DataFrame(ssim_diff_val)
    
    print("SSIM")
    print(f'stage : mean={df_stage.mean()[0]:.3f}, median={df_stage.median()[0]:.3f}, std={df_stage.std()[0]:.3f}')
    print(f'cur : mean={df_cur.mean()[0]:.3f}, median={df_cur.median()[0]:.3f}, std={df_cur.std()[0]:.3f}')
    print(f'diff : mean={df_diff.mean()[0]:.3f}, median={df_diff.median()[0]:.3f}, std={df_diff.std()[0]:.3f}')

def getMean_Max_PSNR_SSIM(all_df_dict):
    max_psnr, max_ssim = [], []
    for dfName, df in all_df_dict.items():
        max_psnr.append(df.loc[df['psnr'].idxmax()]['psnr'])
        max_ssim.append(df.loc[df['ssim'].idxmax()]['ssim'])
        
    max_psnr = pd.DataFrame(max_psnr).mean()[0]
    max_ssim = pd.DataFrame(max_ssim).mean()[0]
    
    print(f'max_psnr={max_psnr:.3f}, max_ssim={max_ssim:.3f}')    
    

def getMax_val(all_df_dict, label, val):
    idx_stage, idx_psnr, idx_ssim = [], [], []
    for dfName, df in all_df_dict.items():
        gt_beta = float(dfName.split('_')[-1])
        
        idx = df.index[(df[label] <= val)].tolist()[0]
        
        idx_stage.append([df.loc[idx]['stage']*df.loc[idx]['step_beta'] - gt_beta])
        idx_psnr.append(df.loc[idx]['psnr'])
        idx_ssim.append(df.loc[idx]['ssim'])
    
    
    idx_stage = pd.DataFrame(idx_stage)
    idx_psnr = pd.DataFrame(idx_psnr)
    idx_ssim = pd.DataFrame(idx_ssim)
    
    print(f'{val} : psnr_mean={idx_psnr.mean()[0]:.3f}({idx_psnr.std()[0]:.3f}), ssim_mean={idx_ssim.mean()[0]:.3f}({idx_ssim.std()[0]:.3f}), beta_err={idx_stage.mean()[0]:.3f}')
    

if __name__ == '__main__':
    # dataRoot = 'output_log'
    dataRoot = 'D:/data/RESIDE_beta_sample_100'
    # for beta in [[0.04], [0.06], [0.08], [0.1], [0.12], [0.16], [0.2]]:
    #     all_df_dict = read_csv_all(dataRoot, 'Entropy_Module', beta)
    #     getMaxPSNR_SSIM(all_df_dict)
    #     getMean_Max_PSNR_SSIM(all_df_dict)
    
    
    beta = [0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2]
    #beta = [0.1]
    all_df_dict = read_csv_all(dataRoot, 'Entropy_Module', beta)
    for dfName, df in all_df_dict.items():
        data_plot(dfName, df, labels=['cur_val', 'psnr'])
    
    
    
    # for beta in [[0.04], [0.06], [0.08], [0.1], [0.12], [0.16], [0.2]]:
    #     all_df_dict = read_csv_all(dataRoot, 'Entropy_Module', beta)
    #     getMax_val(all_df_dict, 'diff_val', 0.0)
        
    # beta = [0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2]    
    # all_df_dict = read_csv_all(dataRoot, 'Entropy_Module', beta)
    # getMax_val(all_df_dict, 'diff_val', -0.01)
    # getMax_val(all_df_dict, 'diff_val', 0.0)
    # getMax_val(all_df_dict, 'diff_val', 0.01)
    # getMax_val(all_df_dict, 'diff_val', 0.005)
    
    # getMax_val(all_df_dict, 'cur_val', 7.4)    
    # getMax_val(all_df_dict, 'cur_val', 7.5)
    # getMax_val(all_df_dict, 'cur_val', 7.6)
    
    # for beta in [[0.04], [0.06], [0.08], [0.1], [0.12], [0.16], [0.2]]:
    #     all_df_dict = read_csv_all(dataRoot, 'Entropy_Module', beta)
    #     getMax_val(all_df_dict, 'cur_val', 7.5) 
    
    