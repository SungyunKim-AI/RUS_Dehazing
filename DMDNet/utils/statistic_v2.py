import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv_all(dataRoot, target_beta=None, cerry_picker_flag=False):

    header = ['stage', 'entropy', 'psnr', 'ssim']
    all_df_dict = {}
    for folder in glob(dataRoot + '/*'):
        beta = float(folder.split('_')[-1])
        if beta != target_beta:
            continue
            
        for file in glob(folder + '/*.csv'):
            if cerry_picker_flag == True:
                cerry_picking = os.listdir('C:/Users/IIPL/Desktop/RUS_Dehazing/DMDNet/output/cerry_picking')
                if os.path.basename(file) in cerry_picking:
                    continue
            
            fileName = os.path.basename(file)[:-4]
            df = pd.read_csv(file, header=None, names=header)
            all_df_dict[fileName] = df
        
    return all_df_dict


def data_plot(dfName, df):
    fig, ax1 = plt.subplots() 
    ax1.set_title(dfName, fontsize=16)
    ax1.set_xticks(df['stage'])
    
    color_1 = 'tab:blue' 
    ax1.set_ylabel('PSNR (blue)', color=color_1) 
    ax1.plot(df['psnr'], marker='s', color=color_1) 
    ax1.tick_params(axis='y', labelcolor=color_1) 
    
    ax2 = ax1.twinx() 
    color_2 = 'tab:red' 
    ax2.set_ylabel('Entropy (red)', color=color_2) 
    ax2.plot(df['entropy_mean'], marker='.', color=color_2) 
    ax2.tick_params(axis='y', labelcolor=color_2)
    plt.show()
    

# PSNR, SSIM 최대값들의 평균
def getMean_Max_PSNR_SSIM(all_df_dict):
    max_psnr_list, max_ssim_list = [], []
    for _, df in all_df_dict.items():
        max_psnr_list.append(df.loc[df['psnr'].idxmax()]['psnr'])
        max_ssim_list.append(df.loc[df['ssim'].idxmax()]['ssim'])
        
    max_psnr_mean = pd.DataFrame(max_psnr_list).mean()[0]
    max_psnr_std = pd.DataFrame(max_psnr_list).std()[0]
    max_ssim_mean = pd.DataFrame(max_ssim_list).mean()[0]
    max_ssim_std = pd.DataFrame(max_psnr_list).std()[0]
    
    return max_psnr_mean, max_ssim_mean, max_psnr_std, max_ssim_std

# Entrpoy 극대일 때 PSNR, SSIM 평균
"""
Stopper 종료 조건
 - 후보 1: 직전보다 작아지는 시점에서 종료
 - 후보 2: 기회 2번 주기 (2번 이내에 커지면 커지는걸로 인정)
 - 후보 2: 후보 2번 + 1단계 전에 것 (2번 이내에 커지면 커지는걸로 인정)
"""
def getMean_Stopper_PSNR_SSIM_1(all_df_dict):
    stopper_psnr_list, stopper_ssim_list = [], []
    for _, df in all_df_dict.items():
        ent_max = 0
        for row in df.itertuples():
            if row.stage < 3:
                continue
            
            if ent_max < row.entropy:
                ent_max = row.entropy
                prev_psnr = row.psnr
                prev_ssim = row.ssim
            else:
                stopper_psnr_list.append(prev_psnr)
                stopper_ssim_list.append(prev_ssim)
                break
    
    stopper_psnr_mean = np.array(stopper_psnr_list).mean()
    stopper_psnr_std = np.array(stopper_psnr_list).std()
    stopper_ssim_mean = np.array(stopper_ssim_list).mean()
    stopper_ssim_std = np.array(stopper_ssim_list).std()
    
    return stopper_psnr_mean, stopper_ssim_mean, stopper_psnr_std, stopper_ssim_std
      
def getMean_Stopper_PSNR_SSIM_2(all_df_dict):
    stopper_psnr_list, stopper_ssim_list = [], []
    for _, df in all_df_dict.items():
        ent_max, ent_flag = 0, 0
        prev_psnr, prev_ssim = 0, 0
        for row in df.itertuples():
            if row.stage < 3:
                continue
            
            if ent_max < row.entropy:
                ent_max = row.entropy
                ent_flag = 0
                prev_psnr = row.psnr
                prev_ssim = row.ssim
            else:
                if ent_flag == 0:
                    ent_max = row.entropy
                    ent_flag += 1
                elif ent_flag == 1:
                    stopper_psnr_list.append(prev_psnr)
                    stopper_ssim_list.append(prev_ssim)
                    break
    
    stopper_psnr_mean = np.array(stopper_psnr_list).mean()
    stopper_psnr_std = np.array(stopper_psnr_list).std()
    stopper_ssim_mean = np.array(stopper_ssim_list).mean()
    stopper_ssim_std = np.array(stopper_ssim_list).std()
    
    return stopper_psnr_mean, stopper_ssim_mean, stopper_psnr_std, stopper_ssim_std

def getMean_Stopper_PSNR_SSIM_3(all_df_dict):
    stopper_psnr_list, stopper_ssim_list = [], []
    for _, df in all_df_dict.items():
        ent_max, ent_flag = 0, 0
        prev_psnr, prev_ssim = 0, 0
        for row in df.itertuples():
            if row.stage < 3:
                continue
            
            if ent_max < row.entropy:
                ent_max = row.entropy
                ent_flag = 0
                pprev_psnr, pprev_ssim = prev_psnr, prev_ssim
                prev_psnr, prev_ssim = row.psnr, row.ssim
            else:
                if ent_flag == 0:
                    ent_max = row.entropy
                    ent_flag += 1
                elif ent_flag == 1:
                    stopper_psnr_list.append(pprev_psnr)
                    stopper_ssim_list.append(pprev_ssim)
                    break
    
    stopper_psnr_mean = np.array(stopper_psnr_list).mean()
    stopper_psnr_std = np.array(stopper_psnr_list).std()
    stopper_ssim_mean = np.array(stopper_ssim_list).mean()
    stopper_ssim_std = np.array(stopper_ssim_list).std()
    
    return stopper_psnr_mean, stopper_ssim_mean, stopper_psnr_std, stopper_ssim_std

def getMean_Stopper_PSNR_SSIM_4(all_df_dict, limit=20):
    stopper_psnr_list, stopper_ssim_list = [], []
    max_psnr, max_name = 0, ''
    for df_name, df in all_df_dict.items():
        last_idx = df.iloc[-1]['stage']
        ent_max, ent_flag = 0, 0
        stopper_psnr, stopper_ssim = 0, 0
        for row in df.itertuples():
            if ent_max < row.entropy:
                ent_max, ent_flag = row.entropy, 0
                stopper_psnr, stopper_ssim = row.psnr, row.ssim
            else:
                if row.stage < last_idx:
                    if ent_flag < (limit-1):
                        ent_flag += 1
                    else:
                        stopper_psnr_list.append(stopper_psnr)
                        stopper_ssim_list.append(stopper_ssim)
                        if max_psnr < stopper_psnr:
                            max_psnr = stopper_psnr
                            max_name = df_name
                        break
                else:
                    stopper_psnr_list.append(stopper_psnr)
                    stopper_ssim_list.append(stopper_ssim)
                    if max_psnr < stopper_psnr:
                        max_psnr = stopper_psnr
                        max_name = df_name
    
    stopper_psnr_mean = np.array(stopper_psnr_list).mean()
    stopper_psnr_std = np.array(stopper_psnr_list).std()
    stopper_ssim_mean = np.array(stopper_ssim_list).mean()
    stopper_ssim_std = np.array(stopper_ssim_list).std()
    
    return stopper_psnr_mean, stopper_ssim_mean, stopper_psnr_std, stopper_ssim_std, max_psnr, max_name

if __name__ == '__main__':
    # dataRoot = 'D:/data/output_dehaze/output_RESIDE'
    dataRoot = 'C:/Users/pc/Documents/GitHub/RUS_Dehazing/DMDNet/output/output_RESIDE'
    beta_list = [0.08, 0.12, 0.16, 0.2]
    total_mean_psnr, total_mean_ssim = 0, 0
    for beta in beta_list:
        print(f'beta : {beta}')
        all_df_dict = read_csv_all(dataRoot, beta, cerry_picker_flag=False)
        # for dfName, df in all_df_dict.items():
        #     print(df)
        #     exit()
        
        # max_psnr_mean, max_ssim_mean, max_psnr_std, max_ssim_std = getMean_Max_PSNR_SSIM(all_df_dict)
        # print(f'max_psnr_mean={max_psnr_mean:.3f}({max_psnr_std:.3f})         {max_ssim_mean:.3f}({max_ssim_std:.3f})')
        
        stopper_psnr_mean, stopper_ssim_mean, stopper_psnr_std, stopper_ssim_std, max_psnr, max_name = getMean_Stopper_PSNR_SSIM_4(all_df_dict)
        print(f'stopper_psnr_mean={stopper_psnr_mean:.3f}({stopper_psnr_std:.3f})       {stopper_ssim_mean:.3f}({stopper_ssim_std:.4f})')
        print(max_psnr, max_name)
        total_mean_psnr += stopper_psnr_mean
        total_mean_ssim += stopper_ssim_mean
        print()
        
        # Data Plot
        # for dfName, df in all_df_dict.items():
        #     data_plot(dfName, df)
    
    total_mean_psnr /= 4
    total_mean_ssim /= 4
    print(f"Total : {total_mean_psnr:.3f} {total_mean_ssim:.4f}")
    
    
"""
0. max
            max_psnr_mean       max_ssim_mean
(beta=0.08) 44.047(3.857)       0.996(3.857)
(beta=0.12) 43.143(3.998)       0.996(3.998)
(beta=0.16) 41.630(3.909)       0.995(3.909)
(beta=0.20) 39.985(4.742)       0.993(4.742)


1. naive
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 34.238(8.840)       0.965(0.047)
(beta=0.12) 30.973(8.782)       0.949(0.062)
(beta=0.16) 30.340(8.611)       0.942(0.078)
(beta=0.20) 28.328(9.252)       0.926(0.096)


2. 기회 2번
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 35.197(8.298)       0.971(0.037)
(beta=0.12) 33.205(8.093)       0.964(0.048)
(beta=0.16) 32.786(7.112)       0.963(0.050)
(beta=0.20) 30.928(7.587)       0.955(0.061)
(Total)     33.029              0.963


3. 기회 2번 + 전거
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 34.305(8.022)       0.962(0.113)
(beta=0.12) 33.117(8.564)       0.955(0.129)
(beta=0.16) 33.875(7.386)       0.967(0.053)
(beta=0.20) 31.208(7.527)       0.958(0.063)
(Total)     33.126              0.961

"""
    