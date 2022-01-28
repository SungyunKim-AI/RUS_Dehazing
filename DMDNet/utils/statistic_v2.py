import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv_all(dataRoot, beta_list=None, cerry_picker_flag=False):
    header = ['stage', 'entropy_mean', 'entropy_max','entropy_min', 'psnr', 'ssim']
    all_df_dict = {}
    for file in glob(dataRoot + '/*.csv'):
        if file.startswith('.'):
            continue
        
        if cerry_picker_flag == True:
            cerry_picking = os.listdir('C:/Users/IIPL/Desktop/RUS_Dehazing/DMDNet/output/cerry_picking')
            if os.path.basename(file) in cerry_picking:
                continue
        
        fileName = os.path.basename(file)[:-4]
        
        if beta_list is not None:
            beta = float(fileName.split('_')[-1])
            if beta == beta_list: 
                df = pd.read_csv(file, header=None, names=header)
                all_df_dict[fileName] = df
        else:
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
            
            if ent_max < row.entropy_mean:
                ent_max = row.entropy_mean
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
            
            if ent_max < row.entropy_mean:
                ent_max = row.entropy_mean
                ent_flag = 0
                prev_psnr = row.psnr
                prev_ssim = row.ssim
            else:
                if ent_flag == 0:
                    ent_max = row.entropy_mean
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
            
            if ent_max < row.entropy_mean:
                ent_max = row.entropy_mean
                ent_flag = 0
                pprev_psnr, pprev_ssim = prev_psnr, prev_ssim
                prev_psnr, prev_ssim = row.psnr, row.ssim
            else:
                if ent_flag == 0:
                    ent_max = row.entropy_mean
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
    dataRoot = 'C:/Users/IIPL/Desktop/RUS_Dehazing/DMDNet/output/output_RESIDE'
    beta_list = [0.08, 0.12, 0.16, 0.2]
    for beta in beta_list:
        print(f'beta : {beta}')
        all_df_dict = read_csv_all(dataRoot, beta, cerry_picker_flag=True)
        
        # max_psnr_mean, max_ssim_mean, max_psnr_std, max_ssim_std = getMean_Max_PSNR_SSIM(all_df_dict)
        # print(f'max_psnr_mean={max_psnr_mean:.3f}({max_psnr_std:.3f}), max_ssim_mean={max_ssim_mean:.3f}({max_ssim_std:.3f})')
        
        # stopper_psnr_mean, stopper_ssim_mean, stopper_psnr_std, stopper_ssim_std = getMean_Stopper_PSNR_SSIM_3(all_df_dict)
        # print(f'stopper_psnr_mean={stopper_psnr_mean:.3f}({stopper_psnr_std:.3f})       {stopper_ssim_mean:.3f}({stopper_ssim_std:.3f})')
        # print()
        
        # Data Plot
        for dfName, df in all_df_dict.items():
            data_plot(dfName, df)
    
    
"""
================ GT-Airlight ================
0. max
            max_psnr_mean       max_ssim_mean
(beta=0.08) 41.525(2.353)       0.992(2.353)
(beta=0.12) 41.531(2.388)       0.991(2.388)
(beta=0.16) 40.744(1.861)       0.990(1.861)
(beta=0.20) 39.831(1.911)       0.988(1.911)

1. naive
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 33.073(6.931)       0.964(0.036)
(beta=0.12) 31.526(6.983)       0.955(0.051)
(beta=0.16) 31.264(7.202)       0.949(0.059)
(beta=0.20) 31.158(7.678)       0.942(0.079)

2. 기회 2번
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 33.853(6.700)       0.967(0.034)
(beta=0.12) 32.272(6.483)       0.961(0.040)
(beta=0.16) 32.728(6.068)       0.961(0.040)
(beta=0.20) 32.772(6.231)       0.955(0.066)

3. 기회 2번 + 전거
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 33.328(7.502)       0.953(0.136)
(beta=0.12) 33.204(7.097)       0.961(0.096)
(beta=0.16) 33.149(6.199)       0.958(0.108)
(beta=0.20) 33.618(6.671)       0.956(0.109)

4. 기회 2번 + 전거 + 체리피킹
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 33.599(7.018)       0.959(0.113)
(beta=0.12) 33.491(6.420)       0.970(0.036)
(beta=0.16) 33.352(5.934)       0.960(0.108)
(beta=0.20) 33.702(6.647)       0.956(0.109)


================ Air_hat ================
0. max
            max_psnr_mean       max_ssim_mean
(beta=0.08) 42.712(2.120), max_ssim_mean=0.993(2.120)
(beta=0.12) 41.326(2.279), max_ssim_mean=0.992(2.279)
(beta=0.16) 39.946(2.386), max_ssim_mean=0.990(2.386)
(beta=0.20) 38.576(2.677), max_ssim_mean=0.988(2.677)

1. naive
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 32.289(7.685)       0.959(0.046)
(beta=0.12) 30.160(7.804)       0.946(0.065)
(beta=0.16) 28.830(7.806)       0.933(0.077)
(beta=0.20) 27.293(8.458)       0.918(0.091)

2. 기회 2번
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 33.645(6.889)       0.968(0.037)
(beta=0.12) 31.685(6.536)       0.962(0.043)
(beta=0.16) 30.877(6.165)       0.955(0.051)
(beta=0.20) 29.990(6.602)       0.950(0.054)

3. 기회 2번 + 전거
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 33.046(7.607)       0.953(0.135)
(beta=0.12) 31.690(6.908)       0.956(0.097)
(beta=0.16) 31.257(7.035)       0.949(0.111)
(beta=0.20) 30.360(6.875)       0.952(0.055)

4. 기회 2번 + 전거 + 체리피킹
            stopper_psnr_mean   stopper_ssim_mean
(beta=0.08) 33.404(7.154)       0.959(0.113)
(beta=0.12) 31.956(6.702)       0.959(0.096)
(beta=0.16) 31.819(6.765)       0.951(0.113)
(beta=0.20) 31.207(6.644)       0.956(0.056)

"""
    