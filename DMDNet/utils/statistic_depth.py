import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv_all(dataRoot, beta_list=None):
    header = ['stage', 'abs_rel', 'sq_rel','rmse', 'rmse_log', 'a1', 'a2', 'a3']
    all_df_dict = {}
    for file in glob(dataRoot + '/*.csv'):
        if file.startswith('.'):
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
    ax1.set_ylabel('abs_rel (blue)', color=color_1) 
    ax1.plot(df['abs_rel'], marker='s', color=color_1) 
    ax1.tick_params(axis='y', labelcolor=color_1) 
    
    ax2 = ax1.twinx() 
    color_2 = 'tab:red' 
    ax2.set_ylabel('a1 (red)', color=color_2) 
    ax2.plot(df['a1'], marker='.', color=color_2) 
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

# 특정 labels 에서 depth score 가 좋아질 확률
def get_Varitation(all_df_dict, gt_beta, labels, step_beta=0.01):
    if not isinstance(labels, list):
        labels = list(labels)
    
    var_dict = {}
    gt_step = (gt_beta / step_beta) + 1
    for label in labels:
        improve_cnt, decade_cnt, improve_list, decade_list = 0, 0, [], []
        
        for _, df in all_df_dict.items():
            if label in ['abs_rel', 'sq_rel','rmse', 'rmse_log']:
                var = df.loc[0, label] - df.loc[gt_step, label]
            elif label in ['a1', 'a2', 'a3']:
                var =  df.loc[gt_step, label] - df.loc[0, label]
            
            if var > 0:
                improve_cnt += 1
                improve_list.append(var)
            else:
                decade_cnt += 1
                decade_list.append(var)
        
        percent = improve_cnt / (improve_cnt + decade_cnt) * 100
        improve_mean = np.array(improve_list).mean()
        decade_mean = np.array(decade_list).mean()
        
        var_dict[label] = {'improve_mean':improve_mean, 'decade_mean':decade_mean, 'percent':percent}
    
    return var_dict
   

if __name__ == '__main__':
    # header = ['stage', 'abs_rel', 'sq_rel','rmse', 'rmse_log', 'a1', 'a2', 'a3']
    # dataRoot = 'C:/Users/IIPL/Desktop/RUS_Dehazing/DMDNet/output/output_DPT_depth_NYU'
    
    dataRoot = 'C:/Users/IIPL/Desktop/RUS_Dehazing/DMDNet/output/output_Monodepth_depth_KITTI'
    beta_list = [0.1, 0.2, 0.3]
    
    # dataRoot = 'C:/Users/IIPL/Desktop/RUS_Dehazing/DMDNet/output/output_DPT_depth_NYU'
    # beta_list = [0.1, 0.3, 0.5, 0.7]
    
    # dataRoot = 'C:/Users/IIPL/Desktop/RUS_Dehazing/DMDNet/output/output_DPT_depth_KITTI'
    # beta_list = [0.1, 0.2, 0.3]
    
    for beta in beta_list:
        print(f'beta : {beta}')
        all_df_dict = read_csv_all(dataRoot, beta)
        
        # var_dict = get_Varitation(all_df_dict, beta, labels=['abs_rel', 'sq_rel','rmse', 'rmse_log', 'a1', 'a2', 'a3'])
        # for key, value in var_dict.items():
        #     print(key)
        #     percent, improve_mean, decade_mean = var_dict[key]['percent'], var_dict[key]['improve_mean'], var_dict[key]['decade_mean']
        #     print(f'depth 개선 비율 : {percent:.2f}%')
        #     print(f'개선 정도 : {improve_mean:.5f}')
        #     print(f'퇴보 정도 : {decade_mean:.5f}')
        # print()
        
        # Data Plot
        for dfName, df in all_df_dict.items():
            data_plot(dfName, df)
    