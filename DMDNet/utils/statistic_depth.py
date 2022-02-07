import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv_all(dataRoot, target_beta=None):
    header = ['stage', 'abs_rel', 'sq_rel','rmse', 'rmse_log', 'a1', 'a2', 'a3', 'entropy']
    all_df_dict = {}

    for file in glob(dataRoot + '/*.csv'):
        fileName = os.path.basename(file)[:-4]
        if fileName[0] == '_':
            continue
        
        if float(fileName.split('_')[-1]) != target_beta:
            continue
        
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


def get_Varitation(dataRoot, all_df_dict, labels, gt_beta=None, step_beta=0.005):
    if not isinstance(labels, list):
        labels = list(labels)
    
    haze_err_dict = {'abs_rel':[], 'sq_rel':[], 'rmse':[], 'rmse_log':[], 'a1':[], 'a2':[], 'a3':[]}
    dehaze_err_dict = {'abs_rel':[], 'sq_rel':[], 'rmse':[], 'rmse_log':[], 'a1':[], 'a2':[], 'a3':[]}
    improve_err_dict = {'name':[], 'abs_rel':[], 'sq_rel':[], 'rmse':[], 'rmse_log':[], 'a1':[], 'a2':[], 'a3':[]}
    beta_err_list, cnt = [], 0
    for df_name, df in all_df_dict.items():
        improve_err_dict['name'].append(df_name)
        est_step = stopper(df)
        if gt_beta is not None:
            beta_err = abs(gt_beta - (est_step*step_beta))
            beta_err_list.append(beta_err)

        if est_step == 0:
            cnt += 1
            # print(df_name)
        
        for label in labels:
            haze_err_dict[label].append(df.loc[0, label])
            dehaze_err_dict[label].append(df.loc[est_step, label])
            
            percent = abs(df.loc[0, label] - df.loc[est_step, label]) / df.loc[0, label] * 100
            improve_err_dict[label].append(percent)
    
    improve_best_top(improve_err_dict, dataRoot, gt_beta)
    beta_err_mean = np.array(beta_err_list).mean()
    return haze_err_dict, dehaze_err_dict, beta_err_mean, cnt

def stopper(df, limit=20):
    last_idx = df.iloc[-1]['stage']
    ent_max, ent_flag = 0, 0
    stop_index = 0
    for row in df.itertuples():
        if ent_max < row.entropy:
            ent_max, ent_flag = row.entropy, 0
            stop_index = row.stage
        else:
            if row.stage < last_idx:
                if ent_flag < (limit-1):
                    ent_flag += 1
                else:
                    return stop_index
            else:
                return stop_index
    
    return 0

def improve_best_top(improve_err_dict, save_path, gt_beta, target_label='a1', top=50):
    df = pd.DataFrame(improve_err_dict)
    topDF = df.nlargest(50, target_label)
    topDF.to_csv(f'{save_path}/_{gt_beta}_{target_label}_improve.csv', sep=',',na_rep='NaN', index=False)
    
    

if __name__ == '__main__':
    dataRoot = 'D:/data/output_depth/Monodepth_depth_KITTI/_statistics'
    # dataRoot = 'D:/data/output_depth/DenseDenpth_depth_KITTI/_statistics'
    # dataRoot = 'D:/data/output_depth/DPT_depth_KITTI/_statistics'
    
    labels = ['abs_rel', 'sq_rel','rmse', 'rmse_log', 'a1', 'a2', 'a3']
    beta_list = [0.02, 0.04, 0.06]
    
    for gt_beta in beta_list:
        print(f'beta : {gt_beta}')
        all_df_dict = read_csv_all(dataRoot, gt_beta)
        
        haze_err_dict, dehaze_err_dict, beta_err_mean, cnt = get_Varitation(dataRoot, all_df_dict, labels, gt_beta)
        
        for label in labels:
            before_err = np.array(haze_err_dict[label]).mean()
            after_err = np.array(dehaze_err_dict[label]).mean()
            percent = abs(before_err - after_err) / before_err * 100
            print(f"{label} : {before_err:.3f} -> {after_err:.3f} ({percent:.1f})")
        
        print(f"beta err : {beta_err_mean:.3f}")
        print(f"not found stopper : {cnt}")
        print()
        
        # Data Plot
        # for dfName, df in all_df_dict.items():
        #     data_plot(dfName, df)



"""
============ output_DenseDenpth_depth_KITTI ============
beta : 0.02
abs_rel : 0.09627 -> 0.07664 (20.39)
sq_rel : 0.70126 -> 0.48461 (30.89)  
rmse : 4.40986 -> 3.59545 (18.47)    
rmse_log : 0.12472 -> 0.10155 (18.58)
a1 : 0.89985 -> 0.93533 (3.94)       
a2 : 0.98712 -> 0.99222 (0.52)       
a3 : 0.99827 -> 0.99859 (0.03)       
beta err : 0.01361    
not found stopper : 308

beta : 0.04
abs_rel : 0.16443 -> 0.09867 (39.99)
sq_rel : 1.83232 -> 0.78449 (57.19)  
rmse : 6.91377 -> 4.50341 (34.86)    
rmse_log : 0.20275 -> 0.12963 (36.06)
a1 : 0.76966 -> 0.89214 (15.91)
a2 : 0.94123 -> 0.98234 (4.37)
a3 : 0.98755 -> 0.99653 (0.91)
beta err : 0.01370
not found stopper : 28

beta : 0.06
abs_rel : 0.23752 -> 0.11971 (49.60)
sq_rel : 3.49824 -> 1.13785 (67.47)
rmse : 9.09958 -> 5.37325 (40.95)
rmse_log : 0.27735 -> 0.15497 (44.12)
a1 : 0.65327 -> 0.84914 (29.98)
a2 : 0.87569 -> 0.96870 (10.62)
a3 : 0.96206 -> 0.99370 (3.29)
beta err : 0.01110
not found stopper : 9

============ output_DPT_depth_KITTI ============
beta : 0.02
abs_rel : 0.06109 -> 0.05563 (8.95)
sq_rel : 0.30824 -> 0.26405 (14.34)
rmse : 2.79777 -> 2.56540 (8.31)
rmse_log : 0.09333 -> 0.08601 (7.85)
a1 : 0.95805 -> 0.96599 (0.83)
a2 : 0.99308 -> 0.99423 (0.12)
a3 : 0.99843 -> 0.99865 (0.02)
beta err : 0.01584
not found stopper : 427

beta : 0.04
abs_rel : 0.09261 -> 0.07322 (20.94)
sq_rel : 0.74603 -> 0.48722 (34.69)
rmse : 4.46308 -> 3.47573 (22.12)
rmse_log : 0.13789 -> 0.11082 (19.63)
a1 : 0.90643 -> 0.93987 (3.69)
a2 : 0.97641 -> 0.98665 (1.05)
a3 : 0.99351 -> 0.99667 (0.32)
beta err : 0.01870
not found stopper : 59

beta : 0.06
abs_rel : 0.12154 -> 0.09290 (23.56)
sq_rel : 1.32448 -> 0.81042 (38.81)
rmse : 5.98856 -> 4.42252 (26.15)
rmse_log : 0.17511 -> 0.13617 (22.24)
a1 : 0.86140 -> 0.90931 (5.56)
a2 : 0.95526 -> 0.97505 (2.07)
a3 : 0.98590 -> 0.99301 (0.72)
beta err : 0.01861
not found stopper : 23


============ output_Monodepth_depth_KITTI ============
beta : 0.02
abs_rel : 0.14518 -> 0.12433 (14.36)
sq_rel : 4.44249 -> 3.70331 (16.64)
rmse : 11.51670 -> 10.50621 (8.77)
rmse_log : 0.20133 -> 0.17937 (10.91)
a1 : 0.85686 -> 0.88355 (3.11)
a2 : 0.94946 -> 0.96153 (1.27)
a3 : 0.97614 -> 0.98210 (0.61)
beta err : 0.01558
not found stopper : 406

beta : 0.04
abs_rel : 0.26345 -> 0.17452 (33.76)
sq_rel : 9.38792 -> 5.58815 (40.48)
rmse : 15.20385 -> 12.36625 (18.66)
rmse_log : 0.31382 -> 0.23189 (26.11)
a1 : 0.72209 -> 0.82659 (14.47)
a2 : 0.88413 -> 0.93543 (5.80)
a3 : 0.94056 -> 0.96736 (2.85)
beta err : 0.01682
not found stopper : 53

beta : 0.06
abs_rel : 0.40582 -> 0.21128 (47.94)
sq_rel : 17.30151 -> 6.96641 (59.74)
rmse : 17.46637 -> 13.33381 (23.66)
rmse_log : 0.41792 -> 0.26628 (36.28)
a1 : 0.59849 -> 0.78215 (30.69)
a2 : 0.81060 -> 0.91392 (12.75)
a3 : 0.89923 -> 0.95607 (6.32)
beta err : 0.01545
not found stopper : 12
"""
    