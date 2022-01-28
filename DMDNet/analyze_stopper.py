import csv
from glob import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

csv_files = glob('output_RESIDE/*.csv')

for csv_file in tqdm(csv_files):

    data = np.genfromtxt(csv_file, delimiter=',')
    name = os.path.basename(csv_file)
    #print(data)
    plt.figure(figsize=(12,4))
    ax0, ax1, ax2, ax3, ax4= plt.gca(), plt.gca().twinx(), plt.gca().twinx(), plt.gca().twinx(), plt.gca().twinx()
    
    
    #ax0.set_ylabel('mean_entropy', color='tab:blue')
    #ax0.axes.yaxis.set_visible(False)
    c0 = ax0.plot(data[:,0], data[:,1], marker='.', color='tab:blue')
    ax0.tick_params(axis='y', labelcolor='tab:blue')
    
    #ax1.set_ylabel('max_entropy', color='tab:red')
    ax1.axes.yaxis.set_visible(False)
    c1 = ax1.plot(data[:,0], data[:,2], marker='.', color='tab:red')
    #ax1.tick_params(axis='y', labelcolor='tab:red')
    
    #ax2.set_ylabel('min_entropy', color='tab:green')
    ax2.axes.yaxis.set_visible(False)
    c2 = ax2.plot(data[:,0], data[:,3], marker='.', color='tab:green')
    #ax2.tick_params(axis='y', labelcolor='tab:green')
    
    #ax3.set_ylabel('psnr', color='tab:pink')
    #ax3.axes.yaxis.set_visible(False)
    c3 = ax3.plot(data[:,0], data[:,4], marker='.', color='tab:pink')
    ax3.tick_params(axis='y', labelcolor='tab:pink', 
                    left=False, labelleft=False,
                    right=True, labelright=True)
    # ax3.spines['left'].set_position('right',0.1)
    
    #ax4.set_ylabel('ssim', color='tab:orange')
    ax4.axes.yaxis.set_visible(False)
    c4 = ax4.plot(data[:,0], data[:,5], marker='.', color='tab:orange')
    #ax4.tick_params(axis='y', labelcolor='tab:orange')
    
    c = c0+c1+c2+c3+c4
    ax0.legend(c,['mean_entropy','max_entropy','min_entropy','psnr','ssim'])
    
    #plt.show()
    plt.savefig(f'output_RESIDE_fig/{name[:-4]}.png')