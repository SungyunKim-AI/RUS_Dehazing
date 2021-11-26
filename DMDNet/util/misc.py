import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dpt.vit import get_mean_attention_map

def visualize_attention(input, model, prediction, model_type):
    input = (input + 1.0)/2.0

    attn1 = model.pretrained.attention["attn_1"]
    attn2 = model.pretrained.attention["attn_2"]
    attn3 = model.pretrained.attention["attn_3"]
    attn4 = model.pretrained.attention["attn_4"]

    plt.subplot(3,4,1), plt.imshow(input.squeeze().permute(1,2,0)), plt.title("Input", fontsize=8), plt.axis("off")
    plt.subplot(3,4,2), plt.imshow(prediction), plt.set_cmap("inferno"), plt.title("Prediction", fontsize=8), plt.axis("off")

    if model_type == "dpt_hybrid":
        h = [3,6,9,12]
    else:
        h = [6,12,18,24]

    # upper left
    plt.subplot(345),
    ax1 = plt.imshow(get_mean_attention_map(attn1, 1, input.shape))
    plt.ylabel("Upper left corner", fontsize=8)
    plt.title(f"Layer {h[0]}", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])


    plt.subplot(346),
    plt.imshow(get_mean_attention_map(attn2, 1, input.shape))
    plt.title(f"Layer {h[1]}", fontsize=8)
    plt.axis("off"),

    plt.subplot(347),
    plt.imshow(get_mean_attention_map(attn3, 1, input.shape))
    plt.title(f"Layer {h[2]}", fontsize=8)
    plt.axis("off"),


    plt.subplot(348),
    plt.imshow(get_mean_attention_map(attn4, 1, input.shape))
    plt.title(f"Layer {h[3]}", fontsize=8)
    plt.axis("off"),


    # lower right
    plt.subplot(3,4,9), plt.imshow(get_mean_attention_map(attn1, -1, input.shape))
    plt.ylabel("Lower right corner", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])

    plt.subplot(3,4,10), plt.imshow(get_mean_attention_map(attn2, -1, input.shape)), plt.axis("off")
    plt.subplot(3,4,11), plt.imshow(get_mean_attention_map(attn3, -1, input.shape)), plt.axis("off")
    plt.subplot(3,4,12), plt.imshow(get_mean_attention_map(attn4, -1, input.shape)), plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_plt(x, y, index):
    plt.subplot(index)
    plt.bar(x, y, align='center')
    plt.xlabel('deg.')
    plt.xlim([x[0], x[-1]])
    plt.ylim(0,np.max(y))
    plt.ylabel('prob.')
    for i in range(len(y)):
        plt.vlines(x[i], 0, y[i])

def show_histogram(img,index):
    
    val, cnt = np.unique(img, return_counts=True)
    img_size = img.shape[0] * img.shape[1]
    #prob = cnt / img_size    # PMF
    prob = cnt
    
    l = np.zeros((np.max(val)+1))
    for i, v in enumerate(val):
        l[v] = prob[i]
    show_plt(range((np.max(val)+1)), l,index)
    
    
def multi_show(image_list):
    """
    First_hazy       Final_prediction   GT
    First_depth      Final_depth        depth_GT
    First_airlight   Final_airlight     airlight_GT
    one_shot_prediction
    """
    init_hazy, prediction, clear, init_depth, depth, clear_depth, init_airlight, airlight, clear_airlight, one_shot_prediction = image_list
    
    
    init_hazy = cv2.cvtColor(init_hazy.astype(np.float32), cv2.COLOR_RGB2BGR)
    prediction = cv2.cvtColor(prediction.astype(np.float32), cv2.COLOR_RGB2BGR)
    clear = cv2.cvtColor(clear.astype(np.float32), cv2.COLOR_RGB2BGR)
    init_airlight = cv2.cvtColor(init_airlight.astype(np.float32), cv2.COLOR_RGB2BGR)
    airlight = cv2.cvtColor(airlight.astype(np.float32), cv2.COLOR_RGB2BGR)
    clear_airlight = cv2.cvtColor(clear_airlight.astype(np.float32), cv2.COLOR_RGB2BGR)
    one_shot_prediction = cv2.cvtColor(one_shot_prediction.astype(np.float32), cv2.COLOR_RGB2BGR)
    
    init_depth = cv2.cvtColor(init_depth/10, cv2.COLOR_GRAY2BGR)
    depth = cv2.cvtColor(depth/10, cv2.COLOR_GRAY2BGR)
    clear_depth = cv2.cvtColor(clear_depth/10, cv2.COLOR_GRAY2BGR)

    v1 = np.hstack((init_hazy,     prediction, clear))
    v2 = np.hstack((init_depth,    depth,      clear_depth))
    v3 = np.hstack((init_airlight, airlight,   clear_airlight))
    final = np.vstack((v1, v2, v3))
    
    cv2.imshow('Image List', final)
    
    if one_shot_prediction is not None:
        cv2.imshow('One Shot prediction', one_shot_prediction)
    
    # cv2.imshow('Final depth', depth)
    # cv2.imshow('Final prediction', prediction)
    # cv2.imshow("First hazy", init_hazy)
    # cv2.imshow("GT", clear)
    # cv2.imshow("First airlight", init_airlight)
    # cv2.imshow("Final airlight", airlight)
    # cv2.imshow("First depth", init_depth)
    # cv2.imshow("First clear depth", clear_depth)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def all_results_saveORshow(dataRoot, input_name, airlight_step_flag, one_shot_flag, images_dict, saveORshow):
    """
    init_hazy       metrics_best_prediction   psnr_best_prediction  ssim_best_prediction    one_shot_prediction    clear
    init_depth      metrics_best_depth        psnr_best_depth       ssim_best_depth         one_shot_depth         clear_depth
    init_airlight   metrics_best_airlight     psnr_best_airlight    ssim_best_airlight      one_shot_airlight      clear_airlight
    """

    for name, images in images_dict.items():
        if 'depth' in name:
            # images * 255 / 10 = images * 25.5
            images_dict[name] = cv2.cvtColor(np.rint(images_dict[name]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            if np.max(images) <= 1.0:
                images_dict[name] = np.rint(images*255).astype(np.uint8)
            images_dict[name] = cv2.cvtColor(images_dict[name].astype(np.uint8), cv2.COLOR_RGB2BGR)

    if one_shot_flag:
        v1 = np.hstack((images_dict['init_hazy'],     images_dict['metrics_best_prediction'], images_dict['psnr_best_prediction'], images_dict['ssim_best_prediction'], images_dict['one_shot_prediction'], images_dict['clear']))
        v2 = np.hstack((images_dict['init_depth'],    images_dict['metrics_best_depth'],      images_dict['psnr_best_depth'],      images_dict['ssim_best_depth'],      images_dict['init_depth'],          images_dict['clear_depth']))
    else:
        v1 = np.hstack((images_dict['init_hazy'],     images_dict['metrics_best_prediction'], images_dict['psnr_best_prediction'], images_dict['ssim_best_prediction'], images_dict['clear']))
        v2 = np.hstack((images_dict['init_depth'],    images_dict['metrics_best_depth'],      images_dict['psnr_best_depth'],      images_dict['ssim_best_depth'],      images_dict['clear_depth']))
    if airlight_step_flag:
        v3 = np.hstack((images_dict['init_airlight'], images_dict['metrics_best_airlight'],   images_dict['psnr_best_airlight'], images_dict['ssim_best_airlight'],  images_dict['one_shot_airlight'],   images_dict['clear_airlight']))
        final_image = np.vstack((v1, v2, v3))
    else:
        final_image = np.vstack((v1, v2))
    
    if saveORshow == 'show':
        cv2.imshow('Results Images', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif saveORshow == 'save':
        dir_name = dataRoot + 'results/' + input_name.split('\\')[-2]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_path = os.path.join(dir_name, os.path.basename(input_name))
        cv2.imwrite(save_path, final_image)