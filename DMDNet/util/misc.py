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