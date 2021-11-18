import numpy as np

def cosine_similarity(a, b, eps=1e-12):
    a1 = a / np.expand_dims(np.fmax(np.linalg.norm(a, axis=-1), eps), axis=-1)
    b1 = b / np.expand_dims(np.fmax(np.linalg.norm(b, axis=-1), eps), axis=-1)
    return np.matmul(a1, b1.T)


if __name__ == '__main__':
    input_dataset = np.load('input_dataset.npy')
    input_denorm = np.load('input_denorm.npy')
    input_255 = np.load('input_255.npy')
    input_Origin = np.load('input_Origin.npy')
    
    print(input_255)
    print(input_Origin)
    # print(cosine_similarity(input_255, input_Origin))
    