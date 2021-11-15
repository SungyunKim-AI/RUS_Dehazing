"""
I_min * (1 - ((I_max - I_min) / max(1, I_max)))
"""
import cv2
import numpy as np
import copy

class Hazy_Density():
    def get_min(self, image):
        R, G, B = cv2.split(image)
        min_channel = np.amin([R, G, B],0)
        return min_channel
    
    def get_max(self, image):
        R, G, B = cv2.split(image)
        max_channel = np.amax([R, G, B],0)
        return max_channel
        
    def get_density(self, image):
        image = cv2.imread(image)
        I_min = self.get_min(image)
        I_max = self.get_max(image)
        I_max_threshold = copy.deepcopy(I_max)
        I_max_threshold[I_max <= 1] = 1
        density = I_min * (1 - ((I_max - I_min) / I_max_threshold))
        
        return density

if __name__=='__main__':
    file = '/Users/sungyoon-kim/Documents/GitHub/RUS_Dehazing/Airlight/test01.jpg'
    
    density = Hazy_Density().get_density(file)
    print(density.shape)
        