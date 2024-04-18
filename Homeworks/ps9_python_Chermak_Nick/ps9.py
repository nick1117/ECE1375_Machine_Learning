import os
import numpy as np
from PIL import Image
from kmeans_multiple import kmeans_multiple
import matplotlib.pyplot as plt

#part c
image1 = Image.open('input\\Im1.jpg').resize((100, 100))
image2 = Image.open('input\\Im2.jpg').resize((100, 100))
image3 = Image.open('input\\Im3.png').resize((100, 100))

image1 = np.array(image1)
image2 = np.array(image2)
image3 = np.array(image3)

image1 = np.reshape(image1, (100 * 100, 3))
image2 = np.reshape(image2, (100 * 100, 3))
image3 = np.reshape(image3, (100 * 100, 3))

images = [image1, image2, image3]

K_values = [3, 5, 7]
iters_values = [7, 13, 20]
R_values = [5, 15, 30]

#fig, axes = plt.subplots(len(K_values), len(R_values)*len(iters_values), figsize=(20, 20))
for i, image in enumerate(images):
    #for i, ax in enumerate(axes.flat):
        for K in K_values:
            for iters in iters_values:
                for R in R_values:
                    final_ids, final_means, final_ssd = kmeans_multiple(image, K, iters, R)
                    recolored_image = final_means[final_ids].astype(int)
                    recolored_image = np.reshape(recolored_image, (100, 100, 3))
                    recolored_image = recolored_image.astype(np.uint8)
                    #ax.imshow(recolored_image)
                    #ax.set_title(f'K={K}, iters={iters}, R={R}')
                    plt.imshow(recolored_image)
                    plt.title(f'K={K}, iters={iters}, R={R}')
                    plt.savefig(f"output\\ps9-image{i+1}-K={K}-iters={iters}-R={R}.png")
                    plt.show()
    #plt.tight_layout()
    #plt.show()
