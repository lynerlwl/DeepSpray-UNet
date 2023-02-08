import argparse
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from natsort import os_sorted

def get_args():
    parser = argparse.ArgumentParser(description='Isolate each classes from a binary mask')
    parser.add_argument('--dir', '-d', metavar='DIR', type=str, help='Path to image directory', required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()

    filenames = [name for name in os_sorted(os.listdir(args.dir)) if not name.startswith('.')]

    label_dict = {'droplet': 0, 'detached_ligament': 0, 'attached_ligament':0, 'lobe':0}
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white', 'blue'])

    for file in filenames: 
        path = f"{args.dir}/count/{file}"
        CHECK_FOLDER = os.path.isdir(path)
        if not CHECK_FOLDER:
            os.makedirs(path)
        
        load_name = f"{args.dir}/{file}"
        img = Image.open(load_name)
        img_arr = np.array(img)
        
        for i in range(0,4):
            selected_pixel = img_arr.copy()
            selected_pixel[selected_pixel != i+1] = 0
            
            if len(np.unique(selected_pixel)) == 2:
                save_name = f"{args.dir}/count/{file}/{list(label_dict)[i]}"
    #             print(save_name)
                plt.imshow(selected_pixel, cmap=cmap)
                plt.axis('off')
                figure = plt.gcf()
                figure.set_size_inches(6.5, 6.5)
                plt.tight_layout()
                plt.savefig(save_name, bbox_inches='tight', pad_inches = 0, dpi=300)
    #             plt.show()