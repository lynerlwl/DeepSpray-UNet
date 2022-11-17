import argparse
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Convert binary mask to RGB image')
    parser.add_argument('--path', '-p', metavar='PATH', nargs='+', help='Path to binary mask', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white', 'red', 'green', 'yellow', 'blue'])

    filenames = [name for name in os.listdir(args.input[0]) if not name.startswith('.')] 

    if os.path.exists(f"{args.input[0]}-color") == False:
        os.makedirs(f"{args.input[0]}-color")

    for file in filenames:
        load_name = f"{args.input[0]}/{file}"
        img = Image.open(load_name)
        save_name = f"{args.input[0]}-color/{file}"
        
        plt.imshow(np.array(img), cmap=cmap)
        plt.axis('off')
        figure = plt.gcf()
        figure.set_size_inches(6.5, 6.5)
        plt.tight_layout()
        plt.savefig(save_name, bbox_inches='tight', pad_inches = 0, dpi=300)