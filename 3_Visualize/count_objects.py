import argparse
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
# font = ImageFont.truetype(<font-file>, <font-size>)
# font-file should be present in provided path.
font = ImageFont.truetype("SansSerifBldFLFCond.otf", 50)
import cv2 
import itertools
from natsort import os_sorted

def get_args():
    parser = argparse.ArgumentParser(description='Count objects for each class')
    parser.add_argument('--dir', '-d', metavar='DIR', type=str, help='Path to image to read', required=True)
    parser.add_argument('--out', '-o', metavar='OUT', type=str, help='Path to image to write', required=True)
    return parser.parse_args()

def calculate_centroid(cnt):
    M = cv2.moments(cnt)
    cX = int(M['m10']/M['m00'])
    cY = int(M['m01']/M['m00'])  
    return (cX, cY)

def calculate_distance(a, b):
    euclidean_distance = ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
    return euclidean_distance 

def remove_overlap(r):
    selected_contours = [c for c in contours if 50 <= cv2.arcLength(c,True)]
    point_list = [calculate_centroid(i) for i in selected_contours]
    point_combined = itertools.combinations(point_list, 2)
    d_list = ['y' for a,b in point_combined if calculate_distance(a,b) <= (r*2)]
    count = len(selected_contours) - len(d_list)
    return count

if __name__ == '__main__':
    args = get_args()

    label_dict = {'droplet': 0, 'detached_ligament': 0, 'attached_ligament':0, 'lobe':0}

    img_path = [os.path.join((args.dir),f) for f in os_sorted(os.listdir(args.dir)) if not f.startswith('.')]

    for path in img_path:
        for i in label_dict:
            image = cv2.imread(f"{path}/{i}.png")
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #to RGB 
            gray_img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #to grayscale

            # Applying binary thresholding on the image  
            _, binary = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV) 
            # Grab the contours in the image
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if i == 'droplet':
                label_dict[i] = len(contours)
        
            if i == 'detached_ligament':
                label_dict[i] = remove_overlap(30)
            
            if i == 'attached_ligament':
                label_dict[i] = remove_overlap(20)
            
            if i == 'lobe':
                label_dict[i] = remove_overlap(15)

        # Write the counts on the image    
        image = Image.open(f"{args.out}/{path.split('/')[-1]}").convert('RGB')

        ImageDraw.Draw(image).text(
            (10, 10),  # Coordinates
            f"{label_dict}",  # Text
            (0, 0, 0),  # Color
            font #font
        )

        

        image.save(f"{args.dir}/whiteBg_{path.split('/')[-1]}")