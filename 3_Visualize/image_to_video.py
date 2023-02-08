import argparse
import os
import cv2
from PIL import Image
from natsort import os_sorted

def get_args():
    parser = argparse.ArgumentParser(description='Turn image sequences to video')
    parser.add_argument('--dir', '-d', metavar='DIR', nargs='+', help='Path to image directory', required=True)
    parser.add_argument('--fps', '-f', metavar='FPS', type=int, default=30, help='Video frame per seconds')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    frames = os_sorted(os.listdir(args.dir[0]))

    img = Image.open(os.path.join(args.dir[0], frames[0])).convert("RGB")
    img_size = img.size

    sequence_name = os.path.dirname(args.dir[0]).split('/')[-1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # video codec
    video_writer = cv2.VideoWriter("img-sequences.mp4", fourcc, args.fps, img_size)

    for frame in frames:
        f_path = os.path.join(args.dir[0], frame)
        image = cv2.imread(f_path)
        video_writer.write(image)
        # print(f"{frame} has been written.")

    video_writer.release()