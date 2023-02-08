import argparse
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Turn image sequences to video')
    parser.add_argument('--vid1', '-v1', metavar='VID1', help='Path to video1', required=True)
    parser.add_argument('--vid2', '-v2', metavar='VID2', help='Path to video2', required=True)
    parser.add_argument('--out', '-o', metavar='OUT', help='Name of combined video', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # Read video
    videoLeft = cv2.VideoCapture(args.vid1)
    videoRight = cv2.VideoCapture(args.vid2)

    # Get video info
    fps = videoLeft.get(cv2.CAP_PROP_FPS)
    width = (int(videoLeft.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(videoLeft.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height*2))

    # Read the first frame of data from each of the 2 videos
    successLeft,frameLeft = videoLeft.read()
    successRight, frameRight = videoRight.read()

    while successLeft and successRight:

        # Resize for putting together
        frameLeft = cv2.resize(frameLeft, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        frameRight = cv2.resize(frameRight, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

        # Horizontally or vertically stacked arrays 
        # frame = np.hstack((frameLeft, frameRight))
        frame = np.vstack((frameLeft, frameRight))

        # Save the processed video frames
        videoWriter.write(frame)

        # Cycles through next frame
        successLeft,frameLeft = videoLeft.read()
        successRight, frameRight = videoRight.read()

    # Output
    videoWriter.release()
    videoLeft.release()
    videoRight.release()