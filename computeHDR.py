

import numpy as np
import cv2
from hdrtool import hdr
from hdrtool import align
import pandas as pd
import os
import importlib
import time
import matplotlib.pyplot as plt
import argparse


# Settings
parser = argparse.ArgumentParser(description='Compute HDR')
parser.add_argument('--img-dir',default='./example/park3',  type=str,
                    help='path to image folder')
parser.add_argument('--meta-path',default = './example/park3.csv', type=str,
                    help='path to meta data')
parser.add_argument('--save-hdr-to',default = 'output.hdr', type=str,
                    help='path for .hdr file')                   
parser.add_argument('--jpg-output-path',default = 'output.jpg', type=str,
                    help='path for output data')
parser.add_argument('--lwhite', type=float, default=0.8,
                    help='the number for constraint the highest value in hdr image(default: 0.8)')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='The number for correction. Higher value for brighter result; lower for darker(default: 0.5)')


                    
def main():
    args = parser.parse_args()
    IMGDIR = args.img_dir
    META = args.meta_path
    OUTPUT_PATH = args.jpg_output_path
    HDR_PATH = args.save_hdr_to
                    
    #read the meta data
    df = pd.read_csv(META, sep='\s+')
    df['exposetime'] = 1/df['1/shutter_speed']
    
    # read the images
    imgs = [cv2.imread(os.path.join(IMGDIR,fn)) for fn in df.Filename]

    # image alignment
    image_alignment = align.ImageAlignment()

    def solve_alignment(images, d=4):
        for i in range(1, len(images)):
            print('\r[Alignment] %d' % (i + 1), end='')
            images[i] = image_alignment.fit(images[i], images[i-1], d)
        print()
        return images
    ## optional
    #imgs = solve_alignment(imgs)


    # compute high dynamic range image
    hdrimg = hdr.computeHDR(imgs, np.log(df.exposetime))
                    
    # save high dynamic range image
    cv2.imwrite(HDR_PATH, hdrimg)

    # map hdr image to 0-255
    hdr_mapped = hdr.globalToneMapping(hdrimg, Lwhite=np.exp(hdrimg.max())*0.8, alpha=0.5)

    cv2.imwrite(OUTPUT_PATH, (hdr_mapped).astype(np.uint8) )

if __name__ == '__main__':
    main()   
