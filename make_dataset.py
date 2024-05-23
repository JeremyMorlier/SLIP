# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import pickle
import re
from urllib.parse import unquote
from tqdm import tqdm

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--metadata", default="yfcc100m_dataset", type=str, help="YFCC100M metadata file path")
    parser.add_argument("--images_ids", type=str, default="flickr_unique_ids", help="SLIP flickr images ids")
    parser.add_argument("--yfcc15m_tsv", type=str, default="yfcc100m_subset_data.tsv", help="OpenAI YFCC100M subset: YFCC15M")
    parser.add_argument("--output", type=str, default="temp/", help="output pickle directory")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    DATASET = args.metadata

    cleanhtml = re.compile('<a.*?>|</a>|<b>|</b>|<i>|</i>')
    cleanurl = re.compile('http\S+|www\S+')

    print('=> loading YFCC image ids')
    image_ids = np.load(args.images_ids)
    image_ids = set(image_ids)

    print('=> loading CLIP image ids')
    clip_ids = set()
    with open(args.yfcc15m_tsv) as f:
        for l in tqdm(f.readlines()):
            row = l.strip().split('\t')
            clip_ids.add(int(row[0]))

    print('=> collecting and cleaning subset captions')
    captioned = []
    uncaptioned = []
    with open('yfcc100m_dataset.txt') as f:
        for l in tqdm(f.readlines()):
            row = l.strip().split('\t')
            if int(row[0]) in image_ids:
                uncaptioned.append(int(row[0]))
                if int(row[0]) in clip_ids:
                    title = unquote(row[8]).replace('+', ' ')
                    title = re.sub(cleanhtml, '', title)
                    title = re.sub(cleanurl, '', title)

                    desc = unquote(row[9]).replace('+', ' ')
                    desc = re.sub(cleanhtml, '', desc)
                    desc = re.sub(cleanurl, '', desc)
                    
                    captioned.append((int(row[0]), title, desc))

    with open(os.path.join(args.output, 'yfcc15m.pkl'), 'wb') as f:
        pickle.dump(captioned, f)

    with open(os.path.join(args.output, 'yfcc100m.pkl'), 'wb') as f:
        pickle.dump(uncaptioned, f)

    print('Total captioned images:', len(captioned))  # 14689580
    print('Total uncaptioned images:', len(uncaptioned))  # 95920149
