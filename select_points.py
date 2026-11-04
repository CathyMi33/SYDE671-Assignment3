"""Optional mouse tool to replace the supplied annotations with your own clicks.
Example: python select_points.py --pair 1 --count 12
Rectification: python select_points.py --rect 1
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent


def click_points(filename, count, title):
    im = ImageOps.exif_transpose(Image.open(BASE/'images'/filename)).convert('RGB')
    im = im.resize((768,1024),Image.Resampling.LANCZOS)
    fig,ax = plt.subplots(figsize=(8,10))
    ax.imshow(im)
    ax.set_title(title+'\nLeft click: add; right click: undo; middle click: finish')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    points = plt.ginput(count,timeout=0)
    plt.close(fig)
    if len(points) != count:
        raise ValueError('Not enough points. No changes were saved.')
    return np.round(points,3).tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pair',type=int,choices=range(1,7))
    group.add_argument('--rect',type=int,choices=range(1,3))
    parser.add_argument('--count',type=int,default=12)
    args = parser.parse_args()
    path = BASE/'points.json'
    data = json.loads(path.read_text())
    if args.pair:
        if args.count < 5:
            parser.error('For A.2 mosaics, choose more than four points.')
        pair = data['pairs'][args.pair-1]
        p = click_points(pair['source'],args.count,'Source: select distinctive corners')
        # A numbered source window helps maintain the same order in the reference.
        im = ImageOps.exif_transpose(Image.open(BASE/'images'/pair['source'])).convert('RGB')
        fig,ax = plt.subplots(figsize=(6,8))
        ax.imshow(im.resize((768,1024)))
        for i,(x,y) in enumerate(p):
            ax.plot(x,y,'r+');ax.text(x+5,y,str(i+1),color='yellow')
        ax.set_title('Source point order - use the SAME order in reference')
        plt.show(block=False)
        q = click_points(pair['reference'],args.count,'Reference: select the SAME corners in order')
        plt.close(fig)
        pair['source_points'],pair['reference_points'] = p,q
        pair['annotation_provenance'] = 'User-selected mouse clicks with select_points.py.'
    else:
        item = data['rectifications'][args.rect-1]
        item['corners'] = click_points(item['image'],4,'Order: top-left, top-right, bottom-right, bottom-left')
        item['annotation_provenance'] = 'User-selected mouse clicks with select_points.py.'
    path.write_text(json.dumps(data,indent=2))
    print('Saved:',path)
    print('Run python run_all.py again to update the results.')
