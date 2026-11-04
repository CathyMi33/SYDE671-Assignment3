"""Small plotting helpers. These do not compute the image transformations."""
from pathlib import Path
import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(image, 0, 1)*255+0.5)).save(buffer, format='PNG')
    path.write_bytes(buffer.getvalue())


def save_figure(fig, path, dpi=150):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    Path(path).write_bytes(buffer.getvalue())


def panels(path, images, titles, points=None):
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5.8), squeeze=False)
    for i, (im, title) in enumerate(zip(images, titles)):
        ax = axes[0, i]
        ax.imshow(np.clip(im, 0, 1), interpolation='nearest', cmap='gray')
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        if points is not None:
            for k, (x, y) in enumerate(points[i]):
                ax.plot(x, y, '+', color='yellow', markersize=7)
                ax.annotate(str(k+1), (x, y), xytext=(3, -8), textcoords='offset points',
                            fontsize=10, color='black', bbox=dict(facecolor='white', alpha=.85, pad=1))
    fig.tight_layout()
    save_figure(fig,path,150)
    plt.close(fig)


def corner_plot(path, image, all_points, kept_points):
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    for ax, p, title, size in zip(axes, [all_points, kept_points],
            [f'Harris: {len(all_points)} candidates', f'ANMS: {len(kept_points)} corners'], [1, 9]):
        ax.imshow(image)
        ax.scatter(p[:, 0], p[:, 1], s=size, c='red', linewidths=0)
        ax.set_title(title); ax.axis('off')
    fig.tight_layout(); save_figure(fig,path,160); plt.close(fig)


def match_plot(path, im1, im2, p, q, mask=None, title='Matches'):
    h, w = im1.shape[:2]
    joined = np.concatenate([im1, im2], axis=1)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(joined)
    if mask is None:
        mask = np.ones(len(p), bool)
    for keep in [False, True]:
        for a, b in zip(p[mask == keep], q[mask == keep]):
            ax.plot([a[0], b[0]+w], [a[1], b[1]],
                    color='#14ce70' if keep else '#ff5959', alpha=.65 if keep else .35, lw=.55)
    ax.scatter(p[:, 0], p[:, 1], s=8, c=np.where(mask, '#14ce70', '#ff5959'))
    ax.scatter(q[:, 0]+w, q[:, 1], s=8, c=np.where(mask, '#14ce70', '#ff5959'))
    ax.set_title(title); ax.axis('off')
    fig.tight_layout(); save_figure(fig,path,170); plt.close(fig)


def descriptor_plot(path, patches, samples, descriptors, points, ids):
    fig, axes = plt.subplots(3, len(ids), figsize=(2*len(ids), 6))
    for col, i in enumerate(ids):
        axes[0, col].imshow(patches[i], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'#{i}: {tuple(points[i])}', fontsize=9)
        axes[1, col].imshow(samples[i], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[2, col].imshow(descriptors[i].reshape(8, 8), cmap='gray', vmin=-3, vmax=3,
                            interpolation='nearest')
        for ax in axes[:, col]:
            ax.set_xticks([]); ax.set_yticks([])
    for ax, label in zip(axes[:, 0], ['Raw 40 x 40', 'Blur + sample 8 x 8', 'Normalize 8 x 8']):
        ax.set_ylabel(label, fontsize=10)
    fig.tight_layout(); save_figure(fig,path,160); plt.close(fig)
