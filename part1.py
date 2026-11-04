"""SYDE 671 A3 Part 1: homography, inverse warping and feather blending.
Only NumPy is used for the algorithms. Coordinates are (x, y).
H maps a SOURCE image to a REFERENCE image; arrays use image[y, x].
"""
import numpy as np


# A.2: each match contributes TWO equations for EIGHT unknowns.
def build_system(im1_pts, im2_pts):
    p = np.asarray(im1_pts, dtype=float)
    q = np.asarray(im2_pts, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2 or len(p) < 4:
        raise ValueError('Supply at least four corresponding (x, y) pairs.')
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Coordinates must be finite.')
    A, b = [], []
    for (x, y), (u, v) in zip(p, q):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        b.extend([u, v])
    return np.array(A, dtype=float), np.array(b, dtype=float)


def computeH(im1_pts, im2_pts):
    A, b = build_system(im1_pts, im2_pts)
    # lstsq solves the overdetermined system without explicitly inverting A.T@A.
    h, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    if rank < 8:
        raise ValueError('Degenerate points: use points spread over a 2D area.')
    H = np.append(h, 1).reshape(3, 3)
    if abs(np.linalg.det(H)) < 1e-12:
        raise ValueError('Singular homography. Check the point order.')
    return H


def transform_points(points, H):
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (H @ homogeneous.T).T
    if np.any(np.abs(mapped[:, 2]) < 1e-10):
        raise ValueError('A point maps to infinity.')
    return mapped[:, :2] / mapped[:, 2:3]


# A.3: find the output bounds by transforming the four image corners.
def get_canvas(images, homographies):
    corners_in_reference = []
    for image, H in zip(images, homographies):
        height, width = image.shape[:2]
        corners = np.array([[0, 0], [width-1, 0],
                            [width-1, height-1], [0, height-1]])
        denominator = np.column_stack([corners, np.ones(4)]) @ H[2]
        if not (np.all(denominator > 0) or np.all(denominator < 0)):
            raise ValueError('Image crosses the projective horizon; check H.')
        corners_in_reference.append(transform_points(corners, H))
    corners = np.vstack(corners_in_reference)
    # Avoid an extra blank row/column from floating-point round-off.
    corners = np.where(np.abs(corners - np.round(corners)) < 1e-8,
                       np.round(corners), corners)
    xmin, ymin = np.floor(corners.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(corners.max(axis=0)).astype(int)
    return (int(xmin), int(ymin), int(xmax-xmin+1), int(ymax-ymin+1))


def _inverse_warp(im, H, method, canvas=None):
    """Handwritten sampling. A boolean mask distinguishes black pixels from holes."""
    im = np.asarray(im, dtype=float)
    if canvas is None:
        canvas = get_canvas([im], [H])
    xmin, ymin, width, height = canvas
    if width <= 0 or height <= 0 or width*height > 40000000:
        raise ValueError('Invalid or very large canvas; check correspondences.')
    source_h, source_w = im.shape[:2]
    output = np.zeros((height, width) + im.shape[2:], dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    Hinv = np.linalg.inv(H)
    # Process one row at a time: easy to follow and avoids large temporary arrays.
    for row in range(height):
        X = np.arange(width) + xmin
        Y = np.full(width, row + ymin)
        source = Hinv @ np.vstack([X, Y, np.ones(width)])
        denominator = source[2]
        finite = np.abs(denominator) > 1e-10
        x = np.full(width, np.nan)
        y = np.full(width, np.nan)
        x[finite] = source[0, finite] / denominator[finite]
        y[finite] = source[1, finite] / denominator[finite]
        valid = finite & (x >= -1e-8) & (x <= source_w-1+1e-8)
        valid &= (y >= -1e-8) & (y <= source_h-1+1e-8)
        x = np.clip(x[valid], 0, source_w-1)
        y = np.clip(y[valid], 0, source_h-1)
        if method == 'nearest':
            xi = np.floor(x + 0.5).astype(int)
            yi = np.floor(y + 0.5).astype(int)
            values = im[yi, xi]
        elif method == 'bilinear':
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
            x1, y1 = np.minimum(x0+1, source_w-1), np.minimum(y0+1, source_h-1)
            dx, dy = x-x0, y-y0
            if im.ndim == 3:
                dx, dy = dx[:, None], dy[:, None]
            values = ((1-dx)*(1-dy)*im[y0, x0] + dx*(1-dy)*im[y0, x1]
                      + (1-dx)*dy*im[y1, x0] + dx*dy*im[y1, x1])
        else:
            raise ValueError('Unknown interpolation method.')
        output[row, valid] = values
        mask[row, valid] = True
    return output, mask


def warpImageNearestNeighbor(im, H, canvas=None, return_mask=False):
    result, mask = _inverse_warp(im, H, 'nearest', canvas)
    return (result, mask) if return_mask else result


def warpImageBilinear(im, H, canvas=None, return_mask=False):
    result, mask = _inverse_warp(im, H, 'bilinear', canvas)
    return (result, mask) if return_mask else result


# A.4: weights decrease toward each source image's boundary.
def feather_weights(image):
    height, width = image.shape[:2]
    y, x = np.indices((height, width))
    distance = np.minimum.reduce([x+1, width-x, y+1, height-y]).astype(float)
    # A one-pixel floor keeps boundary pixels valid when only one image covers them.
    return distance / distance.max()


def make_mosaic(images, homographies, method='bilinear', canvas=None):
    if canvas is None:
        canvas = get_canvas(images, homographies)
    width, height = canvas[2:]
    if width*height > 40000000:
        raise ValueError('Canvas too large; check points before allocating memory.')
    total_rgb = np.zeros((height, width, 3))
    total_weight = np.zeros((height, width))
    overwrite = np.zeros_like(total_rgb)
    warp = warpImageBilinear if method == 'bilinear' else warpImageNearestNeighbor
    for image, H in zip(images, homographies):
        warped, valid = warp(image, H, canvas, return_mask=True)
        weight = warp(feather_weights(image), H, canvas) * valid
        total_rgb += warped * weight[:, :, None]
        total_weight += weight
        overwrite[valid] = warped[valid]  # Baseline for the report.
    result = np.zeros_like(total_rgb)
    valid = total_weight > 0
    result[valid] = total_rgb[valid] / total_weight[valid, None]
    return result, valid, overwrite, canvas
