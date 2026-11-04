"""B.1-B.4: simple single-scale feature matching and four-point RANSAC.
Only the supplied Harris detector uses scikit-image. Other steps use NumPy.
"""
import itertools
import numpy as np
from harris import get_harris_corners
from part1 import computeH, transform_points


def to_gray(image):
    return image[:, :, :3] @ np.array([0.299, 0.587, 0.114])


def harris_features(image):
    response, yx = get_harris_corners(to_gray(image), edge_discard=20)
    # The supplied function returns (y,x). Our other functions use (x,y).
    points = yx.T[:, [1, 0]]
    scores = response[yx[0], yx[1]]
    good = np.isfinite(scores) & (scores > 0)
    return points[good], scores[good]


def anms(points, scores, number=500, c=0.9):
    radii = np.full(len(points), np.inf)
    for i in range(len(points)):
        stronger = scores[i] < c * scores
        if np.any(stronger):
            distance = np.linalg.norm(points[stronger] - points[i], axis=1)
            radii[i] = distance.min()
    # Large radius first. Score and original index break equal-radius ties.
    order = np.lexsort((np.arange(len(points)), -scores, -radii))
    return order[:min(number, len(points))], radii


def gaussian_blur(image, sigma=2.0):
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x*x / (2*sigma*sigma))
    kernel /= kernel.sum()
    height, width = image.shape
    padded = np.pad(image, ((0, 0), (radius, radius)), mode='reflect')
    horizontal = np.zeros_like(image)
    for i, weight in enumerate(kernel):
        horizontal += weight * padded[:, i:i+width]
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode='reflect')
    result = np.zeros_like(image)
    for i, weight in enumerate(kernel):
        result += weight * padded[i:i+height, :]
    return result


def extract_descriptors(image, points):
    gray = to_gray(image)
    smooth = gaussian_blur(gray, sigma=2.0)
    kept, descriptors, patches, samples, statistics = [], [], [], [], []
    for x, y in points:
        x, y = int(x), int(y)
        if x < 20 or y < 20 or x+20 > gray.shape[1] or y+20 > gray.shape[0]:
            continue
        # 40x40 window; take indices 2,7,12,...,37 -> 8x8.
        small = smooth[y-20:y+20, x-20:x+20][2::5, 2::5]
        mean, std = small.mean(), small.std()
        if std < 1e-6:
            continue
        descriptors.append(((small-mean)/std).ravel())
        kept.append([x, y])
        patches.append(gray[y-20:y+20, x-20:x+20])
        samples.append(small)
        statistics.append([mean, std])
    return (np.array(kept).reshape(-1, 2), np.array(descriptors).reshape(-1, 64),
            np.array(patches).reshape(-1, 40, 40), np.array(samples).reshape(-1, 8, 8),
            np.array(statistics).reshape(-1, 2))


def match_descriptors(d1, d2, ratio_threshold=0.65):
    # The ratio uses SQUARED Euclidean distances (SSD), not Euclidean distances.
    if len(d1) == 0 or len(d2) < 2:
        return np.empty((0, 2), int), np.empty((0, 3))
    candidates = []
    for i in range(len(d1)):
        distances = np.sum((d2-d1[i])**2, axis=1)
        order = np.argsort(distances, kind='stable')
        best, second = order[:2]
        if distances[second] <= 1e-12:
            continue  # Identical candidates are ambiguous.
        ratio = distances[best] / distances[second]
        if ratio < ratio_threshold:
            candidates.append((ratio, i, int(best), distances[best], distances[second]))
    # Keep at most one source per target, preferring the smaller ratio.
    candidates.sort()
    pairs, values, used = [], [], set()
    for ratio, i, j, first, second in candidates:
        if j not in used:
            pairs.append([i, j]); values.append([first, second, ratio]); used.add(j)
    return np.array(pairs, int).reshape(-1, 2), np.array(values).reshape(-1, 3)


def transfer_error(p, q, H):
    try:
        return np.linalg.norm(transform_points(p, H)-q, axis=1)
    except ValueError:
        return np.full(len(p), np.inf)


def degenerate_four(points):
    # Any three collinear points make a four-point sample unreliable.
    for a, b, c in itertools.combinations(points, 3):
        u, v = b-a, c-a
        twice_area = abs(u[0]*v[1]-u[1]*v[0])
        if twice_area < 1.0:
            return True
    return False


def ransac_homography(p, q, threshold=3.0, iterations=2000, seed=42):
    if len(p) < 4:
        raise ValueError('RANSAC needs at least four matches.')
    rng = np.random.default_rng(seed)
    best_mask, best_H = None, None
    best_count, best_median = 0, np.inf
    skipped = 0
    for trial in range(iterations):
        ids = rng.choice(len(p), 4, replace=False)
        if degenerate_four(p[ids]) or degenerate_four(q[ids]):
            skipped += 1
            continue
        try:
            H = computeH(p[ids], q[ids])
        except (ValueError, np.linalg.LinAlgError):
            skipped += 1
            continue
        errors = transfer_error(p, q, H)
        mask = errors < threshold
        count = int(mask.sum())
        median = np.median(errors[mask]) if count else np.inf
        if count > best_count or (count == best_count and median < best_median):
            best_count, best_median = count, median
            best_mask, best_H = mask, H
    if best_count < 4:
        raise ValueError('No valid homography found.')
    # Least-squares refit with the consensus. Stop at stable membership or 5 passes.
    fit_mask = best_mask.copy()
    for step in range(5):
        proposed_fit = best_mask.copy()
        try:
            H = computeH(p[proposed_fit], q[proposed_fit])
        except (ValueError, np.linalg.LinAlgError):
            break
        new_mask = transfer_error(p, q, H) < threshold
        if new_mask.sum() < 4:
            break
        best_H = H
        fit_mask = proposed_fit
        if np.array_equal(new_mask, best_mask):
            best_mask = new_mask
            break
        best_mask = new_mask
    # Record the exact final fit set separately from final threshold membership.
    errors = transfer_error(p, q, best_H)
    return best_H, errors < threshold, errors, fit_mask, skipped
