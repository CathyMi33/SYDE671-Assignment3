# Assignment #3: [Auto]Stitching Photo Mosaics

SYDE 671

Part 1: IMAGE WARPING and MOSAICING

Part 2: FEATURE MATCHING for AUTOSTITCHING

## Report and webpage

Open `index.html` to read the project webpage. The mathematical SVGs are embedded in the HTML. Keep `results/` and `report_assets/` alongside it. `Assignment3_Report.pdf` is the same saved PDF report supplied in the complete assignment package. The figures are copied without resampling or recompression.

## Run the algorithms

Python 3.12 was used for the saved experiment. From this directory:

```sh
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the manual and automatic experiments and writes results. It does not rebuild the static report or webpage. `plots.py` supplies plotting functions. For the recorded BLAS setting, set `OPENBLAS_NUM_THREADS=1` before starting Python. Runtime depends on the computer and installed versions. Python runs locally, not in GitHub Pages.

| Question | Code |
|---|---|
| A.1 | Image loading and scene groups in `run_all.py` |
| A.2 | Homography equations and least-squares estimation in `part1.py` |
| A.3 | Nearest-neighbor and bilinear inverse warping in `part1.py` |
| A.4 | Canvas bounds and feather blending in `part1.py` |
| B.1 | Supplied `harris.py`, and ANMS in `part2.py` |
| B.2 | Descriptor extraction in `part2.py` |
| B.3 | SSD ratio matching in `part2.py` |
| B.4 | Four-point RANSAC in `part2.py`, with mosaic functions in `part1.py` |

## Inputs and saved calculations

`images/` contains the nine original photos. `points.json` contains the manual correspondence sets and rectification quadrilaterals. Working images are 768 x 1024 after EXIF orientation and resizing. Coordinates use (x,y); array indexing uses [y,x].

`select_points.py` is an optional mouse-based coordinate picker:

```sh
python select_points.py --pair 1 --count 12
python select_points.py --rect 1
```

`saved_calculations.zip` contains the saved CSV, JSON and NPZ results, including matrices, residuals, descriptors, matching data and metrics. Extract it into this directory to merge its `results/` tree with the displayed figures. It is not needed just to view the website or run the main experiment.

## GitHub Pages

Upload the contents of this directory to the repository root, with `index.html` directly at the root. In Settings > Pages, choose Deploy from a branch, `main`, and `/(root)`, then Save. Use the published URL shown by GitHub. Do not upload the outer ZIP as the website itself.

## Interpretation and attribution

Fitting RMSE and automatic inlier RMSE use different point sets; neither measures independent ground-truth accuracy. The atrium retains visible depth-dependent misalignment despite small inlier errors. The report includes that limitation. The two rectification output aspect ratios are chosen, not physically measured.

I used AI assistance for code, visual correspondence selection, calculations and report preparation. Visual correspondences were measured from landmarks, not copied from automatic feature matches. The provided `harris(2).py` is included as `harris.py` with identical bytes, including its attribution. The algorithm code implements homography estimation, inverse sampling, blending, ANMS, descriptors, ratio matching and RANSAC without high-level homography or projective-warp functions.
