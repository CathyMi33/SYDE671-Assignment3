"""Run all eight required sections. No manual clicks are needed to reproduce results.
Usage: python run_all.py
Core algorithms are in part1.py and part2.py. All coordinates below are (x,y).
"""
import json
import platform
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import PIL
import skimage
from part1 import (build_system, computeH, transform_points, get_canvas,
                   warpImageNearestNeighbor, warpImageBilinear, make_mosaic, feather_weights)
from part2 import (harris_features, anms, extract_descriptors, match_descriptors,
                   ransac_homography, transfer_error)
from plots import save_image, panels, corner_plot, match_plot, descriptor_plot

BASE = Path(__file__).resolve().parent
OUT = BASE/'results'
MAX_SIZE = 1024
NUMBER_CORNERS = 500
RATIO = 0.65
RANSAC_THRESHOLD = 3.0
RANSAC_TRIALS = 2000
SEED = 42


def folder(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(path, array, header=None):
    kw = {} if header is None else dict(header=header, comments='')
    np.savetxt(path, array, delimiter=',', fmt='%.12g', **kw)


def system_files(path, p, q, H):
    folder(path)
    A, b = build_system(p, q)
    errors = transfer_error(p, q, H)
    for name, array in [('A', A), ('b', b), ('H', H)]:
        save_csv(path/(name+'.csv'), array)
    save_csv(path/'points_and_errors.csv', np.column_stack([p, q, errors]),
             'source_x,source_y,target_x,target_y,error_pixels')
    return dict(n=len(p), rank=int(np.linalg.matrix_rank(A)),
                condition=float(np.linalg.cond(A)), H=H.tolist(),
                rmse=float(np.sqrt(np.mean(errors**2))), max_error=float(errors.max()),
                algebraic_residual=float(np.linalg.norm(A @ H.ravel()[:8]-b)),
                p=p.tolist(), q=q.tolist(), errors=errors.tolist())


def compose_homographies(group, edges):
    # The three selected scenes each have two direct links to the center photo.
    reference = group['reference']
    homographies = []
    for name in group['images']:
        if name == reference:
            homographies.append(np.eye(3))
        else:
            homographies.append(edges[(name, reference)])
    return homographies


def bilinear_example(image, H, canvas):
    # Find a valid output sample with visibly fractional source coordinates.
    xmin, ymin, width, height = canvas
    for Y in range(height//3, height//3+100):
        X = width//2
        p = np.linalg.inv(H) @ np.array([X+xmin, Y+ymin, 1.0])
        x, y = p[:2]/p[2]
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        dx, dy = x-x0, y-y0
        if 0 <= x0 < image.shape[1]-1 and 0 <= y0 < image.shape[0]-1 and .15 < dx < .85 and .15 < dy < .85:
            values = image[[y0, y0, y0+1, y0+1], [x0, x0+1, x0, x0+1]]
            weights = np.array([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])
            return dict(output_xy=[X+xmin,Y+ymin], source_xy=[float(x),float(y)],
                        dx=float(dx),dy=float(dy), weights=weights.tolist(),
                        neighbors_rgb=values.tolist(), result_rgb=(weights @ values).tolist(),
                        nearest_rgb=image[int(np.floor(y+.5)),int(np.floor(x+.5))].tolist())
    raise ValueError('No fractional sample found.')


def feather_example(images, Hs, canvas):
    # Only this small numerical example uses an extra warp of the first two images.
    a, ma = warpImageBilinear(images[0], Hs[0], canvas, True)
    b, mb = warpImageBilinear(images[1], Hs[1], canvas, True)
    wa = warpImageBilinear(feather_weights(images[0]), Hs[0], canvas) * ma
    wb = warpImageBilinear(feather_weights(images[1]), Hs[1], canvas) * mb
    ys, xs = np.where(ma & mb & (wa>.2) & (wb>.2))
    i = len(xs)//2; x, y = int(xs[i]), int(ys[i])
    weights = np.array([wa[y,x], wb[y,x]])
    colors = np.array([a[y,x], b[y,x]])
    return dict(canvas_xy=[x,y], reference_xy=[x+canvas[0],y+canvas[1]],
                weights=weights.tolist(), colors=colors.tolist(),
                result_rgb=(weights @ colors / weights.sum()).tolist(),
                note='Two-image example only, not the final three-image pixel.')


def main():
    full_start = time.perf_counter()
    folder(OUT)
    data = json.loads((BASE/'points.json').read_text())
    names = sorted({n for group in data['sets'] for n in group['images']})
    images, image_info = {}, {}
    for name in names:
        im = ImageOps.exif_transpose(Image.open(BASE/'images'/name)).convert('RGB')
        original = list(im.size)
        im.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
        images[name] = np.asarray(im, float)/255
        image_info[name] = dict(original_size=original, working_size=list(im.size))
    metrics = dict(parameters=dict(max_size=MAX_SIZE, anms_number=NUMBER_CORNERS, anms_c=.9,
        descriptor_window=40, descriptor_size=8, descriptor_stride=5, blur_sigma=2,
        ratio_squared_distances=RATIO, ransac_threshold=RANSAC_THRESHOLD,
        ransac_trials=RANSAC_TRIALS, seed=SEED), images=image_info,
        environment=dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__,
                         skimage=skimage.__version__, platform=platform.platform(), cpu=platform.processor()),
        manual={}, rectification={}, features={}, automatic={}, mosaics={})
    manual_edges, auto_edges = {}, {}
    manual_pairs = {}

    # A.1-A.2: manually marked correspondences, least-squares systems and homographies.
    for pair in data['pairs']:
        s, r = pair['source'], pair['reference']; key = Path(s).stem+'_to_'+Path(r).stem
        p, q = np.array(pair['source_points'],float), np.array(pair['reference_points'],float)
        for pts, name in [(p,s),(q,r)]:
            height, width = images[name].shape[:2]
            scale = np.array([width/data['annotation_width'],height/data['annotation_height']])
            pts[:] = (pts+.5)*scale-.5
        start = time.perf_counter(); H = computeH(p,q); elapsed = time.perf_counter()-start
        path = folder(OUT/'A2_manual'/key)
        info = system_files(path,p,q,H); info['solve_seconds'] = elapsed
        panels(path/'correspondences.png',[images[s],images[r]],[s+' source',r+' target'],[p,q])
        metrics['manual'][key] = info; manual_edges[(s,r)] = H; manual_pairs[key] = (p,q)
        print('Manual',key,'RMSE',round(info['rmse'],3),flush=True)

    # A.3: two rectifications and five timed repeats after one warm-up per method.
    for item in data['rectifications']:
        path = folder(OUT/'A3_rectification'/item['name']); im = images[item['image']]
        p = np.array(item['corners'],float); width,height = item['output_size']
        q = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],float)
        H = computeH(p,q); canvas = (0,0,width,height)
        info = system_files(path,p,q,H); info.update(output_size=[width,height],aspect_note=item['aspect_note'])
        results = []
        for name, warp in [('nearest',warpImageNearestNeighbor),('bilinear',warpImageBilinear)]:
            warp(im,H,canvas)
            times = []
            for repeat in range(5):
                start = time.perf_counter(); result,mask = warp(im,H,canvas,True)
                times.append(time.perf_counter()-start)
            info[name+'_times'] = times; info[name+'_median'] = float(np.median(times))
            save_image(path/(name+'.png'),result); save_image(path/(name+'_mask.png'),mask)
            results.append(result)
        panels(path/'comparison.png',[im]+results,[item['image'],'Nearest neighbor','Bilinear'],[p,[],[]])
        x0,y0 = width//3,height//4
        panels(path/'detail.png',[a[y0:y0+100,x0:x0+100] for a in results],['Nearest: 100 x 100 crop','Bilinear: same crop'])
        info['difference_mae'] = float(np.abs(results[0]-results[1]).mean())
        info['bilinear_example'] = bilinear_example(im,H,canvas)
        metrics['rectification'][item['name']] = info

    # B.1-B.2: Harris, ANMS, blurred samples, and normalized descriptors for all 9 images.
    features = {}
    for name in names:
        path = folder(OUT/'B1_B2_features'/Path(name).stem)
        im = images[name]
        start = time.perf_counter(); p,scores = harris_features(im); harris_time=time.perf_counter()-start
        start = time.perf_counter(); ids,radii = anms(p,scores,NUMBER_CORNERS); anms_time=time.perf_counter()-start
        start = time.perf_counter(); kept,desc,patches,samples,stats = extract_descriptors(im,p[ids]); desc_time=time.perf_counter()-start
        features[name] = (kept,desc)
        selection = np.zeros(len(p),int); selection[ids]=1
        save_csv(path/'harris_anms.csv',np.column_stack([p,scores,radii,selection]),'x,y,response,radius_px,selected')
        save_csv(path/'descriptor_points.csv',kept,'x,y')
        save_csv(path/'descriptors.csv',desc)
        np.savez_compressed(path/'features.npz',points=kept,descriptors=desc,samples=samples,statistics=stats)
        corner_plot(path/'corners.png',im,p,kept)
        example_ids = np.linspace(0,len(kept)-1,6,dtype=int)
        descriptor_plot(path/'descriptors.png',patches,samples,desc,kept,example_ids)
        for i in example_ids:
            save_csv(path/f'sample_{i}_8x8.csv',samples[i])
            save_csv(path/f'normalized_{i}_8x8.csv',desc[i].reshape(8,8))
        # One finite-radius selected point gives a concrete ANMS calculation.
        i = next(int(i) for i in ids if np.isfinite(radii[i]))
        stronger = np.where(scores[i] < .9*scores)[0]
        j = int(stronger[np.argmin(np.linalg.norm(p[stronger]-p[i],axis=1))])
        info = dict(harris_count=len(p),anms_count=len(ids),descriptor_count=len(kept),
            harris_seconds=harris_time,anms_seconds=anms_time,descriptor_seconds=desc_time,
            descriptor_example=dict(index=0,point=kept[0].tolist(),mean=float(stats[0,0]),std=float(stats[0,1]),
                samples=samples[0].tolist(),normalized=desc[0].reshape(8,8).tolist()),
            anms_example=dict(index=i,point=p[i].tolist(),score=float(scores[i]),
                stronger_index=j,stronger_point=p[j].tolist(),stronger_score=float(scores[j]),radius=float(radii[i])))
        metrics['features'][name]=info
        print('Features',name,len(p),'->',len(kept),'seconds',round(harris_time+anms_time+desc_time,2),flush=True)

    # B.3-B.4: ratio matching, RANSAC, and least-squares refitting.
    for pair in data['pairs']:
        s,r = pair['source'],pair['reference']; key = Path(s).stem+'_to_'+Path(r).stem
        path = folder(OUT/'B3_B4_matches'/key)
        ps,ds=features[s]; pr,dr=features[r]
        start=time.perf_counter(); matches,values=match_descriptors(ds,dr,RATIO); match_time=time.perf_counter()-start
        p,q = ps[matches[:,0]],pr[matches[:,1]]
        start=time.perf_counter()
        H,mask,errors,fit_mask,skipped=ransac_homography(p,q,RANSAC_THRESHOLD,RANSAC_TRIALS,SEED)
        ransac_time=time.perf_counter()-start
        auto_edges[(s,r)]=H
        info=system_files(path/'final_refit',p[fit_mask],q[fit_mask],H)
        save_csv(path/'all_matches.csv',np.column_stack([matches,p,q,values,errors,mask,fit_mask]),
            'source_feature,target_feature,source_x,source_y,target_x,target_y,SSD1,SSD2,ratio,error_px,inlier,used_for_refit')
        match_plot(path/'ratio_matches.png',images[s],images[r],p,q,title=f'{key}: {len(p)} ratio matches')
        match_plot(path/'ransac_matches.png',images[s],images[r],p,q,mask,
                   title=f'{key}: {mask.sum()} inliers (green), {(~mask).sum()} outliers (red)')
        # A failed baseline: fit all ratio matches without removing outliers.
        H_all=computeH(p,q); errors_all=transfer_error(p,q,H_all)
        save_csv(path/'H_without_RANSAC.csv',H_all)
        manual_p,manual_q=manual_pairs[key]
        manual_error_auto=transfer_error(manual_p,manual_q,H)
        info.update(matches=len(p),inliers=int(mask.sum()),outliers=int((~mask).sum()),
            inlier_rmse=float(np.sqrt(np.mean(errors[mask]**2))),
            match_seconds=match_time,ransac_seconds=ransac_time,skipped_samples=skipped,
            manual_landmark_rmse=float(np.sqrt(np.mean(manual_error_auto**2))),
            no_ransac_median_error=float(np.median(errors_all)),
            no_ransac_inliers=int((errors_all<RANSAC_THRESHOLD).sum()),
            no_ransac_errors=errors_all.tolist(),all_errors=errors.tolist(),
            ratio_example=dict(source_feature=int(matches[0,0]),target_feature=int(matches[0,1]),
                SSD1=float(values[0,0]),SSD2=float(values[0,1]),ratio=float(values[0,2])),
            threshold_tests=[])
        save_csv(path/'auto_errors_on_manual_landmarks.csv',np.column_stack([manual_p,manual_q,manual_error_auto]),
            'source_x,source_y,target_x,target_y,error_px')
        # Threshold comparison uses the same image pair and seed; final settings stay fixed.
        if key=='IMG_1483_to_IMG_1485':
            for threshold in [.4,.55,.65,.8]:
                mm,vv=match_descriptors(ds,dr,threshold)
                _,mk,ee,_,_=ransac_homography(ps[mm[:,0]],pr[mm[:,1]],seed=SEED)
                info['threshold_tests'].append(dict(ratio=threshold,matches=len(mm),inliers=int(mk.sum()),
                    inlier_rmse=float(np.sqrt(np.mean(ee[mk]**2)))))
        metrics['automatic'][key]=info
        print('Auto',key,len(p),'matches',mask.sum(),'inliers',flush=True)

    # A.4 and B.4: three manual and three automatic mosaics on identical per-group canvases.
    for group in data['sets']:
        path=folder(OUT/'mosaics'/group['name']); ns=group['images']; ims=[images[n] for n in ns]
        hm=compose_homographies(group,manual_edges); ha=compose_homographies(group,auto_edges)
        canvas=get_canvas(ims+ims,hm+ha)
        panels(path/'sources.png',ims,ns)
        result_images=[]; info=dict(images=ns,reference=group['reference'],canvas=list(canvas),manual_Hs=[],auto_Hs=[])
        for kind,Hs in [('manual',hm),('auto',ha)]:
            for name,H in zip(ns,Hs):
                save_csv(path/(kind+'_'+Path(name).stem+'_H.csv'),H)
                info[kind+'_Hs'].append(H.tolist())
            start=time.perf_counter(); mosaic,valid,overwrite,_=make_mosaic(ims,Hs,canvas=canvas)
            info[kind+'_warp_blend_seconds']=time.perf_counter()-start
            save_image(path/(kind+'_mosaic.png'),mosaic)
            save_image(path/(kind+'_mosaic_rgba.png'),np.dstack([mosaic,valid]))
            save_image(path/(kind+'_mask.png'),valid)
            save_image(path/(kind+'_overwrite.png'),overwrite)
            panels(path/(kind+'_blending.png'),[overwrite,mosaic],['Hard overwrite','Feather blend'])
            result_images.append(mosaic)
        panels(path/'manual_vs_auto.png',result_images,['Manual correspondences','Automatic: ANMS + RANSAC'])
        # A common reference-coordinate crop lets us inspect registration at the same location.
        crop_xy = (125,390,620,640) if group['name']=='01_bulletin_board' else ((100,300,600,620) if group['name']=='03_atrium' else (100,450,600,850))
        x0,y0,x1,y1=crop_xy; xmin,ymin=canvas[:2]
        panels(path/'detail_comparison.png',[im[y0-ymin:y1-ymin,x0-xmin:x1-xmin] for im in result_images],
               ['Manual: same reference crop','Automatic: same reference crop'])
        info['detail_crop_reference_xyxy']=list(crop_xy)
        if group['name']=='01_bulletin_board':
            info['feather_example']=feather_example(ims,hm,canvas)
        metrics['mosaics'][group['name']]=info
        print('Mosaic',group['name'],canvas,flush=True)

    metrics['full_run_seconds']=time.perf_counter()-full_start
    metrics['timing_note']='Full run includes plotting, CSV/image saves, 5-repeat rectification benchmarks and ratio sweep. Algorithm stage times exclude plotting and file I/O. Manual point selection is not timed.'
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    # Machine-readable concise table as well as the full JSON.
    rows=[]
    for key,info in metrics['automatic'].items():
        rows.append([key,info['matches'],info['inliers'],info['inlier_rmse'],
                     metrics['manual'][key]['rmse'],info['manual_landmark_rmse'],info['match_seconds'],info['ransac_seconds']])
    import csv
    with (OUT/'pair_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pair','matches','inliers','auto_inlier_RMSE','manual_fit_RMSE',
            'auto_RMSE_on_manual_landmarks','match_seconds','ransac_seconds']);writer.writerows(rows)
    print('Finished in',round(metrics['full_run_seconds'],1),'seconds. Now run python build_report.py',flush=True)


if __name__=='__main__':
    main()
