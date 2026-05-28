import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm

import mmcv
from mmseg.datasets import build_dataset
from mmcv.utils import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('pkl', help='predictions pkl file path')
    parser.add_argument('--out-dir', default='work_dirs/vis', help='output directory')
    parser.add_argument('--max-samples', type=int, default=0, help='max samples to visualize (0=all)')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.pkl)

    mmcv.mkdir_or_exist(args.out_dir)

    palette = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)

    total = len(dataset) if args.max_samples <= 0 else args.max_samples
    for i in tqdm(range(total)):
        img_info = dataset.img_infos[i]
        filename = img_info['filename']

        # Original optical image (show first 3 channels as RGB)
        opt_path = osp.join(dataset.img_dir, filename)
        img = mmcv.imread(opt_path, 'unchanged')  # (256, 256, 4)
        if img.ndim == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img[:, :, :3]  # take first 3 channels
        # normalize to [0, 255]
        if img_rgb.dtype != np.uint8:
            img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8) * 255).astype(np.uint8)

        # Ground truth
        gt_path = osp.join(dataset.ann_dir, filename)
        gt = mmcv.imread(gt_path, 'unchanged')
        gt_color = palette[gt]

        # Prediction
        pred = results[i].astype(np.uint8)
        pred_color = palette[pred]

        # Composite: original | ground truth | prediction
        composite = np.concatenate([img_rgb, gt_color, pred_color], axis=1)

        out_path = osp.join(args.out_dir, filename)
        mmcv.imwrite(composite, out_path)

    print(f'Saved {total} images to {args.out_dir}')


if __name__ == '__main__':
    main()
