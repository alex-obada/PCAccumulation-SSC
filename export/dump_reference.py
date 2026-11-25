"""Convert the selected frames to .bin inputs for the C++ project, plus reference outputs of the
original model for the C++ module tests (Phase 0, steps 1 and 3).

The .npz frames use pickle, so the C++ program never reads them. This script writes, per frame,
a directory  <out>/<seq>_<frame>/  with files in the common/binio.py format:

  inputs (every selected frame)
    raw_points.bin   [Nr, 3] f32   all points of the 5 sweeps, before crop / ground removal
    raw_time.bin     [Nr]    i32   sweep index 0..4 of each point
    poses_gt.bin     [5,4,4] f32   ground-truth ego-motion (only for evaluating the estimate)

  ref/ (only with --ref, a subset of frames; what the original model computes at each step)
    points.bin       [N, 3]  f32   after crop + ground removal (order kept)          preprocess
    time_indice.bin  [N]     i32
    coordinates.bin  [M, 4]  i32   (z, y, x, t) per pillar, numba order               voxelize
    p2v.bin          [N]     i32   pillar index of each point (numba numbering)
    num_points.bin   [M]     i32
    pillar_mean.bin  [M, 3]  f32   mean xyz of the points in each pillar               pillar_ops
    pfn_a_in.bin     [N, 9]  f32   normalized per-point features
    pfn_a_out.bin    [N, 32] f32
    pfn_b_in.bin     [N, 64] f32   cat(pfn_a_out, pillar_max(pfn_a_out)[p2v])
    pfn_b_out.bin    [N, 32] f32
    pfn_c_in.bin     [N, 64] f32
    pfn_c_out.bin    [N, 32] f32
    pillar_feats.bin [M, 32] f32   pillar_max(pfn_c_out) = PillarFeatureNet output
    bev.bin          [5, 32, 288, 288] f32   backbone input (~53 MB)
    fb_seg.bin       [5, 2, 288, 288] f32    backbone output
    poses_est.bin    [5, 4, 4] f32           ego-motion estimate
    mos_label.bin    [N] i32,  offset_est.bin [N, 2] f32,  inst_label.bin [N] i32   final outputs

Usage:
    python export/dump_reference.py                        # inputs for all frames, ref for 3
    python export/dump_reference.py --ref 0 50 100 --only-ref
    bash export/dump_reference.sh ...                      # same, with the venv activated
"""
import argparse, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch_scatter import scatter

from common.binio import write_bin
from common.reference import (build_config, read_frame_list, load_frame, frame_id, make_preprocessor,
                              prepare_batch)
from models.pillar_encoder import scatter_point_pillar
from export_onnx import load_model, run_and_capture, pfn_point_features, PfnA, PfnB, PfnC


def frame_dir(out, rel):
    return os.path.join(out, frame_id(rel).replace('/', '_'))


def dump_inputs(d, f):
    os.makedirs(d, exist_ok=True)
    write_bin(os.path.join(d, 'raw_points.bin'), f['raw_points'].astype(np.float32))
    write_bin(os.path.join(d, 'raw_time.bin'), f['time_indice'].astype(np.int32))
    write_bin(os.path.join(d, 'poses_gt.bin'), f['ego_motion_gt'].astype(np.float32))


def np_(t):
    return t.detach().float().cpu().numpy() if t.is_floating_point() else t.detach().cpu().numpy().astype(np.int32)


@torch.no_grad()
def dump_ref(d, model, batch):
    d = os.path.join(d, 'ref')
    os.makedirs(d, exist_ok=True)
    cap = run_and_capture(model, batch)
    res = cap['results']
    pfn = model.pillar_encoder
    raw_points, p2v, coordinates, pillar_mean, time_indice = cap['pfn']['in']

    w = lambda name, x: write_bin(os.path.join(d, name + '.bin'), np_(x) if torch.is_tensor(x) else x)
    w('points', raw_points)
    w('time_indice', time_indice[:, 1].round().int())
    w('coordinates', coordinates[:, 1:].round().int())                  # drop the batch column -> (z, y, x, t)
    w('p2v', p2v)
    w('num_points', torch.bincount(p2v, minlength=coordinates.shape[0]))
    w('pillar_mean', pillar_mean)

    pmax = lambda x: scatter(x, p2v, dim=0, reduce='max')
    a_in = pfn_point_features(pfn, raw_points, p2v, coordinates, pillar_mean, time_indice)
    a_out = PfnA(pfn)(a_in)
    b_in = torch.cat([a_out, pmax(a_out)[p2v]], dim=1)
    b_out = PfnB(pfn)(b_in)
    c_in = torch.cat([b_out, pmax(b_out)[p2v]], dim=1)
    c_out = PfnC(pfn)(c_in)
    feats = pmax(c_out)
    err = (feats - cap['pfn']['out']).abs().max().item()
    assert err == 0, f'pfn split differs from PillarFeatureNet ({err})'
    for name, x in [('pfn_a_in', a_in), ('pfn_a_out', a_out), ('pfn_b_in', b_in), ('pfn_b_out', b_out),
                    ('pfn_c_in', c_in), ('pfn_c_out', c_out), ('pillar_feats', feats)]:
        w(name, x)

    shape = batch['shape'][0]
    bev = scatter_point_pillar(feats, coordinates, 1, shape)            # [1, C, T, H, W]
    w('bev', bev[0].permute(1, 0, 2, 3).contiguous())                  # [T, C, H, W]
    w('fb_seg', res['fb_seg_est'][0])
    w('poses_est', res['ego_motion_est'][0])
    w('mos_label', res['mos_est'].argmax(1))
    w('offset_est', res['offset_est'])
    w('inst_label', res['inst_labels_est'])
    return raw_points.shape[0], coordinates.shape[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/waymo/waymo.yaml')
    ap.add_argument('--ckpt', default='checkpoints/waymo.pth')
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--frame-list', default='data/waymo/selected_frames.txt')
    ap.add_argument('--out', default='pcacc_cpp/data')
    ap.add_argument('--frames', type=int, nargs='+', help='indices in the frame list to convert (default: all)')
    ap.add_argument('--ref', type=int, nargs='*', default=[0, 1, 2],
                    help='indices to also dump reference outputs for (default: 0 1 2)')
    ap.add_argument('--only-ref', action='store_true', help='convert only the --ref frames')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.chdir(REPO)

    frames = read_frame_list(args.frame_list)
    todo = args.ref if args.only_ref else (args.frames if args.frames is not None else range(len(frames)))
    todo = sorted(set(todo) | set(args.ref))
    os.makedirs(args.out, exist_ok=True)

    model = ds = None
    if args.ref:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        cfg, model = load_model(args.config, args.ckpt)
        ds = make_preprocessor(cfg)

    t0 = time.time()
    for i in todo:
        rel = frames[i]
        d = frame_dir(args.out, rel)
        f = load_frame(args.data_root, rel)
        dump_inputs(d, f)
        msg = f'[{i:3d}] {frame_id(rel)}  raw {f["raw_points"].shape[0]}'
        if i in args.ref:
            n, m = dump_ref(d, model, prepare_batch(ds, f))
            msg += f'  ref: N {n}  M {m}'
        print(msg, flush=True)

    with open(os.path.join(args.out, 'frames.txt'), 'w') as fl:
        for i in range(len(frames)):
            if os.path.isdir(frame_dir(args.out, frames[i])):
                fl.write(os.path.basename(frame_dir(args.out, frames[i])) + '\n')
    print(f'{len(todo)} frames -> {args.out} in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
