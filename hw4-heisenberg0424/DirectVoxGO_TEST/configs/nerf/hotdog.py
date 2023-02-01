_base_ = '../default.py'

expname = 'dvgo_hotdog'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/home/heisenberg/NTU_EE_2022/DLCV/hw4-heisenberg0424/hw4_data/hotdog',
    dataset_type='blender',
    white_bkgd=True,
)

