_RGBD_SOD_ROOT = "<rgbdsod root>"
_RGBT_SOD_ROOT = "<rgbtsod root>"

# RGB-D SOD
LFSD = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/LFSD/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/LFSD/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/LFSD/Mask", suffix=".png"),
)
NLPR_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/NLPR_FULL/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/NLPR_FULL/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/NLPR_FULL/Mask", suffix=".png"),
    index_file="datasets/nlpr_train_jw_name_list.lst",
)
NJUD_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/NJUD_FULL/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/NJUD_FULL/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/NJUD_FULL/Mask", suffix=".png"),
    index_file="datasets/njud_train_jw_name_list.lst",
)
NLPR_TE = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/NLPR_FULL/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/NLPR_FULL/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/NLPR_FULL/Mask", suffix=".png"),
    index_file="datasets/nlpr_test_jw_name_list.lst",
)
NJUD_TE = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/NJUD_FULL/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/NJUD_FULL/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/NJUD_FULL/Mask", suffix=".png"),
    index_file="datasets/njud_test_jw_name_list.lst",
)
RGBD135 = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/RGBD135/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/RGBD135/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/RGBD135/Mask", suffix=".png"),
)
SIP = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/SIP/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/SIP/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/SIP/Mask", suffix=".png"),
)
SSD = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/SSD/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/SSD/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/SSD/Mask", suffix=".png"),
)
STEREO1000 = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/STEREO1000/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/STEREO1000/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/STEREO1000/Mask", suffix=".png"),
)
DUTRGBD_TE = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/DUTLF-Depth/Test/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/DUTLF-Depth/Test/depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/DUTLF-Depth/Test/GT", suffix=".png"),
)
DUTRGBD_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/DUTLF-Depth/Train/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/DUTLF-Depth/Train/depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/DUTLF-Depth/Train/GT", suffix=".png"),
)

REDWEBS_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/trainset/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/trainset/depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/trainset/GT", suffix=".png"),
)
REDWEBS_TE = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/testset/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/testset/depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/testset/GT", suffix=".png"),
)

COME_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TR/imgs_right", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TR/depths", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TR/gt_right", suffix=".png"),
)
COME_TE_E = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-E/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-E/depths", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-E/GT", suffix=".png"),
)
COME_TE_H = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-H/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-H/depths", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-H/GT", suffix=".png"),
)

# RGB-T SOD
VT5000TR = dict(
    image=dict(path=f"{_RGBT_SOD_ROOT}/VT5000/Train/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBT_SOD_ROOT}/VT5000/Train/T", suffix=".jpg"),
    mask=dict(path=f"{_RGBT_SOD_ROOT}/VT5000/Train/GT", suffix=".png"),
)
VT5000TE = dict(
    image=dict(path=f"{_RGBT_SOD_ROOT}/VT5000/Test/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBT_SOD_ROOT}/VT5000/Test/T", suffix=".jpg"),
    mask=dict(path=f"{_RGBT_SOD_ROOT}/VT5000/Test/GT", suffix=".png"),
)
VT1000 = dict(
    image=dict(path=f"{_RGBT_SOD_ROOT}/VT1000/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBT_SOD_ROOT}/VT1000/T", suffix=".bmp"),
    mask=dict(path=f"{_RGBT_SOD_ROOT}/VT1000/GT", suffix=".jpg"),
)
VT821 = dict(
    image=dict(path=f"{_RGBT_SOD_ROOT}/VT821/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBT_SOD_ROOT}/VT821/thermal", suffix=".jpg"),
    mask=dict(path=f"{_RGBT_SOD_ROOT}/VT821/GT", suffix=".jpg"),
)
