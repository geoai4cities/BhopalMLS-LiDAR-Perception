_base_ = ["../bhopal/utonia_140gb_21class_linear_probe_9ch.py"]

H5_PATH = "data/Bhopal/bhopal_scene_v10_train_1234_val_56_test_78.h5"

# V10 moving-class fold (12c-equivalent targets written in parentheses):
#   62 (Moving Vehicle)       -> 12 4-Wheeler (same group as DynCar/DynTruck)
#   63 (Moving Humans)        -> 16 Human
#   64 (Moving Female)        -> 16 Human
#   65 (Moving Male)          -> 16 Human
#   66 (Moving Animal)        -> IGNORED (no group contains it), mirroring v9 21c policy
#                                which already ignores static Animal 47, Dog 48, Cow 49.
#   67 (Mobike Rider Moving)  -> 10 2-Wheeler (same group as static MobikeRider 33)
class_groups = [
    [0, 1, 2],
    [3],
    [4, 5, 6, 7],
    [8, 9, 11, 12],
    [10],
    [13],
    [14],
    [15, 17, 19],
    [16],
    [18],
    [22, 23, 32, 33, 34, 67],
    [25, 27, 28, 30],
    [20, 21, 26, 29, 31, 62],
    [24],
    [35, 36, 37],
    [38, 39, 40, 41, 42],
    [43, 44, 45, 46, 63, 64, 65],
    [53],
    [52],
    [57],
    [50, 51, 54, 55, 56, 58, 59, 60, 61],
]

data = dict(
    train=dict(h5_path=H5_PATH, class_groups=class_groups),
    val=dict(h5_path=H5_PATH, class_groups=class_groups),
    test=dict(h5_path=H5_PATH, class_groups=class_groups),
)
