_base_ = ["../bhopal/concerto_140gb_12class_linear_probe_9ch.py"]

H5_PATH = "data/Bhopal/bhopal_scene_v10_train_1234_val_56_test_78.h5"

# V10 moving-class fold (all 6 moving codes 62-67 kept as training targets):
#   62 (Moving Vehicle)       -> 7 Vehicle
#   63 (Moving Humans)        -> 10 Human and Animal
#   64 (Moving Female)        -> 10 Human and Animal
#   65 (Moving Male)          -> 10 Human and Animal
#   66 (Moving Animal)        -> 10 Human and Animal
#   67 (Mobike Rider Moving)  -> 7 Vehicle
class_groups = [
    list(range(0, 4)),
    list(range(4, 8)),
    list(range(8, 13)),
    [13],
    [14],
    list(range(15, 19)),
    [19],
    list(range(20, 35)) + [62, 67],
    list(range(35, 38)),
    list(range(38, 43)),
    list(range(43, 50)) + [63, 64, 65, 66],
    list(range(50, 62)),
]

data = dict(
    train=dict(h5_path=H5_PATH, class_groups=class_groups),
    val=dict(h5_path=H5_PATH, class_groups=class_groups),
    test=dict(h5_path=H5_PATH, class_groups=class_groups),
)
