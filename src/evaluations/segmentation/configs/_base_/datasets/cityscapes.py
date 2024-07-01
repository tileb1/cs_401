_base_ = ["../custom_import.py"]
# dataset settings
dataset_type = "TarCityscapesDataset"
# data_root = "./data/cityscapes"
data_root = "/scratch/project_465000727/datasets/cityscapes.tar"

has_background = False

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="ToRGB"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="ImageToTensorV2", keys=["img"]),
            dict(
                type="Collect",
                keys=["img"],
                meta_keys=["ori_shape", "img_shape", "pad_shape", "flip", "img_info"],
            ),
        ],
    ),
]
data = dict(
    test=dict(
        type=dataset_type,
        untar_path="/dev/shm",
        local_rank=0,
        data_root=data_root,
        img_dir="leftImg8bit/val",
        ann_dir="gtFine/val",
        pipeline=test_pipeline,
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
