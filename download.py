from huggingface_hub import snapshot_download
snapshot_download("timm/resnet50.a1_in1k", local_dir="./resnet50_weights")
snapshot_download("timm/efficientnet_b2.ra_in1k", local_dir="./efficientnet_weights")