taskrepr_features = [
    "autoencoder",
    "denoise",
    "colorization",
    "curvature",
    "edge2d",
    "edge3d",
    "keypoint2d",
    "keypoint3d",
    "segment2d",
    "segment25d",
    "segmentsemantic",
    "rgb2depth",
    "rgb2mist",
    "reshade",
    "rgb2sfnorm",
    "room_layout",
    "vanishing_point",
    "class_1000",
    "class_places",
    "jigsaw",
    "inpainting_whole",
]

conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
scenenet_layers = ["conv1", "block1", "block2", "block3", "block4", "avgpool", "fc"]
sf_methods = ["latent", "subsample"]

model_features = dict()
model_features["taskrepr"] = taskrepr_features
model_features["convnet"] = conv_layers
model_features["scenenet"] = scenenet_layers
model_features["surfaceNormal"] = sf_methods
model_features["pic2vec"] = ["8", "50", "200"]

task_label = {
    "class_1000": "Object Class",
    "segment25d": "2.5D Segm.",
    "room_layout": "Layout",
    "rgb2sfnorm": "Normals",
    "rgb2depth": "Depth",
    "rgb2mist": "Distance",
    "reshade": "Reshading",
    "keypoint3d": "3D Keypoint",
    "keypoint2d": "2D Keypoint",
    "autoencoder": "Autoencoding",
    "colorization": "Color",
    "edge3d": "Occlusion Edges",
    "edge2d": "2D Edges",
    "denoise": "Denoising",
    "curvature": "Curvature",
    "class_places": "Scene Class",
    "vanishing_point": "Vanishing Pts.",
    "segmentsemantic": "Semantic Segm.",
    "segment2d": "2D Segm.",
    "jigsaw": "Jigsaw",
    "inpainting_whole" :"Inpainting",
    # "conv1": "Conv1",
    # "conv2": "Conv2",
    # "conv3": "Conv3",
    # "conv4": "Conv4",
    # "conv5": "Conv5",
    # "fc6": "FC6",
    # "fc7": "FC7",
}

ROIS = ["OPA", "PPA", "LOC", "EarlyVis", "RSC"]
SIDE = ["LH", "RH"]


ROI_labels = [
    str(SIDE[j] + ROIS[i]) for j in range(len(SIDE)) for i in range(len(ROIS))
]