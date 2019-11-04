"This scripts load feature spaces and prepares it for encoding model"

import pickle
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable

from util.util import *
from featureprep.conv_autoencoder import Autoencoder, preprocess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_features(subj, model, layer=None, dim=None, dataset="", br_subset_idx=None, indoor_only=False):
    print("Getting features for {}{}, for subject {}".format(model, layer, subj))

    with open("../BOLD5000/CSI0{}_stim_lists.txt".format(subj)) as f:
        sl = f.readlines()
    stim_list = [item.strip("\n") for item in sl]

    imgnet_idx, imgnet_cats = extract_dataset_index(sl, dataset="imagenet", rep=False)
    scene_idx, scene_cats = extract_dataset_index(sl, dataset="SUN", rep=False)
    COCO_idx, COCO_cats = extract_dataset_index(sl, dataset="COCO", rep=False)

    # Load features list generated with the whole brain data. This dictionary includes: image names, valence responses,
    # reaction time, session number, etc.
    with open("{}/CSI{}_events.json".format(cortical_dir, subj)) as f:
        events = json.load(f)

    # events also has a stim list, it is same as the "stim_lists.txt"; but repetition is not indicated in the file name.

    if indoor_only:
        with open("../BOLD5000_Stimuli/scene_indoor_cats.txt") as f:
            lst = f.readlines()
        indoor_scene_lst = [item.strip("\n") for item in lst]
        indoor_scene_idx = [
            i for i, s in enumerate(scene_cats) if s in indoor_scene_lst
        ]
        br_subset_idx = np.array(scene_idx)[indoor_scene_idx]
        stim_list = np.array(stim_list)[br_subset_idx]

    if (
            dataset is not ""
    ):  # only an argument for features spaces that applies to all
        if dataset == "ImageNet":
            br_subset_idx = imgnet_idx
            stim_list = np.array(stim_list)[imgnet_idx]
        elif dataset == "COCO":
            br_subset_idx = COCO_idx
            stim_list = np.array(stim_list)[COCO_idx]
        elif dataset == "SUN":
            br_subset_idx = scene_idx
            stim_list = np.array(stim_list)[scene_idx]


    if "convnet" in model or "scenenet" in model:

        # Load order of image features output from pre-trai[ned convnet or scenenet
        # (All layers has the same image order)
        image_order = pickle.load(
            open("../outputs/convnet_features/convnet_image_orders_fc7.p", "rb")
        )
        image_names = [im.split("/")[-1] for im in image_order]

        # Load Image Features
        if model == "convnet":
            if "conv" in layer:
                feature_path = "../outputs/convnet_features/vgg19_avgpool_{}.npy".format(
                    layer
                )
            else:
                feature_path = "../outputs/convnet_features/vgg19__{}.npy".format(layer)
        # elif model == 'convnet_pca':
        #     if 'conv' in layer:
        #         feature_path = glob("../outputs/convnet_features/*eval_pca_black_{}.npy".format(layer))[0]
        #     else:
        #         feature_path = glob("../outptus/convnet_features/*eval_v2_{}*.npy".format(layer))[0]
        elif model == "scenenet":
            feature_path = "../outputs/scenenet_features/avgpool_{}.npy".format(layer)
        else:
            print("model is undefined: " + model)

        feature_mat = np.load(feature_path)
        assert len(image_order) == feature_mat.shape[0]

        featmat = []
        for img_name in stim_list:
            if "rep_" in img_name:
                continue  # repeated images are NOT included in the training and testing sets
            # print(img_name)
            feature_index = image_names.index(img_name)
            featmat.append(feature_mat[feature_index, :])
        featmat = np.array(featmat)

        if br_subset_idx is None:
            br_subset_idx = get_nonrep_index(stim_list)

    elif "taskrepr" in model:
        # latent space in taskonomy, model should be in the format of "taskrep_X", e.g. taskrep_curvature
        task = "_".join(model.split("_")[1:])
        repr_dir = "../genStimuli/{}/".format(task)
        if indoor_only:
            task += "_indoor"

        featmat = []
        for img_name in stim_list:
            if "rep_" in img_name:
                # print(img_name)
                continue
            npyfname = img_name.split(".")[0] + ".npy"
            repr = np.load(repr_dir + npyfname).flatten()
            featmat.append(repr)
        featmat = np.array(featmat)

        if br_subset_idx is None:
            br_subset_idx = get_nonrep_index(stim_list)
        print(featmat.shape[0])
        print(len(br_subset_idx))
        assert featmat.shape[0] == len(br_subset_idx)

    elif model == "pic2vec":
        # only using ImageNet
        from gensim.models import KeyedVectors

        wv_model = KeyedVectors.load(
            "../outputs/models/pix2vec_{}.model".format(dim), mmap="r"
        )
        pix2vec = wv_model.vectors
        wv_words = list(wv_model.vocab)
        br_subset_idx, wv_idx = find_overlap(
            imgnet_cats, wv_words, imgnet_idx, unique=True
        )
        assert len(br_subset_idx) == len(wv_idx)
        featmat = pix2vec[wv_idx, :]

    elif model == "fasttext":
        # import gensim.downloader as api
        # model = api.load('fasttext-wiki-news-subwords-300')
        from gensim.models import KeyedVectors

        ft_model = KeyedVectors.load("../features/fasttext.model", mmap="r")
        if dataset == "SUN":
            cats = scene_cats
            idxes = scene_idx
        elif dataset == "ImageNet":
            cats = imgnet_cats
            idxes = imgnet_idx
        elif dataset == "":
            cats = imgnet_cats + scene_cats
            idxes = imgnet_idx + scene_idx

        featmat, br_subset_idx = [], []
        for i, c in enumerate(cats):
            word = c.split(".")[0]
            if "_" in word:
                word = (
                    re.sub(r"(.)([A-Z])", r"\1-\2", word).replace("_", "-").lower()
                )  # convert phrases to "X-Y"
            try:
                featmat.append(ft_model[word])
                br_subset_idx.append(idxes[i])
            except KeyError:
                if "-" in word:
                    word = word.split("-")[0]  # try to find only "X"
                    try:
                        featmat.append(ft_model[word])
                        br_subset_idx.append(idxes[i])
                    except KeyError:
                        continue
                else:
                    continue

        featmat = np.array(featmat)
        assert featmat.shape[0] == len(br_subset_idx)

    elif model == "response":
        featmat = np.array(events["valence"]).astype(np.float)
        featmat = featmat.reshape(len(featmat), 1)  # make it 2 dimensional

    elif model == "RT":
        featmat = np.array(events["RT"]).astype(np.float)
        featmat = featmat.reshape(len(featmat), 1)  # make it 2 dimensional

    elif model == "surface_normal_latent":
        sf_dir = "../genStimuli/rgb2sfnorm/"
        # load the pre-trained weights
        model_file = "../outputs/models/conv_autoencoder.pth"

        sf_model = Autoencoder()
        checkpoint = torch.load(model_file)
        sf_model.load_state_dict(checkpoint)
        sf_model.to(device)
        for param in sf_model.parameters():
            param.requires_grad = False
        sf_model.eval()

        featmat = []
        for img_name in stim_list:
            if "rep_" in img_name:
                continue
            img = Image.open(sf_dir + img_name)
            inputs = Variable(preprocess(img).unsqueeze_(0)).to(device)
            feat = sf_model(inputs)[1]
            featmat.append(feat.cpu().numpy())
        featmat = np.squeeze(np.array(featmat))

        if br_subset_idx is None:
            br_subset_idx = get_nonrep_index(stim_list)

        assert featmat.shape[0] == len(br_subset_idx)

    elif model == "surface_normal_subsample":
        sf_dir = "../genStimuli/rgb2sfnorm/"

        featmat = []
        for img_name in stim_list:
            if "rep_" in img_name:
                continue
            img = Image.open(sf_dir + img_name)
            inputs = Variable(preprocess(img).unsqueeze_(0))
            k = pool_size(inputs.data, 30000, adaptive=True)
            sub_sf = (
                nn.functional.adaptive_avg_pool2d(inputs.data, (k, k))
                    .cpu()
                    .flatten()
                    .numpy()
            )
            featmat.append(sub_sf)

        featmat = np.squeeze(np.array(featmat))

        if br_subset_idx is None:
            br_subset_idx = get_nonrep_index(stim_list)

        assert featmat.shape[0] == len(br_subset_idx)

    else:
        raise NameError("Model not found.")

    return featmat, br_subset_idx
