from __future__ import division

import os

import click
import cv2
import geopandas as gpd
import numpy as np
import torch
from .darknet import Darknet
from shapely.geometry import box
from torch.autograd import Variable
from .util import write_results


def load_classes(namesfile):
    """

    Args:
        namesfile:

    Returns:

    """
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
    """
    Resize image with unchanged aspect ratio using padding

    Args:
        img:
        inp_dim:

    Returns:

    """

    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network

    Args:
        img:
        inp_dim:

    Returns:

    """

    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def annotate_image(df, img, out_fid):
    """

    Args:
        df:
        img:
        out_fid:

    Returns:

    """

    # build list of colours (assume less than 4 classes!)
    cols = [(255, 0, 0),  # red
            (0, 255, 0),  # green
            (0, 0, 255),  # blue
            (255, 0, 255),  # fuchsia
            (255, 255, 0),  # yellow
            (0, 255, 255)]

    # get all possible categories -> make colour list
    unique_cats = list(set(df['cls']))
    unique_cats.sort()
    cat_cols = {u: cols[x] for x, u in enumerate(unique_cats)}

    for _, row in df.iterrows():
        geo = row.geometry
        cv2.polylines(img, np.int32([geo.exterior.coords[:]]), True, cat_cols[row['cls']], thickness=8)
        cv2.putText(img, row['cls'], (int(geo.bounds[0]), int(geo.bounds[1]) - 10), cv2.FONT_HERSHEY_PLAIN, 4,
                    cat_cols[row['cls']], 3)
    cv2.imwrite(out_fid, img)


def rescale(im_dim_list, output, inp_dim):
    """

    Args:
        im_dim_list:
        output:
        inp_dim:

    Returns:

    """

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    return output


def convert_geometry(detections, input_im, classes, fid_out=None):
    # convert detections to simple GeoDataFrame
    df = []
    fname = os.path.splitext(os.path.split(input_im)[-1])[0]
    for detection in detections.tolist():
        # extract required information (per detection)
        tl = detection[1:3]
        br = detection[3:5]
        geo = box(*tl, *br)
        cls = classes[int(detection[-1])]

        # save into a GeoDataFrame
        df.append(gpd.GeoSeries({'geometry': geo, 'src_img': fname, 'cls': cls}))

    df = gpd.GeoDataFrame(df)

    # save as a GeoJSON
    if fid_out is not None:
        if os.path.exists(fid_out):
            os.remove(fid_out)
        df.to_file(fid_out, driver='GeoJSON')

    return df


def yolo_detect(input_im, out_dir, cfg, names, weights, confidence, nms_thresh, resolution):
    """

    Args:
        input_im:
        out_dir:
        confidence:
        nms_thresh:
        cfg:
        names:
        weights:
        resolution:

    Returns:

    """

    # check if GPU is available
    CUDA = torch.cuda.is_available()

    # load neural network + weights + classes
    model = Darknet(cfg)
    model.load_weights(weights)
    classes = load_classes(names)

    # add resolution information to model object
    model.net_info["height"] = resolution
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # put model on GPU (if available)
    if CUDA:
        model.cuda()

    # set model object to "evaluation mode"
    model.eval()

    # prepare image (into "batch")
    batch, im_data, im_dim = prep_image(input_im, inp_dim)
    im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

    # convert params to GPU (if available)
    if CUDA:
        im_dim = im_dim.cuda()
        batch = batch.cuda()

    # Apply offsets to the result predictions -> tranform the predictions as described in the YOLO paper
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    # get the boxes with object confidence > threshold -> convert to absolute coordinates
    detections = write_results(prediction, confidence, len(classes), nms=True, nms_conf=nms_thresh)

    if CUDA:
        torch.cuda.synchronize()

    # post-process detections -> json + annotated image
    detections = rescale(im_dim, detections, inp_dim)  # rescale output
    geo_fid = os.path.join(out_dir, '{}.geojson'.format(os.path.splitext(os.path.split(input_im)[-1])[0]))
    df = convert_geometry(detections, input_im, classes, geo_fid)
    annotated_fid = os.path.join(out_dir, os.path.split(input_im)[-1])
    annotate_image(df, im_data, annotated_fid)

    torch.cuda.empty_cache()


@click.command()
@click.argument('input_im', type=click.Path(exists=True, dir_okay=False))
@click.argument('out_dir', type=click.Path(exists=False, dir_okay=True))
@click.argument('cfg', type=click.Path(exists=True, dir_okay=False))
@click.argument('names', type=click.Path(exists=True, dir_okay=False))
@click.argument('weights', type=click.Path(exists=True, dir_okay=False))
@click.option('--confidence', type=float, default=0.5, help="minimum object confidence for filtering predictions")
@click.option('--nms_thresh', type=float, default=0.2, help="non-maximum suppression for filtering predictions")
@click.option('--resolution', type=int, default=416,
              help="input resolution of the network (balance accuracy and speed)")
def main(input_im, out_dir, cfg, names, weights, confidence, nms_thresh, resolution):
    yolo_detect(input_im, out_dir, cfg, names, weights, confidence, nms_thresh, resolution)


if __name__ == '__main__':
    main()
