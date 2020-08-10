# -*- coding:utf-8 -*-
###
# File: ji.py
# Created Date: Monday, August 10th 2020, 7:54:12 pm
# Author: yusnows
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 yusnows
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
from __future__ import print_function

import logging as log
import json
import os
import sys
import torch
from collections import defaultdict
from io import StringIO
import cv2
import numpy as np
from mmcv import Config
from mmdet.models import build_detector

input_w, input_h, input_c, input_n = (512, 768, 3, 1)

label_id_map = {
    1: "rat"
}


def init():
    """Initialize model

    Returns: model

    """
    model_path = "/usr/local/ev_sdk/model/latest.pth"
    if not os.path.isfile(model_path):
        log.error(f'{model_path} does not exist')
        return None
    log.info('Loading model...')
    cfg = Config.fromfile("/project/train/src_repo/detect/configs/ttfnet/ttfnet_mobilev2_rat.py")
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    return model


def process_image(net, input_image, thresh):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        thresh: thresh value

    Returns: process result

    """

    # ------------------------------- Prepare input -------------------------------------
    if not net or input_image is None:
        log.error('Invalid input args')
        return None
    ih, iw, _ = input_image.shape
    if ih != input_h or iw != input_w:
        input_image = cv2.resize(input_image, (input_w, input_h))
    h, w, _ = input_image.shape
    im_meta = {}
    im_meta["scale_factor"] = np.array([w/iw, h/ih, w/iw, h/ih])
    im_meta["pad_shape"] = input_image.shape
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.transpose(0, 3, 1, 2)
    det_reses = net.simple_test(input_image, im_meta, rescale=True)
    detect_objs = []
    for idx in range(len(det_reses)):
        det_res = det_reses[idx]
        if det_res.shape[0] == 0:
            continue
        detect_objs.append({
            "xmin": int(det_res[0]),
            "ymin": int(det_res[1]),
            "xmax": int(det_res[2]),
            "ymax": int(det_res[3]),
            "name": label_id_map[idx]
        })
    return json.dumps(detect_objs)
