import os
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_behind
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format
from elevation_estimate.estimate_wild_imgs import estimate_elev

import matplotlib.pyplot as plt
from bottle import route, run, template, request, static_file, url, get, post, response, error, abort, redirect, os
import datetime
import uuid
import numpy as np
import cv2 

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')

def image_preprocess_crop_info(input_image, lower_contrast=True, rescale=True):
    # (1024, 626)
    image_arr = np.array(input_image)
    in_w, in_h = image_arr.shape[:2]

    if lower_contrast:
        alpha = 0.8  # Contrast control (1.0-3.0)
        beta =  0   # Brightness control (0-100)
        # Apply the contrast adjustment
        image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
        image_arr[image_arr[...,-1]>200, -1] = 255

    ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    # 0 0 626 1024
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    ratio = 0.75
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len//2
    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    # padded: (1365, 1365, 4)
    rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)

    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
    return Image.fromarray((rgb * 255).astype(np.uint8)), (x, y, w, h, side_len, side_len)

def predict_behind_stage(model, device, exp_dir, input_im, scale, ddim_steps, n_samples=3):
    behind_dir = os.path.join(exp_dir, "behind")
    os.makedirs(behind_dir, exist_ok=True)
    
    output_ims = predict_behind(model, input_im, save_path=behind_dir, device=device, 
                                n_samples=n_samples, ddim_steps=ddim_steps, scale=scale)
    return output_ims

def remove_padding(output_im, crop_info):
    x, y, w, h, padded_w, padded_h = crop_info
    resized_output = output_im.resize((padded_w, padded_h), Image.LANCZOS)
    center = padded_w // 2
    cropped_output = resized_output.crop((center-w//2, center-h//2, center+w//2, center+h//2))
    # cropped_output = resized_output.crop((x, y, x+w, y+h))
    return cropped_output

@get('/')
def upload():
    return '''
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="submit" value="Upload"></br>
            <input type="file" name="image"></br>
        </form>
    '''
@route('/assets/<filepath:path>', name='assets')
def server_static(filepath):
    return static_file(filepath, root=base_shape_dir)

@route('/upload', method='POST')
def do_upload():
    upload = request.files.get('image', '')
    if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 'File extension not allowed!'
    
    now = datetime.datetime.now(JST)
    d = '{:%Y%m%d%H%M%S}'.format(now)
    short_id = str(uuid.uuid4())[:8]
    dir_name = f"{d}_{short_id}"
    shape_dir = os.path.join(base_shape_dir, dir_name)
    os.makedirs(shape_dir, exist_ok=True)
    print(shape_dir)

    filename = upload.filename.lower()
    # root, ext = os.path.splitext(filename)
    save_path = os.path.join(shape_dir, filename)
    upload.save(save_path, overwrite=True)

    device = f"cuda:{args.gpu_idx}"

    input_raw = Image.open(save_path)
    n_samples = args.n_samples
    # preprocess the input image
    # input_256 = image_preprocess_nosave(input_raw, lower_contrast=False, rescale=True)
    input_256, crop_info = image_preprocess_crop_info(input_raw, lower_contrast=False, rescale=True)
    output_ims = predict_behind_stage(model_zero123, device, shape_dir,
                                    input_256, scale=3.0, ddim_steps=50, n_samples=n_samples)
    
    saved_paths = []

    for i, im in enumerate(output_ims):
        im = remove_padding(im, crop_info)
        output_ims[i] = im

    for i, im in enumerate(output_ims):
        saved_path = os.path.join(shape_dir, f"output_{i}.png")
        im.save(saved_path)
        saved_paths.append(saved_path)
        print(f"Output image {i} saved to {saved_path}")
    
    # return {"status": "success", "saved_paths": saved_paths}
    body = {"status": 0, "data": f"http://20.168.237.190:8000/assets/{dir_name}/result.png"}
    return body

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_path', type=str, default="./demo/demo_examples/pikachu.png", help='Path to the input image')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision')
    parser.add_argument('--mesh_resolution', type=int, default=256, help='Mesh resolution')
    parser.add_argument('--output_format', type=str, default=".ply", help='Output format: .ply, .obj, .glb')
    parser.add_argument('--n_samples', type=int, default=3, help='Number of samples')
    args = parser.parse_args()

    assert(torch.cuda.is_available())

    base_shape_dir=f'./exp/web'
    os.makedirs(base_shape_dir, exist_ok=True)
    device = f"cuda:{args.gpu_idx}"

    models = init_model(device, 'zero123-ori.ckpt', half_precision=args.half_precision)
    model_zero123 = models["turncam"]

    print("run")
    run(host="0.0.0.0", port=8000, debug=True)

    # shape_id = args.img_path.split('/')[-1].split('.')[0]
    # shape_dir = f"./exp/{shape_id}"
    # os.makedirs(shape_dir, exist_ok=True)
    
    ### predict behind view
    # predict_behindview(shape_dir, args)

    
