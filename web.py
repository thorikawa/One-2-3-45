import os
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format
from elevation_estimate.estimate_wild_imgs import estimate_elev
from bottle import route, run, template, request, static_file, url, get, post, response, error, abort, redirect, os
import datetime
import uuid

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')

def stage1_run(model, device, exp_dir,
               input_im, scale, ddim_steps):
    # folder to save the stage 1 images
    stage1_dir = os.path.join(exp_dir, "stage1_8")
    os.makedirs(stage1_dir, exist_ok=True)

    # stage 1: generate 4 views at the same elevation as the input
    output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)
    
    # stage 2 for the first image
    # infer 4 nearby views for an image to estimate the polar angle of the input
    stage2_steps = 50 # ddim_steps
    zero123_infer(model, exp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)
    # estimate the camera pose (elevation) of the input image.
    try:
        polar_angle = estimate_elev(exp_dir)
    except:
        print("Failed to estimate polar angle")
        polar_angle = 90
    print("Estimated polar angle:", polar_angle)
    gen_poses(exp_dir, polar_angle)

    # stage 1: generate another 4 views at a different elevation
    if polar_angle <= 75:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4,8)), device=device, ddim_steps=ddim_steps, scale=scale)
    else:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(8,12)), device=device, ddim_steps=ddim_steps, scale=scale)
    torch.cuda.empty_cache()
    return 90-polar_angle, output_ims+output_ims_2
    
def stage2_run(model, device, exp_dir,
               elev, scale, stage2_steps=50):
    # stage 2 for the remaining 7 images, generate 7*4=28 views
    if 90-elev <= 75:
        zero123_infer(model, exp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        zero123_infer(model, exp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)

def reconstruct(exp_dir, output_format=".ply", device_idx=0, resolution=256):
    exp_dir = os.path.abspath(exp_dir)
    main_dir_path = os.path.abspath(os.path.dirname("./"))
    os.chdir('reconstruction/')

    bash_script = f'CUDA_VISIBLE_DEVICES={device_idx} python exp_runner_generic_blender_val.py \
                    --specific_dataset_name {exp_dir} \
                    --mode export_mesh \
                    --conf confs/one2345_lod0_val_demo.conf \
                    --resolution {resolution}'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(exp_dir, f"mesh.ply")
    if output_format == ".ply":
        return ply_path
    if output_format not in [".obj", ".glb"]:
        print("Invalid output format, must be one of .ply, .obj, .glb")
        return ply_path
    return convert_mesh_format(exp_dir, output_format=output_format)

@get('/')
def upload():
    return '''
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="submit" value="Upload"></br>
            <input type="file" name="upload"></br>
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
    root, ext = os.path.splitext(filename)
    save_path = os.path.join(shape_dir, filename)
    upload.save(save_path, overwrite=True)

    device = f"cuda:{args.gpu_idx}"

    # initialize the Segment Anything model
    input_raw = Image.open(save_path)

    # preprocess the input image
    input_256 = image_preprocess_nosave(input_raw, lower_contrast=False, rescale=True)

    # generate multi-view images in two stages with Zero123.
    # first stage: generate N=8 views cover 360 degree of the input shape.
    print("===step1===")
    elev, stage1_imgs = stage1_run(model_zero123, device, shape_dir, input_256, scale=3, ddim_steps=75)
    # second stage: 4 local views for each of the first-stage view, resulting in N*4=32 source view images.
    print("===step2===")
    stage2_run(model_zero123, device, shape_dir, elev, scale=3, stage2_steps=50)

    # utilize cost volume-based 3D reconstruction to generate textured 3D mesh
    print("===step3===")
    mesh_path = reconstruct(shape_dir, output_format=args.output_format, device_idx=args.gpu_idx, resolution=args.mesh_resolution)
    print("Mesh saved to:", mesh_path)
    body = {"status": 0, "data": f"http://20.168.237.190:8000/assets/{dir_name}/mesh.obj"}
    return body

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision')
    parser.add_argument('--mesh_resolution', type=int, default=256, help='Mesh resolution')
    parser.add_argument('--output_format', type=str, default=".obj", help='Output format: .ply, .obj, .glb')

    args = parser.parse_args()

    assert(torch.cuda.is_available())

    base_shape_dir = f"./exp/web"
    os.makedirs(base_shape_dir, exist_ok=True)

    device = f"cuda:{args.gpu_idx}"

    # initialize the zero123 model
    models = init_model(device, 'zero123-xl.ckpt', half_precision=args.half_precision)
    model_zero123 = models["turncam"]

    print("run")
    run(host="0.0.0.0", port=8000, debug=True)
