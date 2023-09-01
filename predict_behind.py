import os
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_behind
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format
from elevation_estimate.estimate_wild_imgs import estimate_elev
import matplotlib.pyplot as plt

def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256

def predict_behind_stage(model, device, exp_dir, input_im, scale, ddim_steps, n_samples=3):
    behind_dir = os.path.join(exp_dir, "behind")
    os.makedirs(behind_dir, exist_ok=True)
    
    output_ims = predict_behind(model, input_im, save_path=behind_dir, device=device, 
                                n_samples=n_samples, ddim_steps=ddim_steps, scale=scale)
    return output_ims


def predict_behindview(shape_dir, args):
    device = f"cuda:{args.gpu_idx}"

    # initialize the zero123 model
    print("half precision: ", args.half_precision)
    models = init_model(device, 'zero123-ori.ckpt', half_precision=args.half_precision)
    model_zero123 = models["turncam"]

    # initialize the Segment Anything model
    predictor = sam_init(args.gpu_idx)
    input_raw = Image.open(args.img_path)

    # preprocess the input image
    input_256 = preprocess(predictor, input_raw)

    n_samples = args.n_samples
    output_ims = predict_behind_stage(model_zero123, device, shape_dir,   
                                  input_256, scale=3.0, ddim_steps=50, n_samples=n_samples)

    print("draw imgs")
    # draw images in plt
    plt.figure(figsize=(16, 4))
    plt.subplot(1, n_samples+1, 1)
    plt.imshow(input_raw)
    plt.title("Input Image")
    plt.axis('off')

    for i, im in enumerate(output_ims):
        plt.subplot(1, n_samples+1, i+2)
        plt.imshow(im)
        plt.title(f"Output Image {i+1}")
        plt.axis('off')
    plt.savefig("result")
    plt.close()
    print("Done")

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

    shape_id = args.img_path.split('/')[-1].split('.')[0]
    shape_dir = f"./exp/{shape_id}"
    os.makedirs(shape_dir, exist_ok=True)
    
    ### predict behind view
    predict_behindview(shape_dir, args)

    
