import os
import torch
from unet import UNet_Transformer
from PIL import Image
import torchvision.transforms as transforms
import datetime


def adapt_legacy_state_dict_keys(state_dict):
    adapted = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("attentions."):
            new_key = new_key.replace("attentions.", "gated_attentions.", 1)
        adapted[new_key] = value
    return adapted


def infer_model_config_from_state_dict(state_dict):
    config = {
        "use_residual": True,
        "use_gated_attention": True,
        "use_spatial_attention": False,
        "use_pde_attention": False,
        "use_aspp": False,
        "use_edge_branch": False,
        "return_edge": False,
    }

    if any(k.startswith("downs.0.conv1") for k in state_dict.keys()):
        config["use_residual"] = True
    elif any(k.startswith("downs.0.0") for k in state_dict.keys()):
        config["use_residual"] = False

    config["use_gated_attention"] = any(
        k.startswith("gated_attentions.") or k.startswith("attentions.")
        for k in state_dict.keys()
    )
    config["use_spatial_attention"] = any(k.startswith("spatial_attentions.") for k in state_dict.keys())
    config["use_pde_attention"] = any(k.startswith("pde_attentions.") for k in state_dict.keys())
    config["use_aspp"] = any(k.startswith("aspp.") for k in state_dict.keys())
    config["use_edge_branch"] = any(k.startswith("edge_branch.") for k in state_dict.keys())
    return config


def build_prediction_output_dir(output_root, model_path):
    os.makedirs(output_root, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    final_output_dir = os.path.join(output_root, f"{model_name}_{now}")
    os.makedirs(final_output_dir, exist_ok=True)
    return final_output_dir


def resolve_image_folder(image_folder):
    if os.path.isdir(image_folder):
        return image_folder

    fallback_dirs = [
        "predictset",
        os.path.join("dataset", "test", "images"),
        os.path.join("dataset", "images"),
    ]
    for candidate in fallback_dirs:
        if os.path.isdir(candidate):
            print(f"Input folder '{image_folder}' not found, fallback to: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Image folder not found: {image_folder}. Please create it or pass an existing folder path."
    )

def predict_folder(image_folder, model_path, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_folder = resolve_image_folder(image_folder)

    if not os.path.exists(model_path):
        alt_model_path = os.path.join("model_result", os.path.basename(model_path))
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    raw_state_dict = torch.load(model_path, map_location=device)
    raw_state_dict = adapt_legacy_state_dict_keys(raw_state_dict)
    model_config = infer_model_config_from_state_dict(raw_state_dict)

    model = UNet_Transformer(in_channels=3, out_channels=1, **model_config).to(device)
    model.load_state_dict(raw_state_dict, strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    final_output_folder = build_prediction_output_dir(output_folder, model_path)
    print(f"Prediction output folder: {final_output_folder}")

    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            continue
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype("uint8") * 255
            pred_img = Image.fromarray(pred_mask)
            pred_img.save(os.path.join(final_output_folder, f"{os.path.splitext(img_name)[0]}_pred.png"))

if __name__ == "__main__":
    predict_folder("predictset", "unet+transformer+Res_20260226_1558.pth    ", "predict_result")