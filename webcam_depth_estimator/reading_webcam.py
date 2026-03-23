import torch
import cv2
import numpy as np
from transformers import AutoModelForDepthEstimation
from PIL import Image
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess(frame: np.array) -> np.array:
    """
    Preprocess an image to be passed to our depth estimator

    Args:
        frame: np.array(input_width, input_height) frame from video
    Returns:
        img_tensor: torch.tensor(1, 3, 518, 518) (B,C,W,H) resized and normalized tensor for depth prediction
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    img_resized = pil_image.resize((518, 518))
    img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img_tensor


def postprocess(depth_data, orig_w, orig_h):
    """
    Postprocess an image to be shown on screen.

    Args:
        depth_data: torch.tensor(1, 3, 518, 518) (B,C,W,H) that has been output by our depth estimator
        orig_w: int that designates the original width of the video
        orig_h: int that designates the original height of the video
    Returns:
        colormap_output: np.array(orig_w, orig_h) colored depthmap in the size ofthe original video
    """
    depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized_normal = cv2.resize(depth_vis, (orig_w, orig_h), fx =0, fy= 0, interpolation= cv2.INTER_LINEAR)
    colormap_output = cv2.applyColorMap(resized_normal, cv2.COLORMAP_INFERNO)
    return colormap_output



def main():
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.to(DEVICE).eval()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_w,orig_h, c = frame.shape
        img_tensor = preprocess(frame)

        with torch.no_grad():
            output = model(pixel_values=img_tensor)
        
        depth = output.predicted_depth.squeeze().cpu().numpy()
        colormap_output = postprocess(depth, orig_w, orig_h)
        
        cv2.imshow("Original", frame)
        cv2.imshow("Depth", colormap_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()