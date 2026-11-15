# глубина по одному кадру: Depth Anything V2 Small.
# сеть выдаёт обратную глубину с точностью до масштаба и сдвига, выравнивание в align.py
import numpy as np
import torch
import torch.nn.functional as F

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class MonoDepth:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf",
                 device="cuda", input_height=266, half=True):
        from transformers import AutoModelForDepthEstimation

        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
        if self.half:
            self.model = self.model.half()
        self.input_height = int(round(input_height / 14)) * 14   # кратно патчу ViT (14)
        self._mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, image):
        # image (H, W, 3) в [0, 1] -> обратная глубина (H, W)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        x = image.to(self.device, dtype=torch.float32).permute(2, 0, 1)[None]
        h, w = x.shape[-2:]
        th = self.input_height
        tw = int(round(w * th / h / 14)) * 14
        x = F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - self._mean) / self._std
        if self.half:
            x = x.half()
        pred = self.model(pixel_values=x).predicted_depth   # (1, th, tw)
        pred = F.interpolate(pred[None].float(), size=(h, w), mode="bilinear", align_corners=False)
        return pred[0, 0]
