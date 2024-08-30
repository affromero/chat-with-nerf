from abc import abstractmethod

import torch
from attrs import define
from PIL import Image
from chat_with_nerf.visual_grounder.image_ref import ImageRef


@define
class BaseCaptioner:
    model: torch.nn.Module
    """Base model for image processing."""
    vis_processors: dict
    """Preprocessors for visual inputs."""

    def process_image(self, image_path: str) -> torch.Tensor:
        """Processes an image and returns it as a tensor."""
        raw_image = Image.open(image_path).convert("RGB")
        return self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.model.device)

    @abstractmethod
    def caption(self, positive_words: str, imagerefs: list[ImageRef]) -> dict[str, str]:
        """Generates captions for the images."""
        pass
