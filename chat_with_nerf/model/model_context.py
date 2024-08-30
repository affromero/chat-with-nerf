import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from attrs import define


from chat_with_nerf import logger
from chat_with_nerf.model.scene_config import SceneConfig
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.captioner import (  # Blip2Captioner,
    BaseCaptioner,
)
from chat_with_nerf.visual_grounder.picture_taker import (
    PictureTaker,
    PictureTakerFactory,
)

from chat_with_nerf.settings import Settings


@define
class ModelContext:
    scene_configs: dict[str, SceneConfig]
    picture_takers: dict[str, PictureTaker]
    captioner: BaseCaptioner


class ModelContextManager:
    model_context: Optional[ModelContext] = None

    @classmethod
    def get_model_context(cls, scene_name) -> ModelContext:
        return ModelContextManager.initialize_model_context(scene_name)

    @classmethod
    def get_model_no_gpt_context(cls, scene_name) -> ModelContext:
        if Settings.IS_EVALUATION:
            return ModelContextManager.initialize_model_no_gpt_context(scene_name)
        elif cls.model_context is None:
            cls.model_context = ModelContextManager.initialize_model_no_gpt_context(
                scene_name
            )
        return cls.model_context

    @classmethod
    def get_model_no_visual_feedback_context(cls, scene_name) -> ModelContext:
        if Settings.IS_EVALUATION:
            return ModelContextManager.initialize_model_no_visual_feedback_context(
                scene_name
            )
        elif cls.model_context is None:
            cls.model_context = (
                ModelContextManager.initialize_model_no_visual_feedback_context(
                    scene_name
                )
            )
        return cls.model_context

    @classmethod
    def get_model_context_with_gpt(cls, scene_name: str) -> ModelContext:
        if Settings.IS_EVALUATION:
            return (
                ModelContextManager.initialize_model_no_visual_feedback_openscene_context(scene_name)
            )
        elif cls.model_context is None:
            cls.model_context = (
                ModelContextManager.initialize_model_no_visual_feedback_openscene_context(scene_name)
            )
        return cls.model_context

    @classmethod
    def initialize_model_no_visual_feedback_openscene_context(
        cls, scene_name: str
    ) -> ModelContext:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.append(project_root)
        logger.info("Search for all Scenes and Set the current Scene")
        data_path = Path(Settings.data_path) / scene_name
        scene_configs = ModelContextManager.search_scenes(data_path)
        picture_taker_dict = (
            PictureTakerFactory.get_picture_takers_no_visual_feedback_openscene(
                scene_configs
            )
        )
        return ModelContext(scene_configs, picture_taker_dict, None)

    @staticmethod
    def initialize_model_no_gpt_context(scene_name: str) -> ModelContext:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.append(project_root)
        logger.info("Search for all Scenes and Set the current Scene")
        scene_configs = ModelContextManager.search_scenes(
            Path(Settings.data_path) / scene_name
        )
        picture_taker_dict = PictureTakerFactory.get_picture_takers_no_gpt(
            scene_configs
        )
        # picture_taker_dict = PictureTakerFactory.get_picture_takers_no_visual_feedback_openscene(
        #     scene_configs
        # )
        return ModelContext(scene_configs, picture_taker_dict, None)

    @staticmethod
    def initialize_model_no_visual_feedback_context(scene_name: str) -> ModelContext:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.append(project_root)
        logger.info("Search for all Scenes and Set the current Scene")
        scene_configs = ModelContextManager.search_scenes(
            Settings.data_path, scene_name
        )
        picture_taker_dict = PictureTakerFactory.get_picture_takers_no_visual_feedback(
            scene_configs
        )
        return ModelContext(scene_configs, picture_taker_dict, None)

    @staticmethod
    def initialize_model_context(scene_name: str) -> ModelContext:
        # Get the absolute path of the project's root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Add the project's root directory to sys.path
        sys.path.append(project_root)

        logger.info("Search for all Scenes and Set the current Scene")
        scene_configs = ModelContextManager.search_scenes(
            Settings.data_path, scene_name
        )

        logger.info("Initialize picture_taker for all scenes")
        picture_taker_dict = PictureTakerFactory.get_picture_takers(scene_configs)

        return ModelContext(scene_configs, picture_taker_dict, None)

    @staticmethod
    def search_scenes(path: str | Path) -> dict[str, SceneConfig]:
        scenes = {}
        path = Path(path).resolve()
        sc_name = os.path.basename(path)
        logger.info(f"path: {path}")
        scene_path = (Path(path) / sc_name).with_suffix(".yaml")
        logger.info(f"scene_path: {scene_path}")
        with open(scene_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        replacements = {
            "/workspace/chat-with-nerf-dev/chat-with-nerf/data": "data/lerf_data_experiments/",
            "/workspace/chat-with-nerf-eval/data/scannet": "data/scannet",
            "/workspace/openscene_data": "data/openscene_data",
        }

        for key, value in replacements.items():
            for k, v in data.items():
                if isinstance(v, str):
                    data[k] = v.replace(key, value)
        logger.info(f"scene data: {data}")
        scene = SceneConfig(
            sc_name,
            data["load_lerf_config"],
            data["load_embedding"],
            data["camera_path"],
            data["nerf_exported_mesh_path"],
            data["load_openscene"],
            data["load_mesh"],
            data["load_metadata"],
        )
        scenes[sc_name] = scene
        return scenes