import os
from pathlib import Path
from pydantic.dataclasses import dataclass
import tyro

from chat_with_nerf.chat.agent import Agent
from chat_with_nerf.settings import Chat_With_NeRF_Settings
from rich.console import Console
from chat_with_nerf.app import main as run_gradio
from run_colmap import ExtractNeRF

console = Console()

@dataclass
class Config:
    h5_embeddings: str
    """ The path to the h5 embeddings. """

    mesh: str
    """ The path to the mesh. """

    scene_name: str
    """ The name of the scene. """

    cwn_folder: str
    """ 
    The path to the chat with nerf folder. This method likes the files in a very specific structure.
    Where ${cwn_folder}/${scene_name}/scene_name.yaml, and the scene_name.yaml file contains the following:
        load_embedding: ${h5_embeddings}
        load_mesh: ${mesh}
    """

    def __post_init__(self) -> None:
        self.scene_config = f"{self.cwn_folder}/{self.scene_name}/{self.scene_name}.yaml"
        if not Path(self.scene_config).exists():
            console.log(f"Creating scene config for {self.scene_name}")
            Path(self.scene_config).parent.mkdir(parents=True, exist_ok=True)
            self.create_scene_config()
            
    def create_scene_config(self) -> None:
        with open(self.scene_config, "w") as f:
            f.write(f"load_embedding: {self.h5_embeddings}\n")
            f.write(f"load_mesh: {self.mesh}\n")
            f.write(f"load_openscene: \n")
            f.write(f"load_metadata: \n")

    def create_settings(self) -> Chat_With_NeRF_Settings:
        settings = Chat_With_NeRF_Settings()
        settings.data_path = self.cwn_folder
        settings.NERF_DATA_PATH = self.cwn_folder
        settings.USE_OPENSCENE = False
        settings.default_scene = self.scene_name
        settings.MAX_TURNS = 1000
        settings.CLIP_FILTERING_THRESHOLD: float = 21  # range is (0, 100)
        settings.INITIAL_MSG_FOR_DISPLAY = "Hello there! What can I help you find in this room?"
        return settings

    def run_gradio(self) -> None:
        settings = self.create_settings()
        agent = Agent(scene_name=self.scene_name, settings=settings)
        run_gradio(settings, agent)

def main(root: str, scene_name: str) -> None:
    cwn_folder = os.path.join(root, "chat_with_nerf")
    Path(cwn_folder).mkdir(parents=True, exist_ok=True)
    nerf = ExtractNeRF(root=root, method="lerf")
    config = Config(
        h5_embeddings=nerf.embeddings_file,
        mesh=nerf.mesh_file,
        scene_name=scene_name,
        cwn_folder=cwn_folder,
    )
    config.run_gradio()

def grant_3d_llm() -> None:
    root = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/colmap"
    scene_name = "WakeIsland_1109"
    main(root, scene_name)

def cli() -> None:
    tyro.cli(main)

if __name__ == "__main__":
    grant_3d_llm()