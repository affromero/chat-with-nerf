from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import subprocess
from typing import Any, Literal
import pycolmap
import tyro
from rich.console import Console

from colmap2nerf import colmap2nerf

console = Console()

# inspired from https://github.com/colmap/pycolmap/blob/master/example.py and https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/reconstruction.py

def run_reconstruction(
    root: Path,
    options: dict[str, Any] | None = None,
) -> pycolmap.Reconstruction:
    database_path = root / "database.db"
    if not database_path.exists():
        raise FileNotFoundError(f"Database file {database_path} not found.")
    image_dir = root / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} not found.")
    models_path = root / "sparse"
    models_path.mkdir(exist_ok=True, parents=True)
    console.log("Running 3D reconstruction...")
    if options is None:
        options = {}
    options = {"num_threads": min(multiprocessing.cpu_count(), 16), **options}
    with pycolmap.ostream():
        reconstructions = pycolmap.incremental_mapping(
            str(database_path), str(image_dir), str(models_path), options=options
        )

    if len(reconstructions) == 0:
        console.log(":skull: Could not reconstruct any model!", style="bold red")
        raise RuntimeError("Could not reconstruct any model.")
    console.log(f"Reconstructed {len(reconstructions)} model(s).", style="green")

    largest_index = -1
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    if largest_index == -1:
        raise RuntimeError("Could not find the largest model.")
    console.log(
        f"Largest model is #{largest_index} with {largest_num_images} images."
    )

    bin_folder = models_path / str(largest_index)
    for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
        bin_file = str(models_path / str(largest_index) / filename)
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"File {bin_file} not found.")
    
    # convert bin files to txt files
    txt_folder = str(models_path) + "_txt"
    if not os.path.exists(txt_folder):
        subprocess.run([f"colmap model_converter \
        --input_path {bin_folder} \
        --output_path {txt_folder} \
        --output_type TXT"], shell=True, check=True)

    console.log(
        f"Reconstruction statistics:\n{reconstructions[largest_index].summary()}"
        + f"\n\tnum_input_images = {len(reconstructions[largest_index].images)}",
    )

    return reconstructions[largest_index]

def automatic_reconstructor(root: str, camera_model: str = "SIMPLE_RADIAL_FISHEYE") -> None:
    # different camera models: https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py#L238
    image_folder = os.path.join(root, "images")
    # image_folder contains the images to be reconstructed
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder {image_folder} not found.")
    
    console.rule(f"Automatic 3D Reconstruction - {image_folder}", characters="=")

    console.rule("Extracting features")
    # https://colmap.github.io/tutorial.html#feature-detection-and-extraction
    database_path = os.path.join(root, "database.db")
    if not os.path.exists(database_path):
        pycolmap.extract_features(database_path, image_folder, camera_model=camera_model)
        console.log(f"Extracted features from {image_folder} and saved to {database_path}")
    else:
        console.log(f"Features already extracted and saved to {database_path}")

    console.rule("Matching features")
    # https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification
    pycolmap.match_exhaustive(database_path)

    console.rule("Reconstructing 3D points")
    # https://colmap.github.io/tutorial.html#sparse-reconstruction
    run_reconstruction(Path(root))

    console.rule("Converting to NeRF format")
    colmap2nerf(
        root=os.path.join(root, "sparse_txt"),
        image_folder=os.path.join(root, "images"),
        output_json=os.path.join(root, "transforms.json"),
    )

    console.rule("Finished 3D reconstruction", characters="=")

@dataclass
class ExtractNeRF:
    root: str
    """ Root directory containing the transforms.json file. """

    method: Literal["lerf"] = "lerf"
    """ Method to use for NeRF training. """

    def __post_init__(self) -> None:
        if not os.path.exists(self.transforms_file):
            raise FileNotFoundError(f"File {self.transforms_file} not found. Check root directory.")

    @property
    def transforms_file(self) -> str:
        return os.path.join(self.root, "transforms.json")
    
    @property
    def __output_dir(self) -> str:
        return "outputs"
    
    @property
    def output_dir(self) -> str:
        return os.path.join(self.__output_dir, os.path.basename(self.root), self.method)

    def run_nerf(self) -> None:
        # check there is a transforms.json file
        command = f"poetry run ns-train {self.method} --data {self.root} --output-dir {self.__output_dir}"
        console.log(f"Running NeRF training with {command=}")
        subprocess.run(command, shell=True, check=True)

    def create_h5_from_nerf(self) -> None:
        latest_folder = sorted(list(Path(self.output_dir).glob("*")))[-1]
        if not os.path.exists(latest_folder):
            raise FileNotFoundError(f"Folder {latest_folder} not found. Check output directory.")
        config_file = str(latest_folder / "config.yml")
        output_dir = str(latest_folder / "nerfstudio_models")
        command = f"poetry run ns-export pointcloud --load-config {config_file} --output-dir {output_dir} --normal_method open3d"
        console.log(f"Running NeRF export with {command=}")
        subprocess.run(command, shell=True, check=True)

    def run(self) -> None:
        self.run_nerf()
        self.create_h5_from_nerf()


def main(root: str) -> None:
    automatic_reconstructor(root)
    ExtractNeRF(root).run()

def grant_3d_llm() -> None:
    root = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/colmap"
    main(root)

def cli() -> None:
    tyro.cli(main)

if __name__ == "__main__":
    grant_3d_llm()
    # cli()