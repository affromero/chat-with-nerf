from pydantic.dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import subprocess
from typing import Any, Literal, TypeAlias, cast
import pycolmap
import tyro
from rich.console import Console

from colmap2nerf import colmap2nerf

MATCHING_TYPES: TypeAlias = Literal["exhaustive", "sequential", "spatial", "tree", "transitive"]

console = Console()

# inspired from https://github.com/colmap/pycolmap/blob/master/example.py and https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/reconstruction.py

def run_reconstruction(
    root: Path,
    options: dict[str, Any] | None = None,
) -> tuple[pycolmap.Reconstruction, str]:
    """
    Run the 3D reconstruction using COLMAP.
    Args:
        root: The root folder containing the database.db and images folder.
        options: Additional options to pass to the reconstruction.
    """
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
    txt_folder = str(Path(str(models_path) + "_txt") / str(largest_index))
    if not os.path.exists(txt_folder):
        Path(txt_folder).mkdir(parents=True, exist_ok=True)
        subprocess.run([f"colmap model_converter \
        --input_path {bin_folder} \
        --output_path {txt_folder} \
        --output_type TXT"], shell=True, check=True)

    console.log(
        f"Reconstruction statistics:\n{reconstructions[largest_index].summary()}"
        + f"\n\tnum_input_images = {len(reconstructions[largest_index].images)}",
    )

    return reconstructions[largest_index], txt_folder

def automatic_reconstructor(root: str, /, *, camera_model: str = "SIMPLE_PINHOLE", matching: MATCHING_TYPES) -> None:
    """
    Automatically reconstructs 3D points from images in a folder.
    First, it will extract features from the images, then match features, and then reconstruct 3D points.
    Finally, it will convert the 3D points to NeRF format.
    """
    # different camera models: https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py#L238
    image_folder = os.path.join(root, "images")
    # image_folder contains the images to be reconstructed
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder {image_folder} not found.")
    
    console.rule(f"Automatic 3D Reconstruction - {image_folder}", characters="=")

    # --------------------------------------------------------------------------------------------------- #
    # ---------------------------------------- Extract features ----------------------------------------- #
    # --------------------------------------------------------------------------------------------------- #
    console.rule("Extracting features")
    # https://colmap.github.io/tutorial.html#feature-detection-and-extraction
    database_path = os.path.join(root, "database.db")
    if not os.path.exists(database_path):
        pycolmap.extract_features(database_path, image_folder, camera_model=camera_model)
        console.log(f"Extracted features from {image_folder} and saved to {database_path}")
    else:
        console.log(f"Features already extracted and saved to {database_path}")

    # --------------------------------------------------------------------------------------------------- #
    # ----------------------------------------- Match features ------------------------------------------ #
    # --------------------------------------------------------------------------------------------------- #
    console.rule("Matching features")
    # https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification
    if matching == "exhaustive":
        pycolmap.match_exhaustive(database_path)
    elif matching == "sequential":
        pycolmap.match_sequential(database_path)
    elif matching == "spatial":
        pycolmap.match_spatial(database_path)
    elif matching == "tree":
        # https://demuc.de/colmap/
        # Vocabulary tree with 32K visual words: https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin
        # (for small-scale image collections, i.e. 100s to 1,000s of images)
        # Vocabulary tree with 256K visual words: https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin
        # (for medium-scale image collections, i.e. 1,000s to 10,000s of images)
        # Vocabulary tree with 1M visual words: https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin
        # (for large-scale image collections, i.e. 10,000s - 100,000s of images)
        default_tree = "vocab_tree_flickr100K_words256K.bin"
        if not os.path.exists(default_tree):
            # download the default tree
            console.log(f"Downloading default vocabulary tree {default_tree}")
            subprocess.run(["wget", f"https://demuc.de/colmap/{default_tree}"], check=True)
        matching_opt= pycolmap.VocabTreeMatchingOptions(vocab_tree_path=default_tree)
        pycolmap.match_vocabtree(database_path, matching_options=matching_opt)

    # --------------------------------------------------------------------------------------------------- #
    # -------------------------------------- Reconstruct 3D points -------------------------------------- #
    # --------------------------------------------------------------------------------------------------- #
    console.rule("Reconstructing 3D points")
    # https://colmap.github.io/tutorial.html#sparse-reconstruction
    _, sparse_txt_path = run_reconstruction(Path(root))

    # --------------------------------------------------------------------------------------------------- #
    # ------------------------------------- Convert to NeRF format -------------------------------------- #
    # --------------------------------------------------------------------------------------------------- #
    console.rule("Converting to NeRF format")
    colmap2nerf(
        root=sparse_txt_path,
        image_folder=os.path.join(root, "images"),
        output_json=os.path.join(root, "transforms.json"),
    )

    # --------------------------------------------------------------------------------------------------- #
    # ------------------------------------- Finished 3D reconstruction ---------------------------------- #
    # --------------------------------------------------------------------------------------------------- #
    console.rule("Finished 3D reconstruction", characters="=")

@dataclass
class ExtractNeRF:
    """
    Extract NeRF from the 3D points reconstructed by COLMAP.
    """

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

    @property
    def embeddings_file(self) -> str:
        _file = os.path.join(self.output_dir, "embeddings.h5")
        return _file
    
    @property
    def mesh_file(self) -> str:
        _file = os.path.join(self.output_dir, "poisson_mesh.ply")
        if not os.path.exists(_file):
            raise FileNotFoundError(f"File {_file} not found. Check output directory. Or run create_h5_from_nerf.")
        return _file

    @property
    def point_cloud_file(self) -> str:
        _file = os.path.join(self.output_dir, "point_cloud.ply")
        if not os.path.exists(_file):
            raise FileNotFoundError(f"File {_file} not found. Check output directory. Or run create_h5_from_nerf.")
        return _file

    def run_nerf(self) -> None:
        """ Run NeRF training on the frames with their respective poses. """
        command = f"poetry run ns-train {self.method} --data {self.root} --output-dir {self.__output_dir} --pipeline.model.predict-normals True"
        # normal prediction is important for exporting the mesh
        console.log(f"Running NeRF training with {command=}")
        subprocess.run(command, shell=True, check=True)

    def create_h5_from_nerf(self) -> None:
        """ Create h5 files from the NeRF models. Useful for [ChatWithNeRF](https://chat-with-nerf.github.io). """
        latest_folder = sorted(list(Path(self.output_dir).glob("*")))[-1]
        if not os.path.exists(latest_folder):
            raise FileNotFoundError(f"Folder {latest_folder} not found. Check output directory.")
        config_file = str(latest_folder / "config.yml")
        command = f"poetry run ns-export poisson \
            --load-config {config_file} \
            --output-dir {self.output_dir} \
            --hdf5_file {self.embeddings_file} \
            --save-point-cloud"
        # poisson is used for exporting the mesh: https://docs.nerf.studio/quickstart/export_geometry.html#poisson-surface-reconstruction
        console.log(f"Running NeRF export with {command=}")
        subprocess.run(command, shell=True, check=True)
        console.log(f"Exported mesh to {self.mesh_file}, point cloud to {self.point_cloud_file}, and h5 embeddings to {self.embeddings_file}", style="green")

    def run(self) -> None:
        """ Run the NeRF training and create h5 files. """
        self.run_nerf()
        self.create_h5_from_nerf()


def main(root: str, camera_model: str, matching: MATCHING_TYPES, run_nerf: bool) -> tuple[str, str, str] | None:
    """ Main function to run the 3D reconstruction. """
    automatic_reconstructor(root, camera_model=camera_model, matching=matching)
    if run_nerf:
        nerf = ExtractNeRF(root)
        if not os.path.exists(nerf.embeddings_file):
            nerf.run()
        return (
            nerf.embeddings_file,
            nerf.mesh_file,
            nerf.point_cloud_file,
        )

def grant_3d_llm() -> None:
    root = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/colmap"
    camera_model = "SIMPLE_PINHOLE"
    matching = cast(MATCHING_TYPES, "tree")
    main(root, camera_model=camera_model, matching=matching, run_nerf=False)

def cli() -> None:
    tyro.cli(main)

if __name__ == "__main__":
    grant_3d_llm()
    # cli()