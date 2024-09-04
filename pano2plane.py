from pathlib import Path
import imageio
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import os
from rich.progress import track
from rich.console import Console
import tyro
console = Console()

# https://blogs.codingballad.com/unwrapping-the-view-transforming-360-panoramas-into-intuitive-videos-with-python-6009bd5bca94

def map_to_sphere(x: np.ndarray, y: np.ndarray, z: np.ndarray, yaw_radian: float, pitch_radian: float) -> tuple[np.ndarray, np.ndarray]:


    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    # Apply rotation transformations here
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                           np.cos(theta) * np.sin(pitch_radian),
                           np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()


def interpolate_color(coords: np.ndarray, img: np.ndarray, method: str = 'bilinear') -> np.ndarray:
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_path: str, FOV: int, output_size: tuple[int, int], yaw: int, pitch: int) -> Image.Image:
    panorama = Image.open(panorama_path).convert('RGB')
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    H, W = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    x = u - W / 2
    y = H / 2 - v
    z = f

    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)

    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')

    return output_image

# Usage
def main(folder: str, /, *, degree: int = 2, output_size: tuple[int, int] = (2046, 1028), output_dir: str) -> None:
    if not Path(folder).exists():
        raise FileNotFoundError(f"Panorama folder {folder} not found.")
    console.rule(f"Panorama to Plane for 360 video - {folder}")
    pano_paths = sorted(list(Path(folder).glob("*.jpg")))
    # panorama_path = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/Panoramas/00000-pano.jpg"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx, panorama_path in enumerate(pano_paths):
        out_video_path = str(panorama_path).replace(".jpg", ".mp4")
        if Path(out_video_path).exists() and Path(out_video_path).is_file():
            console.print(f"Skipping {panorama_path.stem} as video already exists. Delete the video to reprocess.")
            continue
        writer = imageio.get_writer(out_video_path, fps=15)

        for deg in track(np.arange(0, 360, degree), description=f"-> [{idx+1}/{len(pano_paths)}] Processing {panorama_path.stem}", transient=True):
            output_image = panorama_to_plane(panorama_path, 90, output_size, deg, 90)
            writer.append_data(np.array(output_image))
            output_file = os.path.join(output_dir, panorama_path.stem + f"_deg{str(deg).zfill(3)}.jpg")
            output_image.save(output_file)

        writer.close()
        console.print(f"Video saved to {out_video_path}")
    console.rule("Done! :thumbs_up:", style="green")

def create_folder_colmap(input_dir: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for pano in Path(input_dir).glob("*.jpg"):
        output_file = os.path.join(output_dir, pano.name)
        relative_path = os.path.relpath(pano, output_dir)
        # create a relative symlink
        Path(output_file).symlink_to(relative_path)

def grant_3d_llm() -> None:
    panos = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/Panoramas"
    output = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/PanoFrames"
    output_colmap = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/colmap/images"
    degree = 10
    output_size = (2048, 1024)
    if Path(output).exists():
        console.print(f"Output folder {output} already exists. Delete the folder to reprocess.")
        return
    elif Path(output_colmap).exists():
        console.print(f"Output folder {output_colmap} already exists. Delete the folder to reprocess.")
        return
    main(panos, output_dir=output, degree=degree, output_size=output_size)
    create_folder_colmap(output, output_colmap)

def cli() -> None:
    tyro.cli(main)

if __name__ == "__main__":
    grant_3d_llm()
    # cli()