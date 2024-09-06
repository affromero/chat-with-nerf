from pathlib import Path
import cv2
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
    """
    Map the x, y, z coordinates to the sphere and apply the rotation transformations.
    Args: 
        x: The x coordinates.
        y: The y coordinates.
        z: The z coordinates.
        yaw_radian: The yaw rotation in radians.
        pitch_radian: The pitch rotation in radians.
    """
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
    """
    Interpolate the color values of the panorama image based on the coordinates.
    Args:
        coords: The coordinates of the plane image.
        img: The panorama image.
        method: The interpolation method.
    """
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_path: str, FOV: int, output_size: tuple[int, int], yaw: int, pitch: int) -> Image.Image:
    """
    Convert the panorama image to a plane image.
    Args:
        panorama_path: The path to the panorama image.
        FOV: The field of view.
        output_size: The output size of the plane image.
        yaw: The yaw rotation.
        pitch: The pitch rotation.
    """
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


def fisheye_to_pinhole(pil_image: Image.Image) -> Image.Image:
    """
    Convert the fisheye image to a pinhole image.
    Args:
        pil_image: The fisheye image.
    """
    # Convert Pillow image to OpenCV format
    open_cv_image = np.array(pil_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Get the dimensions of the image
    height, width = open_cv_image.shape[:2]
    
    # Create the map for x and y coordinates
    # Assuming a generic fisheye distortion for remapping
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    
    # Define center of the image (assumed to be optical center)
    center_x = width / 2
    center_y = height / 2
    
    # Focal length and distortion factor (tunable, this is an approximation)
    focal_length = width / 2  # Focal length estimation
    distortion_strength = 0.5  # Tuning factor for distortion, adjust this value

    # Remapping the fisheye image
    for y in range(height):
        for x in range(width):
            # Normalize coordinates
            norm_x = (x - center_x) / focal_length
            norm_y = (y - center_y) / focal_length
            r = np.sqrt(norm_x**2 + norm_y**2)
            
            # Apply inverse fisheye transformation (basic approximation)
            theta = np.arctan(r)
            scale = np.tan(theta * distortion_strength) / r if r != 0 else 1
            
            # Map the distorted image back to rectilinear coordinates
            map_x[y, x] = center_x + norm_x * scale * focal_length
            map_y[y, x] = center_y + norm_y * scale * focal_length
    
    # Remap the image to create the pinhole effect
    undistorted_image = cv2.remap(open_cv_image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Convert back to Pillow image and return
    undistorted_pil_image = Image.fromarray(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
    return undistorted_pil_image

# Usage
def main(folder: str, /, *, degree: int = 2, output_size: tuple[int, int], output_dir: str, fish2pin: bool = False) -> None:
    """
    Convert the panorama images to plane images.
    Args:
        folder: The folder containing the panorama images.
        degree: The degree of rotation.
        output_size: The output size of the plane image. (height, width)
        output_dir: The output directory to save the plane images.
        fish2pin: Convert the fisheye image to pinhole image.
    """
    if not Path(folder).exists():
        raise FileNotFoundError(f"Panorama folder {folder} not found.")
    console.rule(f"Panorama to Plane for 360 video - {folder}")
    pano_paths = sorted(list(Path(folder).glob("*.jpg")))
    # panorama_path = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/Panoramas/00000-pano.jpg"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx, panorama_path in enumerate(pano_paths):
        if fish2pin:
            out_video_path = str(panorama_path).replace(".jpg", "_fish2pin.mp4")
            Path(out_video_path).unlink(missing_ok=True)
        else:
            out_video_path = str(panorama_path).replace(".jpg", ".mp4")
        if Path(out_video_path).exists() and Path(out_video_path).is_file():
            console.print(f"Skipping {panorama_path.stem} as video already exists. Delete the video to reprocess.")
            continue
        writer = imageio.get_writer(out_video_path, fps=15)

        for deg in track(np.arange(0, 360, degree), description=f"-> [{idx+1}/{len(pano_paths)}] Processing {panorama_path.stem}", transient=True):
            output_image = panorama_to_plane(panorama_path, 90, output_size, deg, 90)
            if fish2pin:
                output_image = fisheye_to_pinhole(output_image)
            writer.append_data(np.array(output_image))
            output_file = os.path.join(output_dir, panorama_path.stem + f"_deg{str(deg).zfill(3)}.jpg")
            output_image.save(output_file)

        writer.close()
        console.print(f"Video saved to {out_video_path}")
    console.rule("Done! :thumbs_up:", style="green")

def create_folder_colmap(input_dir: str, output_dir: str) -> None:
    """
    Create a symlink for the COLMAP images.
    Args:
        input_dir: The input directory containing the images.
        output_dir: The output directory to save the symlink
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    console.rule(f"Creating symlink for COLMAP - {input_dir}")
    for pano in Path(input_dir).glob("*.jpg"):
        output_file = os.path.join(output_dir, pano.name)
        relative_path = os.path.relpath(pano, output_dir)
        # create a relative symlink
        Path(output_file).symlink_to(relative_path)
    console.rule("Done! :thumbs_up:", style="green")

def grant_3d_llm() -> None:
    """
    Main function to convert the panorama images to plane images for the 3D reconstruction.
    """
    panos = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/Panoramas"
    output = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/PanoFrames"
    output_colmap = "user_shared/grant_llm_3d/quest_020924/WakeIsland_1109/colmap/images"
    degree = 10
    output_size = (2048, 1024)
    if Path(output).exists():
        console.print(f"Output folder {output=} already exists. Main function to convert panos to frames will be skipped.")
    else:
        main(panos, output_dir=output, degree=degree, output_size=output_size, fish2pin=False)
    
    if Path(output_colmap).exists():
        console.print(f"Output folder {output_colmap=} already exists. Delete the folder to reprocess.")
    else:
        create_folder_colmap(output, output_colmap)

def cli() -> None:
    tyro.cli(main)

if __name__ == "__main__":
    grant_3d_llm()
    # cli()