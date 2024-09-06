class Chat_With_NeRF_Settings:
    # data_path: str = "/workspace/chat-with-nerf-dev/chat-with-nerf/data"
    data_path: str = "data/lerf_data_experiments/"
    # output_path: str = "/workspace/chat-with-nerf-dev/chat-with-nerf/session_output"
    output_path: str = "session_output"
    CLIP_FILTERING_THRESHOLD: float = 21  # range is (0, 100)
    default_scene: str = "scene0025_00"
    INITIAL_MSG_FOR_DISPLAY: str = "Hello there! What can I help you find in this room?"
    MAX_TURNS: int = 1000
    DEFAULT_IMAGE_TOKEN: str = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN: str = "<im_patch>"
    DEFAULT_IM_START_TOKEN: str = "<im_start>"
    DEFAULT_IM_END_TOKEN: str = "<im_end>"
    MAX_WORKERS: int = 5
    # IMAGES_PATH = "/workspace/chat-with-nerf-dev/chat-with-nerf/scene_images"
    # NERF_DATA_PATH = "/workspace/chat-with-nerf-dev/chat-with-nerf/data"
    IMAGES_PATH: str = "scene_images"
    NERF_DATA_PATH: str = "data/lerf_data_experiments/"
    TOP_THREE_NO_GPT: bool = False
    USE_OPENSCENE: bool = True
    # this flag is reserved for in the wild data
    IS_SCANNET: bool = False
    # this flag is only used for evaluation
    IS_EVALUATION: bool = False