from attrs import define


@define
class SceneConfig:
    scene_name: str
    load_h5_config: str
    load_openscene: str
    load_mesh: str
    load_metadata: str
