from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")


# LIBERO asset destinations. Derived from this file's location
# (gr00t/configs/l_obstacle/) up to the Isaac-GR00T root, then down into
# external_dependencies/LIBERO/...
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIBERO_ROOT = _REPO_ROOT / "external_dependencies" / "LIBERO" / "libero" / "libero"
L_OBSTACLE_XML_PATH = _LIBERO_ROOT / "assets" / "custom_objects" / "l_obstacle" / "l_obstacle.xml"
L_OBSTACLE_BDDL_PATH = (
    _LIBERO_ROOT / "bddl_files" / "libero_object_l_obstacle"
    / "pick_up_the_butter_and_place_it_in_the_basket.bddl"
)


_cached_mtime_ns: int | None = None
_cached_config: dict[str, Any] | None = None


def load() -> dict[str, Any]:
    """Read config.yaml, hot-reloading when the file changes."""
    global _cached_config, _cached_mtime_ns
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing L-obstacle config: {CONFIG_PATH}")
    stat = CONFIG_PATH.stat()
    if _cached_config is None or _cached_mtime_ns != stat.st_mtime_ns:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Expected YAML mapping in {CONFIG_PATH}, got {type(cfg)}")
        _cached_config = cfg
        _cached_mtime_ns = stat.st_mtime_ns
    return _cached_config


def _render_xml(cfg: dict[str, Any]) -> str:
    arm_length = float(cfg["arm_length"])
    arm_thickness = float(cfg["arm_thickness"])
    half_height = float(cfg["height"]) / 2.0
    half_thick = arm_thickness / 2.0
    half_length = arm_length / 2.0

    material_rgba = cfg.get("material_rgba", [0.15, 0.55, 0.85, 0.25])
    if len(material_rgba) != 4:
        raise ValueError(f"material_rgba must have 4 components, got {material_rgba}")
    rgba_str = " ".join(f"{float(v):.6f}" for v in material_rgba)

    # The L corner sits at the body's local origin; the vertical arm extends
    # along +y and the horizontal arm along +x.
    vertical_center = (0.0, half_length, 0.0)
    vertical_size = (half_thick, half_length, half_height)
    horizontal_center = (half_length, 0.0, 0.0)
    horizontal_size = (half_length, half_thick, half_height)

    def fmt(t):
        return " ".join(f"{v:.6f}" for v in t)

    return (
        "<mujoco model=\"l_obstacle\">\n"
        "  <asset>\n"
        f"    <material name=\"l_obstacle_material\" rgba=\"{rgba_str}\" />\n"
        "  </asset>\n"
        "  <worldbody>\n"
        "    <body>\n"
        "      <body name=\"object\">\n"
        f"        <geom name=\"l_obstacle_vertical_visual\" type=\"box\" pos=\"{fmt(vertical_center)}\" size=\"{fmt(vertical_size)}\" material=\"l_obstacle_material\" contype=\"0\" conaffinity=\"0\" group=\"1\" />\n"
        f"        <geom name=\"l_obstacle_horizontal_visual\" type=\"box\" pos=\"{fmt(horizontal_center)}\" size=\"{fmt(horizontal_size)}\" material=\"l_obstacle_material\" contype=\"0\" conaffinity=\"0\" group=\"1\" />\n"
        f"        <geom name=\"l_obstacle_vertical_collision\" type=\"box\" pos=\"{fmt(vertical_center)}\" size=\"{fmt(vertical_size)}\" density=\"250\" group=\"0\" rgba=\"0.8 0.8 0.8 0.0\" />\n"
        f"        <geom name=\"l_obstacle_horizontal_collision\" type=\"box\" pos=\"{fmt(horizontal_center)}\" size=\"{fmt(horizontal_size)}\" density=\"250\" group=\"0\" rgba=\"0.8 0.8 0.8 0.0\" />\n"
        "      </body>\n"
        f"      <site rgba=\"0 0 0 0\" size=\"0.005\" pos=\"0 0 -{half_height:.6f}\" name=\"bottom_site\" />\n"
        f"      <site rgba=\"0 0 0 0\" size=\"0.005\" pos=\"0 0 {half_height:.6f}\" name=\"top_site\" />\n"
        f"      <site rgba=\"0 0 0 0\" size=\"0.005\" pos=\"{half_length:.6f} 0 0\" name=\"horizontal_radius_site\" />\n"
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )


def _render_bddl(cfg: dict[str, Any]) -> str:
    pos_x, pos_y = (float(v) for v in cfg["position_xy"])
    yaw_rad = math.radians(float(cfg["z_rotation_deg"]))

    # LIBERO's BDDL parser expects a tight rectangle (xmin ymin xmax ymax)
    # plus a yaw range (min max). We pin both to the same value for
    # deterministic placement.
    region_range = f"({pos_x:.6f} {pos_y:.6f} {pos_x + 1e-4:.6f} {pos_y + 1e-4:.6f})"
    yaw_range = f"({yaw_rad:.10f} {yaw_rad:.10f})"

    return (
        "(define (problem LIBERO_Floor_Manipulation)\n"
        "  (:domain robosuite)\n"
        "  (:language Pick the butter and place it in the basket)\n"
        "    (:regions\n"
        "      (bin_region\n"
        "          (:target floor)\n"
        "          (:ranges (\n"
        "              (-0.01 0.25 0.01 0.27)\n"
        "            )\n"
        "          )\n"
        "      )\n"
        "      (target_object_region\n"
        "          (:target floor)\n"
        "          (:ranges (\n"
        "              (-0.145 -0.265 -0.095 -0.215)\n"
        "            )\n"
        "          )\n"
        "      )\n"
        "      (other_object_region_0\n"
        "          (:target floor)\n"
        "          (:ranges (\n"
        "              (0.025 -0.125 0.07500000000000001 -0.07500000000000001)\n"
        "            )\n"
        "          )\n"
        "      )\n"
        "      (other_object_region_1\n"
        "          (:target floor)\n"
        "          (:ranges (\n"
        "              (-0.175 0.034999999999999996 -0.125 0.08499999999999999)\n"
        "            )\n"
        "          )\n"
        "      )\n"
        "      (other_object_region_2\n"
        "          (:target floor)\n"
        "          (:ranges (\n"
        "              (0.07500000000000001 -0.225 0.125 -0.17500000000000002)\n"
        "            )\n"
        "          )\n"
        "      )\n"
        "      (other_object_region_3\n"
        "          (:target floor)\n"
        "          (:ranges (\n"
        "              (0.125 0.0049999999999999975 0.175 0.055)\n"
        "            )\n"
        "          )\n"
        "      )\n"
        "      (other_object_region_4\n"
        "          (:target floor)\n"
        "          (:ranges (\n"
        "              (-0.225 -0.10500000000000001 -0.17500000000000002 -0.055)\n"
        "            )\n"
        "          )\n"
        "      )\n"
        "      (other_object_region_5\n"
        "          (:target floor)\n"
        f"          (:ranges (\n              {region_range}\n            )\n          )\n"
        f"          (:yaw_rotation (\n              {yaw_range}\n            )\n          )\n"
        "      )\n"
        "      (contain_region\n"
        "          (:target basket_1)\n"
        "      )\n"
        "    )\n"
        "\n"
        "  (:fixtures\n"
        "    floor - floor\n"
        "    l_obstacle_1 - l_obstacle\n"
        "  )\n"
        "\n"
        "  (:objects\n"
        "    butter_1 - butter\n"
        "    basket_1 - basket\n"
        "    tomato_sauce_1 - tomato_sauce\n"
        "    orange_juice_1 - orange_juice\n"
        "    chocolate_pudding_1 - chocolate_pudding\n"
        "    bbq_sauce_1 - bbq_sauce\n"
        "    ketchup_1 - ketchup\n"
        "  )\n"
        "\n"
        "  (:obj_of_interest\n"
        "    butter_1\n"
        "    basket_1\n"
        "  )\n"
        "\n"
        "  (:init\n"
        "    (On butter_1 floor_target_object_region)\n"
        "    (On tomato_sauce_1 floor_other_object_region_0)\n"
        "    (On orange_juice_1 floor_other_object_region_1)\n"
        "    (On chocolate_pudding_1 floor_other_object_region_2)\n"
        "    (On bbq_sauce_1 floor_other_object_region_3)\n"
        "    (On ketchup_1 floor_other_object_region_4)\n"
        "    (On l_obstacle_1 floor_other_object_region_5)\n"
        "    (On basket_1 floor_bin_region)\n"
        "  )\n"
        "\n"
        "  (:goal\n"
        "    (And (In butter_1 basket_1_contain_region))\n"
        "  )\n"
        "\n"
        ")\n"
    )


def _write_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def bootstrap_assets() -> None:
    """Regenerate l_obstacle.xml and the BDDL from the current YAML config.

    Safe to call repeatedly; only rewrites files when the rendered content
    actually changes. Intended to be called once at LIBERO env registration
    so the MuJoCo scene always matches the source-of-truth config.
    """
    cfg = load()
    _write_if_changed(L_OBSTACLE_XML_PATH, _render_xml(cfg))
    _write_if_changed(L_OBSTACLE_BDDL_PATH, _render_bddl(cfg))


if __name__ == "__main__":
    bootstrap_assets()
    print(f"wrote {L_OBSTACLE_XML_PATH}")
    print(f"wrote {L_OBSTACLE_BDDL_PATH}")
