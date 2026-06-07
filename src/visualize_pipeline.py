from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
import pygame
import torch

from utils import FiLMRoutedUNet3D, extract_porespy_openpnm_network

try:
    import imgui
    from imgui.integrations.opengl import ProgrammablePipelineRenderer
    from imgui.integrations.pygame import PygameRenderer
except ImportError:
    imgui = None
    PygameRenderer = None
    ProgrammablePipelineRenderer = None


DEFAULT_SHAPE = (1000, 1000, 1000)
MODE_NAMES = ("raw", "probability", "mask", "graph")
GRAPH_COORD_ORDER_KEYS = ("zyx", "xyz")
GRAPH_COORD_ORDER_LABELS = ("pore.coords z,y,x -> view x,y,z", "pore.coords x,y,z")


if PygameRenderer is not None and ProgrammablePipelineRenderer is not None:

    class PygameProgrammableRenderer(ProgrammablePipelineRenderer):
        def __init__(self):
            super().__init__()
            self._gui_time = None
            self.custom_key_map = {}
            PygameRenderer._map_keys(self)

        _custom_key = PygameRenderer._custom_key
        _map_keys = PygameRenderer._map_keys
        process_inputs = PygameRenderer.process_inputs

        def process_event(self, event):
            if event.type == pygame.VIDEORESIZE:
                self.io.display_size = event.size
                self.refresh_font_texture()
                return True
            return PygameRenderer.process_event(self, event)

else:
    PygameProgrammableRenderer = None


VOLUME_VERTEX_SHADER = """
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


VOLUME_FRAGMENT_SHADER = """
#version 330

uniform sampler3D volume_tex;
uniform vec3 volume_shape;
uniform mat3 camera_rot;
uniform float zoom;
uniform float density;
uniform float threshold;
uniform float window_ratio;
uniform int palette_mode;

in vec2 v_uv;
out vec4 frag_color;

const int MAX_STEPS = 1024;

bool hit_box(vec3 ro, vec3 rd, out float t0, out float t1) {
    vec3 inv_rd = 1.0 / rd;
    vec3 tmin_tmp = (-0.5 - ro) * inv_rd;
    vec3 tmax_tmp = (0.5 - ro) * inv_rd;
    vec3 tmin = min(tmin_tmp, tmax_tmp);
    vec3 tmax = max(tmin_tmp, tmax_tmp);
    t0 = max(max(tmin.x, tmin.y), tmin.z);
    t1 = min(min(tmax.x, tmax.y), tmax.z);
    return t1 >= max(t0, 0.0);
}

vec3 value_color(float value) {
    if (palette_mode == 1) {
        return mix(vec3(0.02, 0.20, 0.42), vec3(1.0, 0.58, 0.08), value);
    }
    if (palette_mode == 2) {
        return mix(vec3(0.0, 0.45, 0.85), vec3(0.2, 1.0, 0.74), value);
    }
    return vec3(value);
}

float value_alpha(float value) {
    if (palette_mode == 0) {
        return clamp((value - threshold) * density, 0.0, 0.12);
    }
    if (palette_mode == 1) {
        return clamp(value * density * 0.065, 0.0, 0.18);
    }
    return value > threshold ? clamp(density * 0.060, 0.0, 0.20) : 0.0;
}

void main() {
    vec2 p = v_uv * 2.0 - 1.0;
    p.x *= window_ratio;

    vec3 ro = camera_rot * vec3(0.0, 0.0, zoom);
    vec3 rd = normalize(camera_rot * vec3(p, -1.8));

    float t0;
    float t1;
    if (!hit_box(ro, rd, t0, t1)) {
        frag_color = vec4(0.015, 0.017, 0.022, 1.0);
        return;
    }

    t0 = max(t0, 0.0);
    float max_dim = max(max(volume_shape.x, volume_shape.y), volume_shape.z);
    float step_count = min(float(MAX_STEPS), max_dim * 1.9);
    float step_size = 1.75 / step_count;
    float alpha = 0.0;
    vec3 color = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (float(i) >= step_count) {
            break;
        }

        float t = t0 + float(i) * step_size;
        if (t > t1 || alpha > 0.98) {
            break;
        }

        vec3 pos = ro + rd * t;
        vec3 uvw = pos + 0.5;
        float value = texture(volume_tex, uvw).r;
        float a = value_alpha(value);
        vec3 c = value_color(value);
        color += (1.0 - alpha) * a * c;
        alpha += (1.0 - alpha) * a;
    }

    vec3 bg = vec3(0.015, 0.017, 0.022);
    frag_color = vec4(mix(bg, color, alpha), 1.0);
}
"""


GRAPH_VERTEX_SHADER = """
#version 330

uniform mat3 camera_rot;
uniform float zoom;
uniform float window_ratio;
uniform float point_size;

in vec3 in_pos;

void main() {
    vec3 ro = camera_rot * vec3(0.0, 0.0, zoom);
    vec3 camera_pos = transpose(camera_rot) * (in_pos - ro);
    float perspective = -1.8 / min(camera_pos.z, -0.001);
    vec2 screen = camera_pos.xy * perspective;
    gl_Position = vec4(screen.x / window_ratio, screen.y, 0.0, 1.0);
    gl_PointSize = point_size;
}
"""


GRAPH_FRAGMENT_SHADER = """
#version 330

uniform vec4 color;
out vec4 frag_color;

void main() {
    frag_color = color;
}
"""


@dataclass
class GraphBuffers:
    point_vao: moderngl.VertexArray | None = None
    line_vao: moderngl.VertexArray | None = None
    point_buffer: moderngl.Buffer | None = None
    line_buffer: moderngl.Buffer | None = None
    point_count: int = 0
    line_count: int = 0


@dataclass
class PipelineState:
    raw: np.ndarray
    origin: tuple[int, int, int]
    threshold: float
    probability: np.ndarray | None = None
    mask: np.ndarray | None = None
    graph_coords: np.ndarray | None = None
    graph_edges: np.ndarray | None = None
    graph_threshold: float | None = None
    mode: int = 0
    status: str = "raw loaded"
    stage: str = "idle"
    error: str | None = None
    running: bool = False
    worker: threading.Thread | None = None
    events: queue.Queue[tuple[str, Any]] | None = None

    def has_probability(self) -> bool:
        return self.probability is not None

    def has_mask(self) -> bool:
        return self.mask is not None

    def has_graph(self) -> bool:
        return self.graph_coords is not None and self.graph_edges is not None

    def rebuild_mask(self) -> None:
        if self.probability is None:
            self.mask = None
        else:
            self.mask = self.probability >= self.threshold

    def current_volume(self) -> np.ndarray:
        if self.mode == 1:
            if self.probability is None:
                return np.zeros_like(self.raw, dtype=np.uint8)
            return np.clip(self.probability * 255.0, 0, 255).astype(np.uint8)
        if self.mode in (2, 3):
            if self.mask is None:
                return np.zeros_like(self.raw, dtype=np.uint8)
            if self.mode == 3:
                return (self.mask.astype(np.uint8) * 90).astype(np.uint8)
            return self.mask.astype(np.uint8) * 255
        return self.raw

    @property
    def pore_fraction(self) -> float:
        if self.mask is None:
            return 0.0
        return float(self.mask.mean())


def parse_zyx(text: str) -> tuple[int, int, int]:
    parts = text.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("value must look like z,y,x")
    try:
        values = tuple(int(part.strip()) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("coordinates must be integers") from error
    return values


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "utils").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("Project root with src/utils and data was not found")


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def load_subcube(path: Path, shape: tuple[int, int, int], cube_size: int, origin: tuple[int, int, int] | None):
    cube_size = int(cube_size)
    if cube_size <= 0 or cube_size > min(shape):
        raise ValueError(f"cube_size must be in 1..{min(shape)}")

    if origin is None:
        origin = tuple((size - cube_size) // 2 for size in shape)
    else:
        origin = tuple(clamp_int(int(coord), 0, dim - cube_size) for coord, dim in zip(origin, shape))

    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=shape)
    z, y, x = origin
    cube = np.asarray(raw[z : z + cube_size, y : y + cube_size, x : x + cube_size]).copy()
    del raw
    return cube, origin


def sample_origin_from_index(root: Path, sample_index: int, cube_size: int, fallback: tuple[int, int, int] | None):
    index_path = root / "dataset_128" / "index_128.csv"
    if sample_index < 0 or not index_path.exists():
        return fallback

    import pandas as pd

    df = pd.read_csv(index_path)
    if sample_index >= len(df):
        raise ValueError(f"sample_index {sample_index} is outside index_128.csv")

    row = df.iloc[sample_index]
    base = (int(row["z"]), int(row["y"]), int(row["x"]))
    if cube_size >= 128:
        return base

    shift = (128 - cube_size) // 2
    return tuple(value + shift for value in base)


def build_segmentation_model(checkpoint_path: Path, device: torch.device, base_channels: int, ctx_dim: int):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = int(checkpoint.get("base_channels", base_channels))
    ctx_dim = int(checkpoint.get("ctx_dim", ctx_dim))
    model = FiLMRoutedUNet3D(base_channels=base_channels, ctx_dim=ctx_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def segment_cube(model: torch.nn.Module, raw_cube: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(raw_cube.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return probability.astype(np.float32)


def graph_coords_to_view(
    coords: np.ndarray,
    coord_order: str = "zyx",
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return coords.reshape(0, 3)

    if coord_order == "zyx":
        coords = coords[:, [2, 1, 0]]
    elif coord_order == "xyz":
        coords = coords[:, [0, 1, 2]]
    else:
        raise ValueError(f"Unknown graph coordinate order: {coord_order}")

    low = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - low, 1.0e-6)
    view_coords = (coords - low) / span - 0.5
    if flip_x:
        view_coords[:, 0] *= -1.0
    if flip_y:
        view_coords[:, 1] *= -1.0
    if flip_z:
        view_coords[:, 2] *= -1.0
    return view_coords.astype("f4")


def extract_graph(mask: np.ndarray, voxel_size: float, sigma: float, r_max: int):
    pn = extract_porespy_openpnm_network(
        pore_mask=mask,
        voxel_size=voxel_size,
        sigma=sigma,
        r_max=r_max,
    )
    coords = np.asarray(pn["pore.coords"], dtype=np.float32)
    conns = np.asarray(pn["throat.conns"], dtype=np.int64)
    return coords, conns


def prepare_state(args: argparse.Namespace) -> tuple[PipelineState, dict[str, Path | tuple[int, int, int]]]:
    root = find_project_root(Path.cwd())
    raw_path = args.raw or root / "data" / "Berea_2d25um_grayscale_filtered.raw"
    binary_path = args.binary or root / "data" / "Berea_2d25um_binary.raw"
    checkpoint_path = args.checkpoint or root / "models" / "film_routed_unet3d_best.pth"
    origin = sample_origin_from_index(root, args.sample_index, args.cube_size, args.origin)
    raw_cube, origin = load_subcube(raw_path, args.shape, args.cube_size, origin)

    state = PipelineState(raw=raw_cube, origin=origin, threshold=args.threshold)
    paths = {
        "root": root,
        "raw": raw_path,
        "binary": binary_path,
        "checkpoint": checkpoint_path,
    }
    return state, paths


def resolve_mask_source(args: argparse.Namespace, checkpoint_path: Path) -> str:
    if args.mask_source != "auto":
        return args.mask_source
    return "model" if checkpoint_path.exists() else "target"


def start_pipeline_worker(
    state: PipelineState,
    args: argparse.Namespace,
    paths: dict[str, Any],
    run_graph: bool = True,
) -> None:
    if state.running:
        return

    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    state.events = events
    state.running = True
    state.error = None
    threshold_snapshot = float(state.threshold)
    mask_source = resolve_mask_source(args, paths["checkpoint"])

    def emit(tag: str, payload: Any) -> None:
        events.put((tag, payload))

    def worker_loop() -> None:
        try:
            emit("status", f"mask source: {mask_source}")
            if mask_source == "model":
                if not paths["checkpoint"].exists():
                    raise FileNotFoundError(f"checkpoint was not found: {paths['checkpoint']}")
                device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
                emit("status", f"loading model on {device}")
                model = build_segmentation_model(paths["checkpoint"], device, args.base_channels, args.ctx_dim)
                emit("status", "running segmentation")
                probability = segment_cube(model, state.raw, device)
            elif mask_source == "target":
                emit("status", "loading target binary mask")
                binary_cube, _ = load_subcube(paths["binary"], args.shape, args.cube_size, state.origin)
                probability = (binary_cube == args.pore_value).astype(np.float32)
            else:
                emit("status", "thresholding raw intensity")
                probability = (state.raw.astype(np.float32) / 255.0 >= args.raw_threshold).astype(np.float32)

            emit("probability", probability)
            mask = probability >= threshold_snapshot
            emit("mask", mask)

            if run_graph and not args.skip_graph:
                emit("status", "extracting PoreSpy/OpenPNM graph")
                coords, conns = extract_graph(mask, args.voxel_size, args.sigma, args.r_max)
                emit("graph", (coords, conns, threshold_snapshot))

            emit("status", "done")
        except Exception as error:
            emit("error", str(error))
        finally:
            emit("finished", None)

    state.worker = threading.Thread(target=worker_loop, daemon=True)
    state.worker.start()


def start_graph_worker(state: PipelineState, args: argparse.Namespace) -> None:
    if state.running or state.mask is None:
        return

    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    state.events = events
    state.running = True
    state.error = None
    threshold_snapshot = float(state.threshold)
    mask = state.mask.copy()

    def worker_loop() -> None:
        try:
            events.put(("status", "extracting PoreSpy/OpenPNM graph"))
            coords, conns = extract_graph(mask, args.voxel_size, args.sigma, args.r_max)
            events.put(("graph", (coords, conns, threshold_snapshot)))
            events.put(("status", "done"))
        except Exception as error:
            events.put(("error", str(error)))
        finally:
            events.put(("finished", None))

    state.worker = threading.Thread(target=worker_loop, daemon=True)
    state.worker.start()


def poll_pipeline_events(state: PipelineState) -> tuple[bool, bool]:
    if state.events is None:
        return False, False

    texture_dirty = False
    graph_dirty = False
    while True:
        try:
            tag, payload = state.events.get_nowait()
        except queue.Empty:
            break

        if tag == "status":
            state.status = str(payload)
            state.stage = str(payload)
        elif tag == "probability":
            state.probability = np.asarray(payload, dtype=np.float32)
            state.rebuild_mask()
            if state.mode == 1:
                texture_dirty = True
        elif tag == "mask":
            state.mask = np.asarray(payload, dtype=bool)
            if state.mode in (2, 3):
                texture_dirty = True
        elif tag == "graph":
            coords, conns, graph_threshold = payload
            state.graph_coords = coords
            state.graph_edges = conns
            state.graph_threshold = float(graph_threshold)
            graph_dirty = True
        elif tag == "error":
            state.error = str(payload)
            state.status = f"error: {payload}"
        elif tag == "finished":
            state.running = False

    return texture_dirty, graph_dirty


def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    yaw_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype="f4")
    pitch_m = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype="f4")
    return yaw_m @ pitch_m


def create_texture(ctx: moderngl.Context, volume: np.ndarray) -> moderngl.Texture3D:
    shape = volume.shape
    texture = ctx.texture3d((shape[2], shape[1], shape[0]), 1, volume.tobytes())
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False
    texture.repeat_z = False
    texture.use(0)
    return texture


def rebuild_graph_buffers(
    ctx: moderngl.Context,
    program: moderngl.Program,
    state: PipelineState,
    settings: dict[str, Any],
) -> GraphBuffers:
    buffers = GraphBuffers()
    if state.graph_coords is None or state.graph_edges is None or len(state.graph_coords) == 0:
        return buffers

    order_index = int(settings.get("graph_coord_order", 0))
    coord_order = GRAPH_COORD_ORDER_KEYS[order_index]
    view_coords = graph_coords_to_view(
        state.graph_coords,
        coord_order=coord_order,
        flip_x=bool(settings.get("graph_flip_x", False)),
        flip_y=bool(settings.get("graph_flip_y", False)),
        flip_z=bool(settings.get("graph_flip_z", False)),
    )

    buffers.point_count = len(view_coords)
    line_vertices = view_coords[state.graph_edges.reshape(-1)].astype("f4")
    buffers.line_count = len(line_vertices)
    buffers.point_buffer = ctx.buffer(view_coords.astype("f4").tobytes())
    buffers.line_buffer = ctx.buffer(line_vertices.tobytes())
    buffers.point_vao = ctx.vertex_array(program, [(buffers.point_buffer, "3f", "in_pos")])
    buffers.line_vao = ctx.vertex_array(program, [(buffers.line_buffer, "3f", "in_pos")])
    return buffers


def setup_imgui(disabled: bool):
    if disabled:
        return None
    if imgui is None or PygameProgrammableRenderer is None:
        print("imgui is not installed; keyboard controls are still available.")
        print("Install UI extras with: pip install imgui[pygame] PyOpenGL")
        return None
    try:
        imgui.create_context()
        imgui.style_colors_dark()
        return PygameProgrammableRenderer()
    except Exception as error:
        print(f"Could not start imgui: {error}")
        return None


def render_imgui_panel(state: PipelineState, args: argparse.Namespace, settings: dict[str, Any]) -> tuple[bool, bool]:
    texture_dirty = False
    graph_dirty = False

    imgui.set_next_window_position(12, 12, condition=imgui.FIRST_USE_EVER)
    imgui.set_next_window_size(360, 430, condition=imgui.FIRST_USE_EVER)
    imgui.begin("Pipeline")
    imgui.text(f"Status: {state.status}")
    imgui.text(f"Origin z/y/x: {state.origin}")
    imgui.text(f"Cube: {state.raw.shape[0]} x {state.raw.shape[1]} x {state.raw.shape[2]}")
    imgui.text(f"Pore fraction: {state.pore_fraction:.4f}")
    if state.error:
        imgui.text_wrapped(f"Error: {state.error}")
    imgui.separator()

    changed, mode = imgui.combo("View", state.mode, list(MODE_NAMES))
    if changed:
        state.mode = mode
        texture_dirty = True

    changed, threshold = imgui.slider_float("Mask threshold", float(state.threshold), 0.0, 1.0, "%.3f")
    if changed:
        state.threshold = float(threshold)
        state.rebuild_mask()
        texture_dirty = state.mode in (2, 3)

    changed, density = imgui.slider_float("Volume density", float(settings["density"]), 0.1, 16.0, "%.2f")
    if changed:
        settings["density"] = density

    changed, draw_threshold = imgui.slider_float(
        "Raw draw threshold",
        float(settings["draw_threshold"]),
        0.0,
        1.0,
        "%.3f",
    )
    if changed:
        settings["draw_threshold"] = draw_threshold

    imgui.separator()
    if imgui.button("Run full pipeline") and not state.running:
        start_pipeline_worker(state, args, settings["paths"], run_graph=True)
    imgui.same_line()
    if imgui.button("Graph only") and not state.running:
        start_graph_worker(state, args)

    if imgui.button("Raw"):
        state.mode = 0
        texture_dirty = True
    imgui.same_line()
    if imgui.button("Prob"):
        state.mode = 1
        texture_dirty = True
    imgui.same_line()
    if imgui.button("Mask"):
        state.mode = 2
        texture_dirty = True
    imgui.same_line()
    if imgui.button("Graph"):
        state.mode = 3
        texture_dirty = True

    imgui.separator()
    changed, graph_coord_order = imgui.combo(
        "Graph coords",
        int(settings["graph_coord_order"]),
        list(GRAPH_COORD_ORDER_LABELS),
    )
    if changed:
        settings["graph_coord_order"] = int(graph_coord_order)
        graph_dirty = True

    changed, flip_x = imgui.checkbox("Flip graph X", bool(settings["graph_flip_x"]))
    if changed:
        settings["graph_flip_x"] = bool(flip_x)
        graph_dirty = True
    changed, flip_y = imgui.checkbox("Flip graph Y", bool(settings["graph_flip_y"]))
    if changed:
        settings["graph_flip_y"] = bool(flip_y)
        graph_dirty = True
    changed, flip_z = imgui.checkbox("Flip graph Z", bool(settings["graph_flip_z"]))
    if changed:
        settings["graph_flip_z"] = bool(flip_z)
        graph_dirty = True

    imgui.separator()
    imgui.text(f"Probability: {'yes' if state.has_probability() else 'no'}")
    imgui.text(f"Mask: {'yes' if state.has_mask() else 'no'}")
    if state.has_graph():
        imgui.text(f"Graph pores: {len(state.graph_coords)}")
        imgui.text(f"Graph throats: {len(state.graph_edges)}")
        imgui.text(f"Graph threshold: {state.graph_threshold:.3f}")
    else:
        imgui.text("Graph: no")
    imgui.end()

    return texture_dirty, graph_dirty


def update_caption(state: PipelineState, settings: dict[str, Any]) -> None:
    graph_text = ""
    if state.has_graph():
        graph_text = f" | pores {len(state.graph_coords)} throats {len(state.graph_edges)}"
    pygame.display.set_caption(
        f"Micro-CT realtime pipeline | {MODE_NAMES[state.mode]} | {state.status} "
        f"| phi {state.pore_fraction:.3f} | threshold {state.threshold:.2f}{graph_text}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime ModernGL + pygame visualizer for the micro-CT pipeline.")
    parser.add_argument("--raw", type=Path, help="path to grayscale uint8 raw volume")
    parser.add_argument("--binary", type=Path, help="path to binary uint8 raw volume")
    parser.add_argument("--checkpoint", type=Path, help="segmentation checkpoint path")
    parser.add_argument("--shape", type=parse_zyx, default=DEFAULT_SHAPE, help="full volume shape z,y,x")
    parser.add_argument("--origin", type=parse_zyx, help="subcube start coordinate z,y,x")
    parser.add_argument("--sample-index", type=int, default=-1, help="take origin from dataset_128/index_128.csv")
    parser.add_argument("--cube-size", type=int, default=64)
    parser.add_argument("--mask-source", choices=("auto", "model", "target", "raw"), default="auto")
    parser.add_argument(
        "--graph-coord-order",
        choices=GRAPH_COORD_ORDER_KEYS,
        default="zyx",
        help="coordinate order returned by pore.coords before visualization",
    )
    parser.add_argument("--flip-graph-x", action="store_true")
    parser.add_argument("--flip-graph-y", action="store_true")
    parser.add_argument("--flip-graph-z", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5, help="probability threshold for pore mask")
    parser.add_argument("--raw-threshold", type=float, default=0.45, help="fallback raw threshold for --mask-source raw")
    parser.add_argument("--pore-value", type=int, default=0)
    parser.add_argument("--voxel-size", type=float, default=2.25e-6)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--r-max", type=int, default=4)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--no-auto-run", action="store_true", help="show raw cube first and wait for UI button")
    parser.add_argument("--no-imgui", action="store_true", help="disable imgui panel")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="torch device")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=850)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state, paths = prepare_state(args)

    pygame.init()
    pygame.display.set_mode(
        (args.width, args.height),
        pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
    )

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    imgui_impl = setup_imgui(args.no_imgui)
    volume_program = ctx.program(vertex_shader=VOLUME_VERTEX_SHADER, fragment_shader=VOLUME_FRAGMENT_SHADER)
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"))
    quad_vao = ctx.vertex_array(volume_program, [(quad, "2f", "in_pos")])
    volume_program["volume_tex"].value = 0
    volume_program["volume_shape"].value = (args.cube_size, args.cube_size, args.cube_size)

    graph_program = ctx.program(vertex_shader=GRAPH_VERTEX_SHADER, fragment_shader=GRAPH_FRAGMENT_SHADER)
    graph_buffers = GraphBuffers()
    texture = create_texture(ctx, state.current_volume())

    settings: dict[str, Any] = {
        "density": 5.0,
        "draw_threshold": 0.08,
        "graph_coord_order": GRAPH_COORD_ORDER_KEYS.index(args.graph_coord_order),
        "graph_flip_x": bool(args.flip_graph_x),
        "graph_flip_y": bool(args.flip_graph_y),
        "graph_flip_z": bool(args.flip_graph_z),
        "paths": paths,
    }

    yaw, pitch, zoom = -0.65, 0.45, 2.0
    dragging = False
    last_mouse = (0, 0)
    texture_dirty = False

    print("Controls:")
    print("  1 raw, 2 probability, 3 mask, 4 graph")
    print("  mouse drag rotate, wheel zoom, +/- density, </> raw draw threshold, R reset, Esc quit")
    print("  imgui: run full pipeline, graph only, threshold/density sliders")

    if not args.no_auto_run:
        start_pipeline_worker(state, args, paths, run_graph=not args.skip_graph)

    clock = pygame.time.Clock()
    running = True
    while running:
        capture_mouse = False
        capture_keyboard = False
        if imgui_impl is not None:
            io = imgui.get_io()
            capture_mouse = io.want_capture_mouse
            capture_keyboard = io.want_capture_keyboard

        for event in pygame.event.get():
            if imgui_impl is not None:
                imgui_impl.process_event(event)

            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and not capture_keyboard:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    state.mode = event.key - pygame.K_1
                    texture_dirty = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    settings["density"] = min(16.0, float(settings["density"]) * 1.15)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    settings["density"] = max(0.1, float(settings["density"]) / 1.15)
                elif event.key in (pygame.K_COMMA, pygame.K_LEFTBRACKET):
                    settings["draw_threshold"] = max(0.0, float(settings["draw_threshold"]) - 0.02)
                elif event.key in (pygame.K_PERIOD, pygame.K_RIGHTBRACKET):
                    settings["draw_threshold"] = min(1.0, float(settings["draw_threshold"]) + 0.02)
                elif event.key == pygame.K_r:
                    yaw, pitch, zoom = -0.65, 0.45, 2.0
                elif event.key == pygame.K_SPACE:
                    start_pipeline_worker(state, args, paths, run_graph=not args.skip_graph)
                elif event.key == pygame.K_g:
                    start_graph_worker(state, args)
            elif event.type == pygame.MOUSEBUTTONDOWN and not capture_mouse:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    zoom = max(0.7, zoom * 0.9)
                elif event.button == 5:
                    zoom = min(5.5, zoom * 1.1)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging and not capture_mouse:
                x, y = event.pos
                dx, dy = x - last_mouse[0], y - last_mouse[1]
                last_mouse = event.pos
                yaw += dx * 0.008
                pitch = max(-1.45, min(1.45, pitch + dy * 0.008))
            elif event.type == pygame.MOUSEWHEEL and not capture_mouse:
                zoom = max(0.7, min(5.5, zoom * (0.9 if event.y > 0 else 1.1)))
            elif event.type == pygame.VIDEORESIZE:
                ctx.viewport = (0, 0, event.w, event.h)

        event_texture_dirty, event_graph_dirty = poll_pipeline_events(state)
        texture_dirty = texture_dirty or event_texture_dirty
        if event_graph_dirty:
            graph_buffers = rebuild_graph_buffers(ctx, graph_program, state, settings)

        if imgui_impl is not None:
            if hasattr(imgui_impl, "process_inputs"):
                imgui_impl.process_inputs()
            width, height = pygame.display.get_window_size()
            imgui.get_io().display_size = (float(width), float(height))
            imgui.new_frame()
            ui_texture_dirty, ui_graph_dirty = render_imgui_panel(state, args, settings)
            texture_dirty = texture_dirty or ui_texture_dirty
            if ui_graph_dirty:
                graph_buffers = rebuild_graph_buffers(ctx, graph_program, state, settings)

        if texture_dirty:
            texture.write(state.current_volume().tobytes())
            texture_dirty = False

        width, height = pygame.display.get_window_size()
        ratio = width / max(height, 1)
        rot = rotation_matrix(yaw, pitch)
        ctx.viewport = (0, 0, width, height)
        ctx.clear(0.015, 0.017, 0.022)

        volume_program["camera_rot"].write(rot.tobytes())
        volume_program["zoom"].value = zoom
        volume_program["density"].value = (
            float(settings["density"]) if state.mode != 3 else max(1.2, float(settings["density"]) * 0.45)
        )
        volume_program["threshold"].value = float(settings["draw_threshold"]) if state.mode == 0 else 0.02
        volume_program["window_ratio"].value = ratio
        volume_program["palette_mode"].value = min(state.mode, 2)
        quad_vao.render(moderngl.TRIANGLE_STRIP)

        if state.mode == 3 and graph_buffers.line_vao is not None and graph_buffers.point_vao is not None:
            graph_program["camera_rot"].write(rot.tobytes())
            graph_program["zoom"].value = zoom
            graph_program["window_ratio"].value = ratio
            graph_program["point_size"].value = 6.5
            graph_program["color"].value = (0.08, 0.78, 1.0, 0.75)
            graph_buffers.line_vao.render(moderngl.LINES, vertices=graph_buffers.line_count)
            graph_program["color"].value = (1.0, 0.82, 0.18, 0.95)
            graph_buffers.point_vao.render(moderngl.POINTS, vertices=graph_buffers.point_count)

        if imgui_impl is not None:
            imgui.render()
            imgui_impl.render(imgui.get_draw_data())

        update_caption(state, settings)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
