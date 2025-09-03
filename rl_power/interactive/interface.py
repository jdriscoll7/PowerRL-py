import os

os.environ.setdefault("MPLBACKEND", "Agg")
from concurrent.futures import ProcessPoolExecutor, TimeoutError as _Timeout
import multiprocessing as mp
from io import BytesIO
from PIL import Image

import traceback, dill
import gradio as gr
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# from rl_power.envs.edge_agent_branch_env import EdgeAgentBranchEnv
# from rl_power.power.graph_utils import get_adjacent_branches
# from rl_power.power.powermodels_interface import load_test_case, solve_opf
# from rl_power.training.a2c_tester import A2CBranchTester


_EXEC = None


def run_in_proc(fn, *args, timeout_s=30, **kwargs):
    fut = _EXEC.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout_s)
    except _Timeout:
        fut.cancel()
        raise gr.Error(f"Operation timed out after {timeout_s}s")
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Operation failed: {e}")


def _load_test_case_worker(abs_path: str) -> bytes:
    """Runs in a separate process; returns dill-serialized network object."""
    from rl_power.power.powermodels_interface import load_test_case, solve_opf
    net = load_test_case(abs_path)
    return dill.dumps(net)


def _build_tester_worker(net_bytes: bytes, model_dir: str):
    global _TESTER
    from rl_power.training.a2c_tester import A2CBranchTester
    net = dill.loads(net_bytes)
    tester = A2CBranchTester(net, model_dir)
    _TESTER = tester
    # return dill.dumps(tester)


def list_networks():
    path = "./ieee_data/"
    return [file for file in os.listdir(path)]


def load_network(net_path):
    print(f"loading {net_path}")
    abs_path = os.path.abspath("ieee_data/" + net_path)
    net_bytes = run_in_proc(_load_test_case_worker, abs_path, timeout_s=60)
    _network = dill.loads(net_bytes)
    # _network = load_test_case(os.path.abspath("ieee_data/" + "pglib_opf_case118_ieee.m"))
    buses = list(_network["bus"].keys())
    n_buses = len(buses)
    # valid_buses = [b for b in buses if len(get_adjacent_branches(_network, b)[0]) > 2]
    valid_buses = buses

    return _network, gr.Dropdown(choices=valid_buses, interactive=True)


def load_selectable_edges(_network, node_id):
    from rl_power.power.graph_utils import get_adjacent_branches
    return gr.Dropdown(choices=get_adjacent_branches(_network, [node_id])[0], interactive=True, multiselect=True)


def _reset_tester_worker():
    global _TESTER
    _TESTER.env.reset()


def reset_tester(tester):
    run_in_proc(_reset_tester_worker, timeout_s=60)


def load_tester(_net_selection, _model_directory_selection):
    print("tester loading")
    # print(_net_selection)

    if _model_directory_selection is None or _net_selection is None:
        print("Waiting for selections to load tester.")
    else:
        # tester = A2CBranchTester(_net_selection, _model_directory_selection)
        net_bytes = dill.dumps(_net_selection)  # send to worker
        run_in_proc(_build_tester_worker, net_bytes, _model_directory_selection, timeout_s=60)

    print("tester loaded")


def _policy_run_to_frame_worker(edge_selection):
    global _TESTER
    _TESTER.test_step(agents=edge_selection)
    buf = BytesIO()
    _TESTER.current_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close('all')
    return buf.getvalue()


def policy_run_to_frame(edge_selection):

    png_bytes = run_in_proc(_policy_run_to_frame_worker, edge_selection)
    im = Image.open(BytesIO(png_bytes)).convert("RGBA")
    return im


def append_frame(frames, idx, edge_selection):
    """Append a new frame to the list and move index to the newest frame."""
    # Build a new frame from current selections
    print(edge_selection)
    fig = policy_run_to_frame(edge_selection)
    print(fig)
    frames = (frames or []) + [fig]
    idx = len(frames) - 1  # jump to newest
    return frames, idx, frames[idx]


def show_index(frames, idx):
    if not frames:
        return None
    idx = max(0, min(len(frames) - 1, int(idx)))
    return frames[idx]


def shift(frames, idx, delta):
    if not frames:
        return 0, None
    idx = max(0, min(len(frames) - 1, int(idx) + delta))
    return idx, frames[idx]


if __name__ == "__main__":

    # Create the pool ONLY here (avoid recursive spawn on import)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ctx = mp.get_context("spawn")
    _EXEC = ProcessPoolExecutor(max_workers=1, mp_context=ctx)

    with gr.Blocks(title="Demo") as demo:
        network = gr.State(value=None)
        frames_state = gr.State(value=[])
        idx_state = gr.State(value=0)
        # tester_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                model_input = gr.FileExplorer(root_dir="./results", label="Pick model file", height=200,
                                              file_count='single')
                net_input = gr.Dropdown(list_networks(), label="Network", value=None)
                run_btn = gr.Button("Run policy", variant="primary")

            with gr.Column(scale=2):
                node_selector = gr.Dropdown(label="Node", allow_custom_value=True)
                edge_selector = gr.Dropdown(label="Edge", allow_custom_value=True, multiselect=True)
                with gr.Row():
                    prev_btn = gr.Button("⟵ Prev", scale=1)
                    idx_display = gr.Number(label="Frame index", value=0, precision=0, interactive=True, scale=1)
                    next_btn = gr.Button("Next ⟶", scale=1)

                frame_plot = gr.Image(label="Frame", type="pil")

        # When the network changes, update the info panel (no slider involved)
        input_event = net_input.change(fn=load_network, inputs=net_input, outputs=[network, node_selector])
        # Changes related to tester object.
        model_event = model_input.change(fn=load_tester, inputs=[network, model_input], outputs=None)
        input_event.then(fn=load_tester, inputs=[network, model_input], outputs=None)

        node_selector.change(fn=load_selectable_edges, inputs=[network, node_selector], outputs=edge_selector)
        edge_selector.change(fn=reset_tester, inputs=None, outputs=None)

        run_btn.click(fn=append_frame,
                      inputs=[frames_state, idx_state, edge_selector],
                      outputs=[frames_state, idx_state, frame_plot]
                      ).then(lambda i: i, inputs=[idx_state], outputs=[idx_display])

        idx_display.change(fn=show_index,
                           inputs=[frames_state, idx_display],
                           outputs=[frame_plot]
                           ).then(lambda i: i, inputs=[idx_display], outputs=[idx_state])

        # E) Prev/Next buttons
        prev_btn.click(fn=lambda f, i: shift(f, i, -1),
                       inputs=[frames_state, idx_state],
                       outputs=[idx_state, frame_plot]
                       ).then(lambda i: i, inputs=[idx_state], outputs=[idx_display])

        next_btn.click(fn=lambda f, i: shift(f, i, +1),
                       inputs=[frames_state, idx_state],
                       outputs=[idx_state, frame_plot]
                       ).then(lambda i: i, inputs=[idx_state], outputs=[idx_display])

    # __network = load_test_case(os.path.abspath("ieee_data/" + "pglib_opf_case118_ieee.m"))
    # dummy = A2CBranchTester(test_env_path=__network, actor_critic_directory=None)
    # solve_opf(__network)
    # plt.close('all')
    demo.launch(show_error=True, debug=True, max_threads=1)
    # demo.launch(debug=True, max_threads=1)
