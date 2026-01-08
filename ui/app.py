import os, sys, uuid, time, json
from pathlib import Path

import streamlit as st

# ------- Config & Model Registry -------

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root:", project_root)

project_root = Path(project_root) / "earth-api"

# enter the folder to run the app
st.sidebar.text_input("Project Root", value=str(project_root))

from eintelligence.data_prep.aoi import square_aoi
from orchestrator.workflow_manager_multisensor_v1 import (
    DeforestationWorkflowMS, TilingConfigMS, TilingConfigS1, TrainingConfig,
    FloodWorkflowS1
)

st.set_page_config(page_title="Earth Intelligence Platform", layout="wide")
st.title("Earth - API")
st.caption("Backbone: SSL4EO - lite | Tasks: Deforestation Detection (Multi-sensor - S1+S2), Deforestation Detection (S2-only), Flood Change Detection (S1-only)")

# ------- Sidebar -------
task = st.sidebar.selectbox("Task", ["deforestation", "flood"])

if task == "deforestation":
    sensor_mode = st.sidebar.selectbox("Sensor Mode", ["s2", "s1s2"])

elif task == "flood":
    sensor_mode = st.sidebar.selectbox("Sensor Mode", ["s1"])


st.sidebar.subheader("AOI Configuration")
region_name = st.sidebar.text_input("Region Name", value="munich")

lat = st.sidebar.number_input("Latitude", value=48.1351, format="%.6f")
lon = st.sidebar.number_input("Longitude", value=11.5820, format="%.6f")

st.sidebar.subheader("Temporal Configuration")
start = st.sidebar.text_input("Start Date (YYYY-MM-DD)", value="2023-06-01")
end = st.sidebar.text_input("End Date (YYYY-MM-DD)", value="2023-08-01")

max_cloud =  20
if task == "deforestation" and sensor_mode in ("s2", "s1s2"):
    max_cloud = st.sidebar.slider("Max Cloud Cover (%)", min_value=0, max_value=100, value=20, step=5)
    


# ------- Main App -------

log_box = st.empty()

def log(message: str):
    log_box.write(message)


st.subheader("Mode Selection")
mode = st.radio("Mode", ["Training", "Prediction"])

skip_to_pairing = st.checkbox("Skip to Pairing (Skips Data Collection)", value=False)

st.subheader("Detailed Configuration")

st.markdown("### Tiling Parameters")

if st.checkbox("Configure Tiling Parameters"):
    tile_size = st.selectbox("Tile Size", [128, 256, 512], index=1)
    stride = st.selectbox("Stride", [128, 256, 512], index=1)
else:
    tile_size = 256
    stride = 256

st.markdown("### Training Parameters")

if mode == "Training":
    retrain = True
    num_epochs = st.number_input("Number of Training Epochs", min_value=1, max_value=100, value=10, step=1)
    batch_size = st.selectbox("Batch Size", [2, 4, 8, 16], index=1)
    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.6f", step=1e-4)
    amp = st.checkbox("Use Automatic Mixed Precision (AMP)", value=True)

elif mode == "Prediction":
    retrain = False
    num_epochs = 1
    batch_size = 4
    learning_rate = 1e-3
    amp = True

st.markdown("---")

if task == "deforestation":
    case_name = "deforestation"

    tiling_cfg = TilingConfigMS(
        bands_s2=("B02","B03","B04","B08"),
        bands_s1=("vv","vh"),
        tile_size=tile_size, stride=stride, max_cloud=max_cloud,
        sensor_mode=sensor_mode
    )

    train_cfg  = TrainingConfig(batch_size=batch_size, num_epochs=num_epochs, lr=learning_rate, amp=amp)

    wf = DeforestationWorkflowMS(project_root, tiling_cfg, train_cfg, skip_to_pairing=skip_to_pairing)

elif task == "flood":
    case_name = "flood"
    tiling_cfg = TilingConfigS1(tile_size=tile_size, stride=stride)

    train_cfg  = TrainingConfig(batch_size=batch_size, num_epochs=num_epochs, lr=learning_rate, amp=amp)
    
    wf = FloodWorkflowS1(project_root, tiling_cfg, train_cfg, skip_to_pairing=skip_to_pairing)

region_id = f"{region_name}_{case_name}_{sensor_mode}"
aoi = square_aoi(lat, lon)

st.subheader("Execute Workflow")

if st.button("Run"):

    region_dir = Path(project_root) / "data" / region_id
    if skip_to_pairing and not region_dir.exists():
        st.error(f"Region folder not found: {region_dir}. Turn off skip_to_pairing or build once.")
        st.stop()
    
    log(f"Building temporal pairs for AOI: {aoi}, Start: {start}, End: {end}, Region: {region_name}...")
    pairs_manifest = wf.build_data(
        aoi_geojson=aoi,
        start=start,
        end=end,
        region_name=region_id
    )

    ckpt_path = Path(project_root) / "models" / f"{case_name}_{sensor_mode}_adapter.pt"
    out_dir   = Path(project_root) / "data" / region_id / f"pred_{case_name}_{sensor_mode}"
    log(f"Running workflow with checkpoint: {ckpt_path}, Output Directory: {out_dir}...")

    wf.run(pairs_manifest, ckpt_path, out_dir, retrain=retrain, prob_thresh=0.5)
    log("Workflow execution completed.")
        

