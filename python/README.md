# Python Bindings for SVN-ICP

Use the SVN-ICP core library from Python — **no ROS installation required**.

The bindings expose:
- `SteinICPParam` / `ParticleWeightOpt` — configuration structs
- `SVNICP` — the main ICP alignment class
- `VoxelHashMap` — a voxel-hashed local-map data structure
- Helpers for particle initialisation and coordinate conversions

---

## Prerequisites

| Dependency | Notes |
|---|---|
| **CUDA** | Any version supported by your LibTorch build |
| **PyTorch** | `pip install torch` — provides LibTorch and `torch/extension.h` |
| **PCL** | Point Cloud Library (with VTK visualisation support) |
| **GTSAM** | Build and install from source |
| **Eigen 3** | Usually available as a system package |
| **TBB** | Intel Threading Building Blocks |
| **tsl-robin-map** | `git clone` + `cmake --install` |
| **pybind11** | `pip install pybind11` |
| **CMake ≥ 3.18** | |
| **C++17 compiler** | GCC ≥ 9 or Clang ≥ 10 |

---

## Build

```bash
# 1. Clone the repository (if you haven't already)
git clone https://github.com/LIS-TU-Berlin/SVN-ICP.git
cd SVN-ICP/python

# 2. Configure
#    TORCH_DIR and pybind11_DIR are auto-detected from the active Python
#    environment when omitted.  Pass them explicitly if auto-detection fails.
mkdir -p build && cd build
cmake .. \
    [-DTORCH_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")] \
    [-Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")] \
    -DCMAKE_BUILD_TYPE=Release

# 3. Build
make -j$(nproc)

# 4. (Optional) install to site-packages
make install
# or simply copy / symlink the .so into your project:
#   cp pysvnicp*.so /path/to/your/project/
```

The build produces two artefacts inside `build/`:
- `libsvnicp_core.so` — ROS-free shared library with the C++ algorithm
- `pysvnicp*.so` — Python extension module linked against it

---

## Minimal usage example

```python
import numpy as np
import torch
import pysvnicp

# ── 1. Algorithm parameters ────────────────────────────────────────────────
param = pysvnicp.SteinICPParam()
param.iterations          = 30
param.batch_size          = 200
param.lr                  = 0.02
param.max_dist            = 1.0
param.SVN_full_grad       = True
param.check_early_stop    = True
param.convergence_threshold = 1e-5
param.KNN_count           = 5

# ── 2. Particle initialisation ─────────────────────────────────────────────
n_particles = 50
# Bounds: [tx, ty, tz, rx, ry, rz]
ub = torch.tensor([0.5, 0.5, 0.5, 0.05, 0.05, 0.05],
                  dtype=torch.float64, device="cuda")
lb = -ub
init_pose = pysvnicp.initialize_particles(n_particles, cuda=True, ub=ub, lb=lb)
# shape: (6, 50, 1), float64, CUDA

# For a Gaussian initialisation (e.g., from an IMU prediction):
# cov = [var_tx, var_ty, var_tz, var_rx, var_ry, var_rz]
# init_pose = pysvnicp.initialize_particles_gaussian(n_particles, cuda=True,
#                 cov=[0.01, 0.01, 0.01, 0.001, 0.001, 0.001])

# ── 3. Create SVNICP ───────────────────────────────────────────────────────
opt = pysvnicp.ParticleWeightOpt()
icp = pysvnicp.SVNICP(param, init_pose, opt)

# ── 4. Load point clouds ───────────────────────────────────────────────────
# source_np and target_np are Nx3 float64 numpy arrays (x, y, z)
source_np = np.load("source_cloud.npy").astype(np.float64)   # shape (N, 3)
target_np = np.load("target_cloud.npy").astype(np.float64)   # shape (M, 3)

# Convert to CUDA tensors
source_t = pysvnicp.numpy_to_tensor(source_np)  # (N, 3) float64 CUDA
target_t = pysvnicp.numpy_to_tensor(target_np)  # (M, 3) float64 CUDA

# ── 5. Register ────────────────────────────────────────────────────────────
icp.add_cloud(source_t, target_t, init_pose)
state = icp.stein_align()                   # releases the GIL
assert state == pysvnicp.SteinICPState.ALIGN_SUCCESS

# ── 6. Results ─────────────────────────────────────────────────────────────
# 6-element tensor [tx, ty, tz, rx, ry, rz] — axis-angle rotation
pose_vec = icp.get_transformation()

# 4x4 SE(3) numpy matrix
T_4x4 = pysvnicp.tensor_to_pose(pose_vec)
print("Estimated transform:\n", T_4x4)

# Full 6x6 covariance (flat row-major list of 36 floats)
cov_flat = icp.get_cov_matrix()
cov_6x6  = np.array(cov_flat).reshape(6, 6)
print("Pose covariance:\n", cov_6x6)

# Per-DOF variance (quick diagnostic)
variance = pysvnicp.tensor_to_numpy(icp.get_distribution())
print("Per-DOF variance:", variance)

# Runtime breakdown
knn_s, update_s, n_iters = icp.get_runtime()
print(f"KNN: {knn_s*1e3:.1f} ms | update: {update_s*1e3:.1f} ms | iters: {n_iters}")
```

---

## Multi-frame odometry with VoxelHashMap

```python
import numpy as np
import torch
import pysvnicp

param = pysvnicp.SteinICPParam()
param.iterations = 30
param.SVN_full_grad = True
param.KNN_count = 5

opt = pysvnicp.ParticleWeightOpt()

# Local map: 1 m voxels, 80 m range, up to 20 points per voxel
vmap = pysvnicp.VoxelHashMap(voxel_size=1.0, max_range=80.0, max_points=20)

current_pose = np.eye(4)   # running SE(3) estimate (map <- body)

for scan_np in load_scans():          # scan_np: Nx3 float32 numpy array
    if vmap.empty():
        # Seed the map with the first scan
        vmap.add_point_cloud(scan_np, current_pose)
        continue

    # Retrieve local map points near the current pose estimate
    target_np = vmap.get_map_near_pose(current_pose, max_range=50.0)  # Nx4 f32
    target_t  = pysvnicp.numpy_to_tensor(target_np[:, :3].astype(np.float64))
    source_t  = pysvnicp.numpy_to_tensor(scan_np[:, :3].astype(np.float64))

    # Initialise particles around the identity (or your motion-model prediction)
    ub        = torch.tensor([0.3, 0.3, 0.3, 0.03, 0.03, 0.03],
                              dtype=torch.float64, device="cuda")
    init_pose = pysvnicp.initialize_particles(50, cuda=True, ub=ub, lb=-ub)

    # Align
    icp = pysvnicp.SVNICP(param, init_pose, opt)
    icp.add_cloud(source_t, target_t, init_pose)
    icp.stein_align()

    # Compose pose
    delta = pysvnicp.tensor_to_pose(icp.get_transformation())  # 4x4 numpy
    current_pose = current_pose @ delta

    # Update map with the new scan (in map frame)
    vmap.add_point_cloud(scan_np, current_pose)

    cov = np.array(icp.get_cov_matrix()).reshape(6, 6)
    print("pose:", current_pose[:3, 3], "| trace(cov):", np.trace(cov))
```

---

## API reference

### Free functions

| Function | Description |
|---|---|
| `initialize_particles(n, cuda, ub, lb)` | Uniform particle sampling in [lb, ub] → (6, n, 1) tensor |
| `initialize_particles_gaussian(n, cuda, cov)` | Gaussian particle sampling → (6, n, 1) tensor |
| `numpy_to_tensor(array)` | Nx3 float64 numpy → CUDA torch.Tensor |
| `tensor_to_numpy(tensor)` | torch.Tensor → float64 numpy array |
| `tensor_to_pose(tensor)` | 6-elem pose tensor → 4×4 SE(3) numpy |
| `pose_to_tensor(pose)` | 4×4 SE(3) numpy → 6-elem CUDA tensor |

### `SteinICPParam` fields

| Field | Default | Description |
|---|---|---|
| `iterations` | 50 | Max gradient iterations |
| `use_minibatch` | False | Enable random mini-batching |
| `batch_size` | 50 | Points per mini-batch |
| `lr` | 0.02 | Step size |
| `max_dist` | 1.0 | Correspondence distance threshold |
| `optimizer` | `"Adam"` | `"Adam"` / `"RMSprop"` / `"SGD"` / `"Adagrad"` |
| `check_early_stop` | False | Enable gradient-norm stopping |
| `convergence_threshold` | 1e-5 | Threshold for early stop |
| `KNN_count` | 100 | Nearest neighbours per source point |
| `SVN_full_grad` | True | Full SVN (True) vs preconditioned SVGD (False) |

### `SVNICP` methods

| Method | Description |
|---|---|
| `add_cloud(source, target, init_pose)` | Set clouds and reset particles |
| `stein_align()` → `SteinICPState` | Run alignment (releases GIL) |
| `get_transformation()` → `Tensor` | Weighted-mean 6-DOF pose |
| `get_distribution()` → `Tensor` | Per-DOF variance |
| `get_cov_matrix()` → `list[float]` | Flat 6×6 covariance (36 values) |
| `get_particles()` → `list[float]` | All 6·N particle values |
| `get_particle_weight()` → `list[float]` | N particle weights |
| `get_runtime()` → `list[float]` | `[knn_s, update_s, n_iters]` |
| `set_k(k)` | Override KNN count |
| `set_threshold(max_dist)` | Override distance threshold |

### `VoxelHashMap` methods

| Method | Description |
|---|---|
| `add_point_cloud(cloud, pose)` | Insert Nx3/Nx4 float32 scan at SE(3) pose |
| `get_map()` | All map points as Nx4 float32 |
| `get_map_near_pose(pose, max_range)` | Local map within radius as Nx4 float32 |
| `get_neighbour_map(cloud)` | Adjacent voxels as Nx4 float32 |
| `clear()` | Remove all stored points |
| `empty()` | True if map is empty |
| `size()` | Number of occupied voxels |
