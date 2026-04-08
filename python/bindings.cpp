/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 SVN-ICP Authors
    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    bindings.cpp
 * @brief   pybind11 Python bindings for the SVN-ICP core library.
 *
 * Exposes the following to Python (module name: pysvnicp):
 *   - SteinICPParam, ParticleWeightOpt, SteinICPState
 *   - SVNICP class
 *   - VoxelHashMap class
 *   - Helper functions: initialize_particles, initialize_particles_gaussian,
 *                       numpy_to_tensor, tensor_to_numpy,
 *                       tensor_to_pose, pose_to_tensor
 */

// torch/extension.h must come first; it includes pybind11 and registers
// automatic type casters for torch::Tensor <-> Python torch.Tensor.
#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Eigen>
#include <gtsam/geometry/Pose3.h>

#include "core/SVNICP.h"
#include "core/VoxelHashMap.h"
#include "core/ICPUtils.h"
#include "data/DataTypes.h"

namespace py = pybind11;
using namespace svnicp;
using namespace data_types;

// ---------------------------------------------------------------------------
// Internal conversion helpers
// ---------------------------------------------------------------------------

/// Convert an Nx3 or Nx4 float32 numpy array to a PCL PointXYZI cloud.
static pcl::PointCloud<Point_t>
numpy_to_pcl(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2 || (buf.shape[1] != 3 && buf.shape[1] != 4))
        throw std::invalid_argument(
            "Expected Nx3 or Nx4 float32 array for point cloud.");

    const auto* ptr = static_cast<const float*>(buf.ptr);
    const int ncols = static_cast<int>(buf.shape[1]);

    pcl::PointCloud<Point_t> cloud;
    cloud.reserve(static_cast<size_t>(buf.shape[0]));
    for (ssize_t i = 0; i < buf.shape[0]; ++i) {
        Point_t pt;
        pt.x         = ptr[i * ncols + 0];
        pt.y         = ptr[i * ncols + 1];
        pt.z         = ptr[i * ncols + 2];
        pt.intensity = (ncols > 3) ? ptr[i * ncols + 3] : 0.f;
        cloud.push_back(pt);
    }
    return cloud;
}

/// Convert a PCL PointXYZI cloud to an Nx4 float32 numpy array [x, y, z, i].
static py::array_t<float>
pcl_to_numpy(const pcl::PointCloud<Point_t>& cloud)
{
    py::array_t<float> arr({static_cast<ssize_t>(cloud.size()), ssize_t{4}});
    auto buf = arr.request();
    auto* ptr = static_cast<float*>(buf.ptr);
    for (size_t i = 0; i < cloud.size(); ++i) {
        ptr[i * 4 + 0] = cloud[i].x;
        ptr[i * 4 + 1] = cloud[i].y;
        ptr[i * 4 + 2] = cloud[i].z;
        ptr[i * 4 + 3] = cloud[i].intensity;
    }
    return arr;
}

/// Convert a 4x4 float64 numpy SE(3) matrix to gtsam::Pose3.
static gtsam::Pose3
numpy_to_pose3(py::array_t<double, py::array::c_style | py::array::forcecast> mat)
{
    auto buf = mat.request();
    if (buf.ndim != 2 || buf.shape[0] != 4 || buf.shape[1] != 4)
        throw std::invalid_argument("Expected 4x4 float64 pose matrix.");

    const auto* ptr = static_cast<const double*>(buf.ptr);
    Eigen::Matrix3d R;
    R << ptr[0], ptr[1], ptr[2],
         ptr[4], ptr[5], ptr[6],
         ptr[8], ptr[9], ptr[10];
    Eigen::Vector3d t(ptr[3], ptr[7], ptr[11]);
    return gtsam::Pose3(gtsam::Rot3(R), t);
}

// ---------------------------------------------------------------------------
// Module-level Python helpers (also exposed to Python)
// ---------------------------------------------------------------------------

/**
 * Convert an Nx3 float64 numpy array to a CUDA torch::Tensor of shape (N,3).
 * The resulting tensor is a clone (owns its storage) placed on CUDA.
 */
static torch::Tensor
numpy_to_tensor(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw std::invalid_argument("Expected Nx3 float64 numpy array.");

    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    return torch::from_blob(buf.ptr,
                            {static_cast<long>(buf.shape[0]), 3L},
                            opts)
               .clone()
               .to(torch::kCUDA);
}

/**
 * Convert a torch::Tensor (any device, any dtype) to a float64 numpy array.
 * The tensor is moved to CPU and cast to float64 before copying.
 */
static py::array_t<double>
tensor_to_numpy(const torch::Tensor& t)
{
    const auto cpu = t.to(torch::kCPU).to(torch::kFloat64).contiguous();
    std::vector<ssize_t> shape, strides;
    for (int i = 0; i < cpu.dim(); ++i) {
        shape.push_back(static_cast<ssize_t>(cpu.size(i)));
        strides.push_back(static_cast<ssize_t>(cpu.stride(i)) *
                          static_cast<ssize_t>(sizeof(double)));
    }
    return py::array_t<double>(shape, strides, cpu.data_ptr<double>());
}

/**
 * Convert a 6-element pose tensor [tx, ty, tz, rx, ry, rz] (angle-axis
 * rotation) to a 4x4 SE(3) float64 numpy matrix.
 *
 * Uses ICPUtils::tensor2Matrix which treats [rx, ry, rz] as the axis-angle
 * rotation vector composed via ZYX Euler angles.
 */
static py::array_t<double>
tensor_to_pose(const torch::Tensor& tensor)
{
    const auto mat4 = tensor2Matrix(tensor.to(torch::kCPU));
    py::array_t<double> result({4, 4});
    auto buf = result.request();
    auto* ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            ptr[i * 4 + j] = mat4(i, j);
    return result;
}

/**
 * Convert a 4x4 SE(3) float64 numpy matrix to a 6-element CUDA tensor
 * [tx, ty, tz, rx, ry, rz] where [rx, ry, rz] is the SO(3) logarithm
 * (axis-angle) of the rotation.  This is the pose format expected by
 * SVNICP::add_cloud and initialize_particles.
 */
static torch::Tensor
pose_to_tensor(py::array_t<double, py::array::c_style | py::array::forcecast> mat_np)
{
    const gtsam::Pose3 pose = numpy_to_pose3(mat_np);
    const gtsam::Vector3 rot_vec = gtsam::Rot3::Logmap(pose.rotation());
    const gtsam::Point3  trans   = pose.translation();
    const std::vector<double> v = {
        trans.x(), trans.y(), trans.z(),
        rot_vec.x(), rot_vec.y(), rot_vec.z()
    };
    return torch::tensor(v, torch::TensorOptions().dtype(torch::kFloat64))
               .to(torch::kCUDA);
}

// ---------------------------------------------------------------------------
// pybind11 module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(pysvnicp, m)
{
    m.doc() = "pysvnicp: Python bindings for the SVN-ICP core library. "
              "No ROS installation is required.";

    // ------------------------------------------------------------------
    // CovFilterType enum
    // ------------------------------------------------------------------
    py::enum_<CovFilterType>(m, "CovFilterType",
        "Strategy used to filter the output covariance across frames.")
        .value("MEAN",               CovFilterType::MEAN)
        .value("MAX_SLIDING_WINDOW", CovFilterType::MAX_SLIDING_WINDOW)
        .value("NONE",               CovFilterType::NONE)
        .export_values();

    // ------------------------------------------------------------------
    // SteinICPState enum
    // ------------------------------------------------------------------
    py::enum_<SteinICPState>(m, "SteinICPState",
        "Return code from stein_align().")
        .value("ALIGN_SUCCESS", SteinICPState::ALIGN_SUCCESS)
        .value("NO_OPTIMIZER",  SteinICPState::NO_OPTIMIZER)
        .export_values();

    // ------------------------------------------------------------------
    // SteinICPParam
    // ------------------------------------------------------------------
    py::class_<SteinICPParam>(m, "SteinICPParam",
        "Configuration parameters for the Stein-ICP algorithm. "
        "All fields are read/write; construct with defaults then override.")
        .def(py::init<>())
        .def_readwrite("iterations",            &SteinICPParam::iterations,
            "Maximum number of Stein-gradient iterations.")
        .def_readwrite("use_minibatch",         &SteinICPParam::use_minibatch,
            "Use random mini-batches instead of the full source cloud.")
        .def_readwrite("batch_size",            &SteinICPParam::batch_size,
            "Points per mini-batch (ignored when use_minibatch=False).")
        .def_readwrite("lr",                    &SteinICPParam::lr,
            "Learning rate / step size.")
        .def_readwrite("max_dist",              &SteinICPParam::max_dist,
            "Maximum correspondence distance threshold.")
        .def_readwrite("normalize_cloud",       &SteinICPParam::normalize_cloud,
            "Normalise the source cloud before alignment.")
        .def_readwrite("optimizer",             &SteinICPParam::optimizer,
            "Optimizer name: 'Adam', 'RMSprop', 'SGD', or 'Adagrad'.")
        .def_readwrite("check_early_stop",      &SteinICPParam::check_early_stop,
            "Stop early when the gradient norm falls below the threshold.")
        .def_readwrite("convergence_steps",     &SteinICPParam::convergence_steps,
            "Consecutive steps below threshold to declare convergence.")
        .def_readwrite("convergence_threshold", &SteinICPParam::convergence_threshold,
            "Gradient-norm threshold for early stopping.")
        .def_readwrite("KNN_count",             &SteinICPParam::KNN_count,
            "Nearest neighbours used in the correspondence search.")
        .def_readwrite("SVN_full_grad",         &SteinICPParam::SVN_full_grad,
            "Use full SVN gradient (True) or preconditioned SVGD (False).")
        .def_readwrite("cov_filter_type",       &SteinICPParam::cov_filter_type,
            "Post-processing filter applied to the output covariance.");

    // ------------------------------------------------------------------
    // ParticleWeightOpt
    // ------------------------------------------------------------------
    py::class_<ParticleWeightOpt>(m, "ParticleWeightOpt",
        "Options controlling how the final pose estimate is computed "
        "from the particle distribution.")
        .def(py::init<>())
        .def_readwrite("use_weight_mean", &ParticleWeightOpt::use_weight_mean,
            "Use the weighted mean of particles as the point estimate.");

    // ------------------------------------------------------------------
    // SVNICP
    // ------------------------------------------------------------------
    py::class_<SVNICP>(m, "SVNICP",
        "Stein Variational Newton ICP.\n\n"
        "Typical usage:\n"
        "  param = pysvnicp.SteinICPParam()\n"
        "  param.iterations = 30\n"
        "  init_pose = pysvnicp.initialize_particles(50, True, ub, lb)\n"
        "  opt = pysvnicp.ParticleWeightOpt()\n"
        "  icp = pysvnicp.SVNICP(param, init_pose, opt)\n"
        "  icp.add_cloud(source_t, target_t, init_pose)\n"
        "  state = icp.stein_align()\n"
        "  T_vec = icp.get_transformation()  # 6-elem torch.Tensor")
        .def(py::init<const SteinICPParam&,
                      const torch::Tensor&,
                      const ParticleWeightOpt&>(),
             py::arg("param"),
             py::arg("init_pose"),
             py::arg("weight_opt"),
             "Construct SVNICP.\n\n"
             "param      -- SteinICPParam\n"
             "init_pose  -- torch.Tensor shape (6, N, 1), float64, CUDA\n"
             "weight_opt -- ParticleWeightOpt")
        .def("add_cloud", &SVNICP::add_cloud,
             py::arg("source"),
             py::arg("target"),
             py::arg("init_pose"),
             "Set source/target clouds and reset the particle state.\n\n"
             "source    -- torch.Tensor (P, 3) float64 CUDA\n"
             "target    -- torch.Tensor (Q, 3) float64 CUDA\n"
             "init_pose -- torch.Tensor (6, N, 1) float64 CUDA")
        .def("stein_align", &SVNICP::stein_align,
             py::call_guard<py::gil_scoped_release>(),
             "Run Stein-gradient ICP iterations. Returns SteinICPState.")
        .def("get_transformation", &SVNICP::get_transformation,
             "Return weighted-mean pose as a 6-element torch.Tensor "
             "[tx, ty, tz, rx, ry, rz].")
        .def("get_distribution", &SVNICP::get_distribution,
             "Return per-DOF variance of the particle distribution as a "
             "6-element torch.Tensor.")
        .def("get_cov_matrix", &SVNICP::get_cov_matrix,
             "Return the 6x6 weighted covariance as a flat list of 36 floats "
             "(row-major).")
        .def("get_particles", &SVNICP::get_particles,
             "Return all particle poses as a flat list of 6*N floats.")
        .def("get_particle_weight", &SVNICP::get_particle_weight,
             "Return N particle weights as a list of floats.")
        .def("get_runtime", &SVNICP::get_runtime,
             "Return [knn_duration_s, update_duration_s, finish_iter].")
        .def("set_k", &SVNICP::set_k,
             py::arg("k"),
             "Override the number of nearest neighbours used during alignment.")
        .def("set_threshold", &SVNICP::set_threshold,
             py::arg("max_dist"),
             "Override the correspondence distance threshold.");

    // ------------------------------------------------------------------
    // VoxelHashMap
    // ------------------------------------------------------------------
    py::class_<VoxelHashMap>(m, "VoxelHashMap",
        "Voxel-hashed local map storing PointXYZI point clouds.\n\n"
        "Typical usage:\n"
        "  vmap = pysvnicp.VoxelHashMap(voxel_size=1.0, max_range=80.0)\n"
        "  vmap.add_point_cloud(pts_np, pose_4x4)  # pts_np: Nx3/Nx4 float32\n"
        "  target_np = vmap.get_map_near_pose(pose_4x4, max_range=50.0)\n"
        "  target_t  = pysvnicp.numpy_to_tensor(target_np[:, :3].astype('f8'))")
        .def(py::init<double, double, int>(),
             py::arg("voxel_size") = 1.0,
             py::arg("max_range")  = 80.0,
             py::arg("max_points") = 20,
             "voxel_size -- edge length of each voxel in metres\n"
             "max_range  -- far-field culling radius in metres\n"
             "max_points -- maximum points stored per voxel")
        .def("add_point_cloud",
             [](VoxelHashMap& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> cloud_np,
                py::array_t<double, py::array::c_style | py::array::forcecast> pose_np) {
                 self.AddPointCloud(numpy_to_pcl(cloud_np), numpy_to_pose3(pose_np));
             },
             py::arg("cloud"),
             py::arg("pose"),
             "Insert a new scan.\n\n"
             "cloud -- np.ndarray (N, 3|4) float32 in sensor frame\n"
             "pose  -- np.ndarray (4, 4) float64, sensor-to-map SE(3) transform")
        .def("get_map",
             [](VoxelHashMap& self) {
                 return pcl_to_numpy(self.GetMap());
             },
             "Return all map points as an Nx4 float32 array [x, y, z, i].")
        .def("get_map_near_pose",
             [](VoxelHashMap& self,
                py::array_t<double, py::array::c_style | py::array::forcecast> pose_np,
                double max_range) {
                 return pcl_to_numpy(self.GetMap(numpy_to_pose3(pose_np), max_range));
             },
             py::arg("pose"),
             py::arg("max_range"),
             "Return map points within max_range metres of pose as Nx4 float32.")
        .def("get_neighbour_map",
             [](VoxelHashMap& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> cloud_np) {
                 return pcl_to_numpy(self.GetNeighbourMap(numpy_to_pcl(cloud_np)));
             },
             py::arg("cloud"),
             "Return voxels adjacent to those occupied by cloud as Nx4 float32.")
        .def("clear",  &VoxelHashMap::Clear,  "Remove all stored points.")
        .def("empty",  &VoxelHashMap::Empty,  "Return True if the map is empty.")
        .def("size",   &VoxelHashMap::Size,   "Return the number of occupied voxels.");

    // ------------------------------------------------------------------
    // Particle initialisation helpers
    // ------------------------------------------------------------------
    m.def("initialize_particles",
          [](int n, bool cuda, torch::Tensor ub, torch::Tensor lb) {
              return initialize_particles(
                  n, cuda ? torch::kCUDA : torch::kCPU, ub, lb);
          },
          py::arg("n"),
          py::arg("cuda") = true,
          py::arg("ub"),
          py::arg("lb"),
          "Sample n particles uniformly in [lb, ub].\n\n"
          "n    -- number of particles\n"
          "cuda -- place result on CUDA (default True)\n"
          "ub   -- torch.Tensor shape (6,) float64, upper bounds [tx,ty,tz,rx,ry,rz]\n"
          "lb   -- torch.Tensor shape (6,) float64, lower bounds\n"
          "Returns torch.Tensor shape (6, n, 1) float64.");

    m.def("initialize_particles_gaussian",
          [](int n, bool cuda, std::vector<double> cov_vec) {
              if (cov_vec.size() != 6)
                  throw std::invalid_argument("cov must have exactly 6 elements.");
              gtsam::Vector6 cov;
              for (int i = 0; i < 6; ++i) cov[i] = cov_vec[i];
              return initialize_particles_gaussian(
                  n, cuda ? torch::kCUDA : torch::kCPU, cov);
          },
          py::arg("n"),
          py::arg("cuda") = true,
          py::arg("cov"),
          "Sample n particles from zero-mean Gaussian with per-DOF variances cov.\n\n"
          "n    -- number of particles\n"
          "cuda -- place result on CUDA (default True)\n"
          "cov  -- list of 6 floats, per-DOF variances [tx,ty,tz,rx,ry,rz]\n"
          "Returns torch.Tensor shape (6, n, 1) float64.");

    // ------------------------------------------------------------------
    // Coordinate-conversion utilities
    // ------------------------------------------------------------------
    m.def("numpy_to_tensor",
          &numpy_to_tensor,
          py::arg("array"),
          "Convert an Nx3 float64 numpy array to a CUDA torch.Tensor (N,3) float64.");

    m.def("tensor_to_numpy",
          &tensor_to_numpy,
          py::arg("tensor"),
          "Convert a torch.Tensor (any device) to a float64 numpy array.");

    m.def("tensor_to_pose",
          &tensor_to_pose,
          py::arg("tensor"),
          "Convert a 6-element pose tensor [tx,ty,tz,rx,ry,rz] to a 4x4 "
          "float64 numpy SE(3) matrix.");

    m.def("pose_to_tensor",
          &pose_to_tensor,
          py::arg("pose"),
          "Convert a 4x4 float64 numpy SE(3) matrix to a 6-element CUDA "
          "torch.Tensor [tx,ty,tz,rx,ry,rz] (axis-angle rotation).");
}
