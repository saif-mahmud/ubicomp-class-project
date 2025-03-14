# Import required modules
from hmr2.utils.renderer import Renderer  # For rendering SMPL meshes
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT  # HMR2 model loading
import numpy as np
from hmr2.models.smpl_wrapper import SMPL  # SMPL body model wrapper
from hmr2.datasets.vitdet_dataset import (
    DEFAULT_MEAN,
    DEFAULT_STD,
)  # Image normalization constants
import torch


def find_frame_idx(ts_list, ts_target):
    """
    Find the index of the first timestamp in ts_list that is >= ts_target
    Returns -1 if no such timestamp exists
    """
    idx = 0
    while True:
        if idx == len(ts_list):
            return -1
        if ts_target <= ts_list[idx]:
            return idx
        else:
            idx += 1


def normalize_vector(x):
    """Normalize a vector to unit length along the last axis"""
    return x / np.linalg.norm(x, axis=-1)[:, None]


def rot6d_to_matrix(ortho6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix
    Implementation follows Yi Zhou's paper on rotation continuity
    """
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]

    x = normalize_vector(x_raw)  # First basis vector
    y = y_raw - (x * y_raw).sum(-1)[:, None] * x  # Orthogonalize second vector
    y = normalize_vector(y)  # Second basis vector
    z = np.cross(x, y)  # Third basis vector (cross product)

    matrix = np.transpose(
        np.dstack((x, y, z)), (0, 2, 1)
    )  # Combine into rotation matrix
    return matrix


pred_smpl_params = dict()  # Global storage for SMPL parameters


class SMPLVisualizer:
    def __init__(self):
        """Initialize SMPL visualizer with default settings and dependencies"""
        self.color = (0.73333, 0.65, 0.82)  # Mesh color (light purple)
        self.set_defualt_params_of_interest()
        self.init_smpl_cfg()  # Initialize SMPL configuration

        self.smpl_device = torch.device("cpu")  # Use CPU for SMPL computations
        print(f"SMPL device: {self.smpl_device}")

        self.load_model_cfg()  # Load HMR2 model configuration
        self.setup_renderer()  # Initialize renderer
        self.smpl_model = SMPL(**self.smpl_cfg).to(
            self.smpl_device
        )  # Create SMPL model instance

    def set_defualt_params_of_interest(self):
        """Set which body joints to modify (upper body joints)"""
        # Indices correspond to SMPL body joints (excluding head and lower body)
        self.bodyparams_of_interest = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    def init_smpl_cfg(self):
        """Initialize SMPL model configuration with file paths and parameters"""
        SMPL_config = dict(
            MODEL_PATH="/home/saif/.cache/4DHumans/data/smpl",
            GENDER="neutral",
            NUM_BODY_JOINTS=23,
            JOINT_REGRESSOR_EXTRA="/home/saif/.cache/4DHumans/data/SMPL_to_J19.pkl",
            MEAN_PARAMS="/home/saif/.cache/4DHumans/data/smpl_mean_params.npz",
        )
        self.smpl_cfg = {k.lower(): v for k, v in SMPL_config.items()}
        # print(f"SMPL configuration initialized: \n{self.smpl_cfg}")

    def load_model_cfg(self):
        """Load HMR2 model configuration from checkpoint"""
        hmr2_model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.model_cfg = model_cfg
        self.hmr2_model = hmr2_model

    def setup_renderer(self):
        """Initialize mesh renderer with SMPL face topology"""
        print("[INFO] Initializing mesh renderer...")
        self.renderer = Renderer(self.model_cfg, faces=self.hmr2_model.smpl.faces)

    def load_smpl_data(self, gt_path, pred_path=None, model_gt_path=None):
        """
        Load prediction and ground truth data from NPZ files
        Args:
            pred_path: Path to predicted pose parameters (optional)
            model_gt_path: Path to model-generated ground truth (required if pred_path is provided)
            gt_path: Path to original ground truth data
        """
        # Load ground truth data
        self.gt = np.load(gt_path)
        self.gt_ts = self.gt[:, 0]  # Timestamps
        self.gt_pose = self.gt[:, 85:292]  # Pose parameters (rotation matrices)
        self.gt_shape = self.gt[:, 292:302]  # Shape parameters
        self.gt_orient = self.gt[:, 76:85].reshape((-1, 3, 3))  # Global orientation

        if pred_path is not None:
            # Load prediction data and model-generated ground truth
            self.pred = np.load(pred_path)
            self.model_gt = np.load(model_gt_path)
            self.pred_ts = self.model_gt[:, 0]  # Predicted timestamps
            self.pred_pose_6D = self.pred[
                :, -len(self.bodyparams_of_interest) * 6 :
            ]  # 6D pose params

        else:
            print(
                "No pred file provided for the combined visualization. Only GT data loaded."
            )
            self.pred = None
            self.model_gt = None

    def get_smpl_gt_shape(self):
        """Get ground truth shape parameters"""
        return self.gt.shape
    
    def visualize_pred_frame(self, frame_number):
        """
        Generate visualization combining both ground truth and predicted data.
        Returns rendered image of the SMPL mesh from a side view.
        """
        if self.pred is None:
            print(
                "No pred file provided for combined visualization. Load a pred file first."
            )
            return None

        # Get timestamp for current frame
        this_ts = self.gt_ts[frame_number]

        # Find corresponding prediction index
        pred_idx = find_frame_idx(self.pred_ts, this_ts)

        # Process predicted pose parameters
        pred_pose = self.pred_pose_6D[pred_idx]
        pred_pose_matrix = rot6d_to_matrix(
            pred_pose.reshape(len(self.bodyparams_of_interest), 6)
        )

        # Combine ground truth and predicted parameters
        pose_to_draw = self.gt_pose[frame_number].reshape((-1, 3, 3))
        pose_to_draw[self.bodyparams_of_interest] = (
            pred_pose_matrix  # Replace selected joints
        )
        shape_to_draw = self.gt_shape[frame_number]
        global_orient_to_draw = self.gt_orient[frame_number]

        return self.render_frame(pose_to_draw, shape_to_draw, global_orient_to_draw)

    def visualize_gt_frame(self, frame_number):
        """
        Generate visualization using only ground truth data.
        Returns rendered image of the SMPL mesh from a side view.
        """
        # Use ground truth pose parameters without modification
        pose_to_draw = self.gt_pose[frame_number].reshape((-1, 3, 3))
        shape_to_draw = self.gt_shape[frame_number]
        global_orient_to_draw = self.gt_orient[frame_number]

        return self.render_frame(pose_to_draw, shape_to_draw, global_orient_to_draw)


    def render_frame(self, pose_to_draw, shape_to_draw, global_orient_to_draw):
        """
        Common rendering steps for both GT and combined visualization.
        """
        # Prepare SMPL parameters
        pred_smpl_params = {
            "body_pose": torch.from_numpy(pose_to_draw.reshape((1, -1, 3, 3))),
            "betas": torch.from_numpy(shape_to_draw.reshape((1, -1))),
            "global_orient": torch.from_numpy(global_orient_to_draw.reshape(1, 3, 3)),
        }

        # Run SMPL model to get vertices
        smpl_output = self.smpl_model(
            **{k: v.float() for k, v in pred_smpl_params.items()}, pose2rot=False
        )
        pred_vertices = smpl_output["vertices"][0]

        # Prepare blank white background image
        white_img = (
            torch.ones([3, 1024, 1024]).cpu() - DEFAULT_MEAN[:, None, None] / 255
        ) / (DEFAULT_STD[:, None, None] / 255)

        # Render mesh from side view
        side_img = self.renderer(
            pred_vertices.detach().cpu().numpy(),
            np.array([0.0142, 0.2047, 10.372]),  # Camera position
            white_img,
            mesh_base_color=self.color,
            side_view=True,
            scene_bg_color=(1, 1, 1),  # White background
            rot_angle=-15,  # Rotation angle for side view
        )
        return side_img
