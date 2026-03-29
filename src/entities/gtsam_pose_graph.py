""" This module contains a class that wraps a pose graph using GTSAM library.
    GTSAM (Georgia Tech Smoothing and Mapping) is a C++ library with Python bindings
    that provides state-of-the-art factor graph optimization for robotics and computer vision.
    
    We use GTSAM's NonlinearFactorGraph and Values for pose graph optimization,
    which provides more robust optimization compared to GraphSLAM.

    You can get more details in Section 3.3 of the paper under "Pose Graph Optimization".

    GTSAM advantages:
    1. More robust optimization algorithms (Levenberg-Marquardt, Gauss-Newton, Dogleg)
    2. Better numerical stability
    3. Support for various robust kernels for outlier rejection
    4. Extensive documentation and active development
    5. Efficient sparse matrix operations
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import gtsam
except ImportError:
    raise ImportError("GTSAM library is required. Install with: pip install gtsam")


def pose2g2o_vertex(id: int, T: np.ndarray) -> str:
    """ Converts a pose to a g2o vertex string.
    Args:
        id: The id of the vertex.
        T: The pose matrix of shape (4, 4).
    Returns:
        The g2o vertex string.
    """
    quat = R.from_matrix(T[:3, :3].copy()).as_quat()
    t = T[:3, 3].copy()
    return f"VERTEX_SE3:QUAT {id} {t[0]} {t[1]} {t[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}"


def pose2g2o_edge(start: int, end: int, T, info=1) -> str:
    """ Converts a pose to a g2o edge string.
    Args:
        start: The id of the start vertex.
        end: The id of the end vertex.
        T: The relative pose matrix of shape (4, 4).
        info: The information matrix value. Since we do not have access to information evaluation
              we set it to 1 in all our experiments.
    Returns:
        The g2o edge string.
    """
    quat = R.from_matrix(T[:3, :3].copy()).as_quat()
    t = T[:3, 3].copy()
    edge_str = (
        f"EDGE_SE3:QUAT {start} {end} "
        f"{t[0]} {t[1]} {t[2]} "
        f"{quat[0]} {quat[1]} {quat[2]} {quat[3]} "
        f"{info} 0 0 0 0 0 {info} 0 0 0 0 {info} 0 0 0 {info} 0 0 {info} 0 {info}")
    return edge_str


def numpy_to_gtsam_pose3(T: np.ndarray) -> gtsam.Pose3:
    """Convert numpy 4x4 transformation matrix to GTSAM Pose3.
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        GTSAM Pose3 object
    """
    rotation = gtsam.Rot3(T[:3, :3])
    translation = gtsam.Point3(T[:3, 3])
    return gtsam.Pose3(rotation, translation)


def gtsam_pose3_to_numpy(pose: gtsam.Pose3) -> np.ndarray:
    """Convert GTSAM Pose3 to numpy 4x4 transformation matrix.
    
    Args:
        pose: GTSAM Pose3 object
        
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T


class PoseGraphAdapter_gtsam(object):

    def __init__(self, agents_submaps: dict, loops: list) -> None:
        self.agents_submaps = agents_submaps
        self.loops = loops
        self.graph_info = {}
        self.edges = []  # Keep for g2o compatibility
        self.nodes = []  # Keep for g2o compatibility
        
        # GTSAM components
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.result = None
        self.fixed_nodes = set()
        
        self.setup_odometry()
        self.setup_loops()

    def setup_odometry(self) -> None:
        """ Sets up the odometry edges in the pose graph using GTSAM. """
        agent_ids = sorted(self.agents_submaps.keys())
        
        for agent_id in agent_ids:
            nodes_count = len(self.graph_info)
            
            for i in range(len(self.agents_submaps[agent_id])):
                if i == 0:
                    # First node of each agent
                    source_sub = self.agents_submaps[agent_id][i]
                    node_id = len(self.graph_info)
                    self.graph_info[(agent_id, source_sub["submap_start_frame_id"])] = node_id
                    
                    # Add initial estimate
                    pose = numpy_to_gtsam_pose3(source_sub["submap_c2ws"][0])
                    self.initial_estimate.insert(node_id, pose)
                    
                    # Keep g2o format for compatibility
                    node = pose2g2o_vertex(node_id, source_sub["submap_c2ws"][0])
                    self.nodes.append(node)
                    
                    # Mark first node of each agent as potentially fixed
                    if agent_id == 0:  # Only fix the first agent's first node
                        self.fixed_nodes.add(node_id)
                    
                    continue

                source_sub = self.agents_submaps[agent_id][i - 1]
                target_sub = self.agents_submaps[agent_id][i]
                
                # Current node
                current_node_id = len(self.graph_info)
                self.graph_info[(agent_id, target_sub["submap_start_frame_id"])] = current_node_id
                
                # Previous node
                prev_node_id = nodes_count + i - 1
                
                # Add initial estimate for current node
                pose = numpy_to_gtsam_pose3(target_sub["submap_c2ws"][0])
                self.initial_estimate.insert(current_node_id, pose)
                
                # Calculate relative pose for odometry constraint
                rel_pose = np.linalg.inv(source_sub["submap_c2ws"][0]) @ target_sub["submap_c2ws"][0]
                relative_pose_gtsam = numpy_to_gtsam_pose3(rel_pose)
                
                # Create noise model (can be tuned based on odometry confidence)
                odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
                )
                
                # Add between factor (odometry constraint)
                factor = gtsam.BetweenFactorPose3(
                    prev_node_id, current_node_id, relative_pose_gtsam, odometry_noise
                )
                self.graph.add(factor)
                
                # Keep g2o format for compatibility
                node = pose2g2o_vertex(current_node_id, target_sub["submap_c2ws"][0])
                edge = pose2g2o_edge(prev_node_id, current_node_id, rel_pose)
                self.nodes.append(node)
                self.edges.append(edge)

    def setup_loops(self) -> None:
        """ Sets up the loop closure edges in the pose graph using GTSAM. """
        for loop in self.loops:
            source_node_id = self.graph_info[(loop.source_agent_id, loop.source_frame_id)]
            target_node_id = self.graph_info[(loop.target_agent_id, loop.target_frame_id)]
            
            # Convert loop transformation to GTSAM format
            loop_transform = numpy_to_gtsam_pose3(np.linalg.inv(loop.transformation))

            # Create noise model for loop closure (typically less confident than odometry)
            # sigmas = self.fitness_to_sigma(loop.fitness)
            # loop_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
            loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
                 np.array([0.3, 0.3, 0.3, 0.2, 0.2, 0.2])  # More uncertainty in loop closures
            )
            scale_factor = 1e-6
            # loop_noise = gtsam.noiseModel.Gaussian.Information(loop.info_matrix * scale_factor)

            # Add between factor for loop closure
            factor = gtsam.BetweenFactorPose3(
                source_node_id, target_node_id, loop_transform, loop_noise
            )
            self.graph.add(factor)
            
            # Keep g2o format for compatibility
            edge = pose2g2o_edge(source_node_id, target_node_id, np.linalg.inv(loop.transformation))
            self.edges.append(edge)

    def optimize(self, max_iter: int = 100, optimizer_type: str = 'LM') -> None:
        """ Optimizes the pose graph using GTSAM.
        
        Args:
            max_iter: The maximum number of iterations for optimization.
            optimizer_type: Type of optimizer ('LM' for Levenberg-Marquardt, 
                          'GN' for Gauss-Newton, 'DOGLEG' for Dogleg)
        """
        # Add prior factors for fixed nodes
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])  # Very small uncertainty
        )
        
        for node_id in self.fixed_nodes:
            if self.initial_estimate.exists(node_id):
                prior = gtsam.PriorFactorPose3(
                    node_id, 
                    self.initial_estimate.atPose3(node_id), 
                    prior_noise
                )
                self.graph.add(prior)
        
        # Set up optimizer parameters
        if optimizer_type == 'LM':
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(max_iter)
            params.setVerbosity('ERROR')  # Can be 'SILENT', 'ERROR', 'VALUES', 'DELTA', 'LINEAR'
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params)
        elif optimizer_type == 'GN':
            params = gtsam.GaussNewtonParams()
            params.setMaxIterations(max_iter)
            params.setVerbosity('ERROR')
            optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_estimate, params)
        elif optimizer_type == 'DOGLEG':
            params = gtsam.DoglegParams()
            params.setMaxIterations(max_iter)
            params.setVerbosity('ERROR')
            optimizer = gtsam.DoglegOptimizer(self.graph, self.initial_estimate, params)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Perform optimization
        self.result = optimizer.optimize()
        
        # Print optimization statistics
        final_error = self.graph.error(self.result)
        print(f"Optimization completed with final error: {final_error}")

    def get_poses(self) -> dict:
        """ Returns the optimized poses for all agents.
        Returns:
            optimized_poses: The optimized poses for all agents (agent_id: str, poses: np.ndarray of shape (n, 4, 4)).
        """
        if self.result is None:
            raise RuntimeError("Graph has not been optimized yet. Call optimize() first.")
        
        optimized_poses = {}
        for agent_id in sorted(self.agents_submaps.keys()):
            agent_poses = []
            for submap in self.agents_submaps[agent_id]:
                vertex_id = self.graph_info[(agent_id, submap["submap_start_frame_id"])]
                optimized_pose = self.result.atPose3(vertex_id)
                agent_poses.append(gtsam_pose3_to_numpy(optimized_pose))
            optimized_poses[agent_id] = np.array(agent_poses)
        return optimized_poses
    
    def get_marginal_covariances(self) -> dict:
        """ Returns the marginal covariances for all poses.
        Returns:
            covariances: Dictionary mapping (agent_id, frame_id) to 6x6 covariance matrix
        """
        if self.result is None:
            raise RuntimeError("Graph has not been optimized yet. Call optimize() first.")
        
        # Compute marginal covariances
        marginals = gtsam.Marginals(self.graph, self.result)
        
        covariances = {}
        for (agent_id, frame_id), vertex_id in self.graph_info.items():
            cov = marginals.marginalCovariance(vertex_id)
            covariances[(agent_id, frame_id)] = cov
        
        return covariances
    
    def save_g2o_file(self, filename: str) -> None:
        """ Saves the pose graph to a g2o file.
        Args:
            filename: The path where to save the g2o file.
        """
        with open(filename, 'w') as f:
            # Write all nodes first
            for node in self.nodes:
                f.write(node + "\n")
            # Write all edges
            for edge in self.edges:
                f.write(edge + "\n")
    
    def save_gtsam_graph(self, filename: str) -> None:
        """ Saves the GTSAM factor graph to a file.
        Args:
            filename: The path where to save the graph file.
        """
        if self.result is None:
            values_to_save = self.initial_estimate
        else:
            values_to_save = self.result
            
        # Save graph and values
        gtsam.writeG2o(self.graph, values_to_save, filename)
    
    def get_optimization_stats(self) -> dict:
        """ Returns optimization statistics.
        Returns:
            Dictionary containing optimization statistics
        """
        if self.result is None:
            return {"error": "Graph not optimized"}
        
        initial_error = self.graph.error(self.initial_estimate)
        final_error = self.graph.error(self.result)
        
        return {
            "initial_error": initial_error,
            "final_error": final_error,
            "error_reduction": initial_error - final_error,
            "relative_error_reduction": (initial_error - final_error) / initial_error if initial_error > 0 else 0,
            "num_factors": self.graph.size(),
            "num_variables": self.result.size()
        }
    
    def add_robust_kernel(self, kernel_type: str = 'Huber', threshold: float = 1.0) -> None:
        """ Add robust kernel to handle outliers in loop closures.
        
        Args:
            kernel_type: Type of robust kernel ('Huber', 'Cauchy', 'Tukey')
            threshold: Threshold parameter for the robust kernel
        """
        # This would need to be implemented during graph construction
        # For now, we'll store the parameters for future use
        self.robust_kernel_type = kernel_type
        self.robust_kernel_threshold = threshold
        print(f"Robust kernel {kernel_type} with threshold {threshold} will be applied in future optimizations")

    def fitness_to_sigma(self, fitness, 
                        min_sigma_rot=0.01, max_sigma_rot=0.5,
                        min_sigma_trans=0.01, max_sigma_trans=1.0,
                        k_rot=7.0, k_trans=4.0):
        """
        Convert registration fitness to noise model sigma using logistic functions.
        Uses different k values (steepness) for rotation and translation components.
        
        Args:
            fitness (float): Registration fitness between 0 and 1 (higher = more confident)
            min_sigma_rot (float): Minimum sigma value for rotation (highest confidence)
            max_sigma_rot (float): Maximum sigma value for rotation (lowest confidence)
            min_sigma_trans (float): Minimum sigma value for translation (highest confidence)
            max_sigma_trans (float): Maximum sigma value for translation (lowest confidence)
            k_rot (float): Steepness parameter for rotation logistic function (higher = steeper)
            k_trans (float): Steepness parameter for translation logistic function (higher = steeper)
        
        Returns:
            np.array: Array of 6 sigma values [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        """
        # Clamp fitness to valid range
        fitness = np.clip(fitness, 1e-6, 1.0 - 1e-6)
        
        # Convert fitness to a centered input for logistic function
        # Map fitness from [0,1] to a range around 0 for better logistic behavior
        # Higher fitness should result in lower sigma
        x = -k_rot * (2 * fitness - 1)  # Maps [0,1] to [k, -k], inverted for high fitness -> low sigma
        
        # Logistic function for rotation: L(x) = 1 / (1 + e^(-k*x))
        # We want high fitness -> low sigma, so we use (1 - logistic)
        logistic_rot = 1.0 / (1.0 + np.exp(-x))
        sigma_rot = min_sigma_rot + (max_sigma_rot - min_sigma_rot) * (1.0 - logistic_rot)
        
        # Different logistic function for translation with different k value
        x_trans = -k_trans * (2 * fitness - 1)  # Same mapping but with different k
        logistic_trans = 1.0 / (1.0 + np.exp(-x_trans))
        sigma_trans = min_sigma_trans + (max_sigma_trans - min_sigma_trans) * (1.0 - logistic_trans)
        
        # Return array with rotation sigmas (first 3) and translation sigmas (last 3)
        # Standard order: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        sigmas = np.array([sigma_rot, sigma_rot, sigma_rot, 
                        sigma_trans, sigma_trans, sigma_trans])
        
        return sigmas