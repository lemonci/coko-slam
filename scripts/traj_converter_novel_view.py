import numpy as np

def read_trajectory(filename):
    """Read trajectory file and convert to list of 4x4 matrices."""
    matrices = []
    with open(filename, 'r') as f:
        for line in f:
            # Parse 16 values from each line
            values = [float(x) for x in line.strip().split()]
            if len(values) == 16:
                # Reshape to 4x4 matrix (row-major order)
                matrix = np.array(values).reshape(4, 4)
                matrices.append(matrix)
    return matrices

def read_first_pose(filename):
    """Read only the first pose from trajectory file."""
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            values = [float(x) for x in first_line.split()]
            if len(values) == 16:
                return np.array(values).reshape(4, 4)
    return None

def convert_relative_to_reference(reference_pose, poses_to_convert):
    """Convert poses to be relative to the reference pose."""
    # Compute inverse of reference pose
    reference_inv = np.linalg.inv(reference_pose)
    
    converted_poses = []
    for pose in poses_to_convert:
        # Convert pose to be relative to reference frame
        # T_relative = T_reference^(-1) * T_pose
        relative_pose = reference_inv @ pose
        converted_poses.append(relative_pose)
    
    return converted_poses

def write_trajectory(matrices, filename):
    """Write trajectory matrices to file in the same format."""
    with open(filename, 'w') as f:
        for matrix in matrices:
            # Flatten matrix to row-major order and write
            values = matrix.flatten()
            line = ' '.join([f'{val:.15e}' for val in values])
            f.write(line + '\n')

def convert_trajectory_cross_reference(path_a, path_b, output_file):
    """
    Convert trajectory B relative to first frame of trajectory A.
    
    Args:
        path_a: Path to trajectory file A (reference)
        path_b: Path to trajectory file B (to be converted)
        output_file: Path for output file
    """
    print(f"Reading reference pose from {path_a}...")
    reference_pose = read_first_pose(path_a)
    
    if reference_pose is None:
        print(f"Error: Could not read reference pose from {path_a}")
        return
    
    print(f"Reference pose translation: {reference_pose[:3, 3]}")
    print(f"Reference pose rotation (first row): {reference_pose[0, :3]}")
    
    print(f"Reading trajectory to convert from {path_b}...")
    poses_b = read_trajectory(path_b)
    
    if len(poses_b) == 0:
        print(f"Error: No valid poses found in {path_b}")
        return
    
    print(f"Loaded {len(poses_b)} poses from trajectory B")
    
    print("Converting poses relative to reference frame...")
    converted_poses = convert_relative_to_reference(reference_pose, poses_b)
    
    print(f"Writing converted trajectory to {output_file}...")
    write_trajectory(converted_poses, output_file)
    
    print("Conversion complete!")
    print(f"Original first pose of B translation: {poses_b[0][:3, 3]}")
    print(f"Converted first pose of B translation: {converted_poses[0][:3, 3]}")
    
    return converted_poses

def verify_conversion(path_a, path_b, converted_poses):
    """Verify the conversion by checking the transformation."""
    reference_pose = read_first_pose(path_a)
    original_poses_b = read_trajectory(path_b)
    
    if reference_pose is not None and len(original_poses_b) > 0 and len(converted_poses) > 0:
        # Verify: reference_pose @ converted_pose should equal original_pose
        reconstructed = reference_pose @ converted_poses[0]
        original = original_poses_b[0]
        
        diff = np.linalg.norm(reconstructed - original)
        print(f"\nVerification:")
        print(f"Reconstruction error (should be close to 0): {diff:.2e}")

if __name__ == "__main__":


    trajectory_a_path = "./data/Aria_Multiagent/room0/agent_0/traj.txt"
    trajectory_b_path = "./data/Aria_Multiagent/test/room0/traj.txt"
    output_path = "./data/Aria_Multiagent/test/room0/traj_converted.txt"
    print(f"\nUsing:")
    print(f"Reference trajectory A: {trajectory_a_path}")
    print(f"Test trajectory to convert: {trajectory_b_path}")
    print(f"Output file: {output_path}")
    print()
    
    # Convert the trajectory
    converted = convert_trajectory_cross_reference(trajectory_a_path, trajectory_b_path, output_path)
    
    # Verify the conversion
    if converted:
        verify_conversion(trajectory_a_path, trajectory_b_path, converted)