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

def compute_relative_poses(matrices):
    """Compute relative transformations between consecutive frames."""
    relative_poses = []
    for i in range(1, len(matrices)):
        # Relative pose = T_prev^(-1) * T_curr
        T_prev_inv = np.linalg.inv(matrices[i-1])
        T_curr = matrices[i]
        relative_pose = T_prev_inv @ T_curr
        relative_poses.append(relative_pose)
    return relative_poses

def reconstruct_trajectory(relative_poses):
    """Reconstruct trajectory starting from identity matrix."""
    # Start with identity matrix (world frame)
    trajectory = [np.eye(4)]
    
    # Apply relative poses sequentially
    current_pose = np.eye(4)
    for relative_pose in relative_poses:
        current_pose = current_pose @ relative_pose
        trajectory.append(current_pose.copy())
    
    return trajectory

def write_trajectory(matrices, filename):
    """Write trajectory matrices to file in the same format."""
    with open(filename, 'w') as f:
        for matrix in matrices:
            # Flatten matrix to row-major order and write
            values = matrix.flatten()
            line = ' '.join([f'{val:.15e}' for val in values])
            f.write(line + '\n')

def convert_trajectory(input_file, output_file):
    """Main function to convert trajectory with first frame as world frame."""
    print(f"Reading trajectory from {input_file}...")
    original_matrices = read_trajectory(input_file)
    print(f"Loaded {len(original_matrices)} poses")
    
    if len(original_matrices) == 0:
        print("No valid poses found in file!")
        return
    
    print("Computing relative poses...")
    relative_poses = compute_relative_poses(original_matrices)
    
    print("Reconstructing trajectory with identity as first frame...")
    new_trajectory = reconstruct_trajectory(relative_poses)
    
    print(f"Writing converted trajectory to {output_file}...")
    write_trajectory(new_trajectory, output_file)
    
    print("Conversion complete!")
    print(f"Original first frame translation: {original_matrices[0][:3, 3]}")
    print(f"New first frame translation: {new_trajectory[0][:3, 3]}")

if __name__ == "__main__":
    # Convert the trajectory
    file_paths = [
        "./data/Aria_Multiagent/room0/agent_0/",
        "./data/Aria_Multiagent/room0/agent_1/",
        "./data/Aria_Multiagent/room0/agent_2/",
        "./data/Aria_Multiagent/room1/agent_0/",
        "./data/Aria_Multiagent/room1/agent_1/",
        "./data/Aria_Multiagent/room1/agent_2/",
        "./data/ReplicaMultiagent/Office-0/office_0_part1/",
        "./data/ReplicaMultiagent/Office-0/office_0_part2/",
        "./data/ReplicaMultiagent/Apart-0/apart_0_part1/",
        "./data/ReplicaMultiagent/Apart-0/apart_0_part2/",
        "./data/ReplicaMultiagent/Apart-1/apart_1_part1/",
        "./data/ReplicaMultiagent/Apart-1/apart_1_part2/",
        "./data/ReplicaMultiagent/Apart-2/apart_2_part1/",
        "./data/ReplicaMultiagent/Apart-2/apart_2_part2/",
    ]

     
    input_filename = "traj.txt"
    output_filename = "traj_converted.txt"
    
    for file_path in file_paths:
        convert_trajectory(file_path+input_filename, file_path+output_filename)
    
        # Optional: Verify the conversion by checking relative poses
        print("\nVerification:")
        original = read_trajectory(file_path+input_filename)
        converted = read_trajectory(file_path+output_filename)
        
        if len(original) > 1 and len(converted) > 1:
            # Check that relative pose between first two frames is preserved
            orig_rel = np.linalg.inv(original[0]) @ original[1]
            conv_rel = np.linalg.inv(converted[0]) @ converted[1]
            
            diff = np.linalg.norm(orig_rel - conv_rel)
            print(f"Relative pose difference (should be close to 0): {diff:.2e}")
