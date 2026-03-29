import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_trajectory_file(filename):
    """
    Read trajectory file and extract poses.
    Each row contains 16 values representing a 4x4 transformation matrix.
    """
    poses = []
    with open(filename, 'r') as f:
        for line in f:
            # Split the line and convert to float
            values = [float(x) for x in line.strip().split()]
            if len(values) == 16:
                # Reshape to 4x4 matrix
                matrix = np.array(values).reshape(4, 4)
                poses.append(matrix)
    return poses

def extract_positions(poses):
    """
    Extract x, y, z positions from transformation matrices.
    The translation vector is in the last column of the 4x4 matrix.
    """
    positions = []
    for pose in poses:
        # Translation is in the first 3 elements of the last column
        x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
        positions.append([x, y, z])
    return np.array(positions)

def plot_2d_trajectory(positions, plane='xy'):
    """
    Plot 2D trajectory by projecting to a specified plane.
    
    Args:
        positions: Nx3 array of [x, y, z] positions
        plane: 'xy', 'xz', or 'yz' - which plane to project to
    """
    plt.figure(figsize=(10, 8))
    
    if plane == 'xy':
        plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
        plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start', zorder=5)
        plt.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End', zorder=5)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('2D Trajectory (XY Plane)')
    elif plane == 'xz':
        plt.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
        plt.scatter(positions[0, 0], positions[0, 2], color='green', s=100, label='Start', zorder=5)
        plt.scatter(positions[-1, 0], positions[-1, 2], color='red', s=100, label='End', zorder=5)
        plt.xlabel('X Position')
        plt.ylabel('Z Position')
        plt.title('2D Trajectory (XZ Plane)')
    elif plane == 'yz':
        plt.plot(positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
        plt.scatter(positions[0, 1], positions[0, 2], color='green', s=100, label='Start', zorder=5)
        plt.scatter(positions[-1, 1], positions[-1, 2], color='red', s=100, label='End', zorder=5)
        plt.xlabel('Y Position')
        plt.ylabel('Z Position')
        plt.title('2D Trajectory (YZ Plane)')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_3d_trajectory(positions):
    """
    Plot 3D trajectory with orientation arrows.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    
    # Mark start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               color='green', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               color='red', s=100, label='End')
    
    # Add coordinate arrows for reference
    ax.quiver(positions[0, 0], positions[0, 1], positions[0, 2], 
              0.01, 0, 0, color='red', alpha=0.8, arrow_length_ratio=0.1)
    ax.quiver(positions[0, 0], positions[0, 1], positions[0, 2], 
              0, 0.01, 0, color='green', alpha=0.8, arrow_length_ratio=0.1)
    ax.quiver(positions[0, 0], positions[0, 1], positions[0, 2], 
              0, 0, 0.01, color='blue', alpha=0.8, arrow_length_ratio=0.1)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Trajectory')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def plot_3d_trajectory_with_orientation(poses, step=10):
    """
    Plot 3D trajectory with orientation vectors showing the direction of movement.
    
    Args:
        poses: List of 4x4 transformation matrices
        step: Show orientation every 'step' poses to avoid clutter
    """
    positions = extract_positions(poses)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    
    # Mark start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               color='green', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               color='red', s=100, label='End')
    
    # Add orientation arrows
    for i in range(0, len(poses), step):
        pose = poses[i]
        pos = positions[i]
        
        # Extract rotation matrix (top-left 3x3)
        rotation = pose[:3, :3]
        
        # X, Y, Z axes of the coordinate frame (scaled for visibility)
        scale = 0.005
        x_axis = rotation[:, 0] * scale
        y_axis = rotation[:, 1] * scale
        z_axis = rotation[:, 2] * scale
        
        # Plot orientation axes
        ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], 
                 color='red', alpha=0.6, arrow_length_ratio=0.1)
        ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], 
                 color='green', alpha=0.6, arrow_length_ratio=0.1)
        ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], 
                 color='blue', alpha=0.6, arrow_length_ratio=0.1)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Trajectory with Orientation')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace 'trajectory.txt' with your actual filename
    filename = '/home/mistlab/Documents/ml/MAGiC-SLAM/data/limo/mist/rosbag_1/traj.txt'
    
    try:
        # Read trajectory data
        poses = read_trajectory_file(filename)
        positions = extract_positions(poses)
        
        print(f"Loaded {len(poses)} poses")
        print(f"Position range: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
              f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
              f"Z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
        
        # Plot 2D projections
        plot_2d_trajectory(positions, plane='xy')
        plot_2d_trajectory(positions, plane='xz')
        plot_2d_trajectory(positions, plane='yz')
        
        # Plot 3D trajectory
        plot_3d_trajectory(positions)
        
        # Plot 3D trajectory with orientation (every 20th pose to avoid clutter)
        plot_3d_trajectory_with_orientation(poses, step=20)
        
    except FileNotFoundError:
        print(f"File '{filename}' not found. Please check the filename and path.")
    except Exception as e:
        print(f"Error reading trajectory file: {e}")