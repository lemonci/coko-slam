import cv2
import numpy as np
import os
import glob

class DepthImageViewer:
    def __init__(self, depth_path="../../data/limo/mist/rosbag_1/results/"):
        self.depth_path = depth_path
        self.current_index = 0
        self.colormap = cv2.COLORMAP_JET
        self.colormap_names = [
            "JET", "HOT", "COOL", "SPRING", "SUMMER", "AUTUMN", 
            "WINTER", "RAINBOW", "OCEAN", "PLASMA", "VIRIDIS"
        ]
        self.colormap_list = [
            cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_COOL,
            cv2.COLORMAP_SPRING, cv2.COLORMAP_SUMMER, cv2.COLORMAP_AUTUMN,
            cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN,
            cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS
        ]
        self.colormap_index = 0
        
        # Load all depth image files
        self.load_image_files()
        
        if not self.image_files:
            print(f"No depth images found in {self.depth_path}")
            return
        
        print(f"Found {len(self.image_files)} depth images")
        self.show_instructions()
        
        # Create window
        cv2.namedWindow('Depth Image Viewer', cv2.WINDOW_AUTOSIZE)
        
        # Start viewing
        self.view_images()
    
    def load_image_files(self):
        """Load all depth image files from the directory"""
        pattern = os.path.join(self.depth_path, "depth_*.png")
        self.image_files = sorted(glob.glob(pattern))
    
    def show_instructions(self):
        """Print control instructions"""
        instructions = """
╔════════════════════════════════════════════════════════════╗
║                    DEPTH IMAGE VIEWER                      ║
╠════════════════════════════════════════════════════════════╣
║  Controls:                                                 ║
║  • 'a' or Left Arrow  - Previous image                     ║
║  • 'd' or Right Arrow - Next image                         ║
║  • 'c'                - Change colormap                    ║
║  • 'r'                - Refresh image list                 ║
║  • 'f'                - Toggle fullscreen                  ║
║  • 'h'                - Show this help                     ║
║  • 'q' or ESC         - Quit                              ║
║  • Number keys (0-9)  - Jump to percentage of dataset     ║
╚════════════════════════════════════════════════════════════╝
        """
        print(instructions)
    
    def load_depth_image(self, filepath):
        """Load and process depth image"""
        try:
            # Load depth image (typically 16-bit)
            depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            
            if depth_img is None:
                print(f"Could not load image: {filepath}")
                return None
            
            # Normalize depth values for visualization
            if depth_img.dtype == np.uint16:
                # Convert to 8-bit for visualization
                depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                depth_norm = depth_img
            
            # Apply colormap
            colored_depth = cv2.applyColorMap(depth_norm, self.colormap)
            
            return colored_depth, depth_img
            
        except Exception as e:
            print(f"Error loading depth image {filepath}: {e}")
            return None, None
    
    def update_window_title(self, original_depth):
        """Update window title with image info"""
        filename = os.path.basename(self.image_files[self.current_index])
        colormap_name = self.colormap_names[self.colormap_index]
        
        if original_depth is not None:
            min_depth = np.min(original_depth[original_depth > 0]) if np.any(original_depth > 0) else 0
            max_depth = np.max(original_depth)
            title = f"Depth Viewer | {filename} | {self.current_index + 1}/{len(self.image_files)} | {colormap_name} | Range: {min_depth:.1f}-{max_depth:.1f}"
        else:
            title = f"Depth Viewer | {filename} | {self.current_index + 1}/{len(self.image_files)} | {colormap_name}"
        
        cv2.setWindowTitle('Depth Image Viewer', title)
    
    def display_current_image(self):
        """Display the current depth image"""
        if not self.image_files:
            return False
            
        filepath = self.image_files[self.current_index]
        colored_depth, original_depth = self.load_depth_image(filepath)
        
        if colored_depth is not None:
            cv2.imshow('Depth Image Viewer', colored_depth)
            self.update_window_title(original_depth)
            return True
        return False
    
    def next_image(self):
        """Go to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return True
        return False
    
    def prev_image(self):
        """Go to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False
    
    def next_colormap(self):
        """Cycle to next colormap"""
        self.colormap_index = (self.colormap_index + 1) % len(self.colormap_list)
        self.colormap = self.colormap_list[self.colormap_index]
        print(f"Colormap changed to: {self.colormap_names[self.colormap_index]}")
    
    def jump_to_percentage(self, percentage):
        """Jump to a percentage position in the dataset"""
        target_index = int((percentage / 10.0) * (len(self.image_files) - 1))
        target_index = max(0, min(target_index, len(self.image_files) - 1))
        self.current_index = target_index
        print(f"Jumped to image {self.current_index + 1} ({percentage * 10}% through dataset)")
    
    def refresh_images(self):
        """Refresh the list of images"""
        old_count = len(self.image_files)
        self.load_image_files()
        new_count = len(self.image_files)
        
        if new_count != old_count:
            print(f"Image count updated: {old_count} → {new_count}")
            if self.current_index >= new_count:
                self.current_index = max(0, new_count - 1)
        else:
            print("No new images found")
    
    def toggle_fullscreen(self):
        """Toggle between normal and fullscreen mode"""
        # Get current window property
        prop = cv2.getWindowProperty('Depth Image Viewer', cv2.WND_PROP_FULLSCREEN)
        if prop == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty('Depth Image Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_AUTOSIZE)
            print("Switched to windowed mode")
        else:
            cv2.setWindowProperty('Depth Image Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Switched to fullscreen mode")
    
    def view_images(self):
        """Main viewing loop"""
        # Display first image
        if not self.display_current_image():
            print("Could not display first image")
            return
        
        while True:
            key = cv2.waitKey(0) & 0xFF  # Wait for key press
            
            # Quit keys
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
            # Navigation
            elif key == ord('d') or key == 83:  # 'd' or right arrow
                if self.next_image():
                    self.display_current_image()
                else:
                    print("Already at last image")
            
            elif key == ord('a') or key == 81:  # 'a' or left arrow
                if self.prev_image():
                    self.display_current_image()
                else:
                    print("Already at first image")
            
            # Change colormap
            elif key == ord('c'):
                self.next_colormap()
                self.display_current_image()
            
            # Refresh images
            elif key == ord('r'):
                self.refresh_images()
                self.display_current_image()
            
            # Toggle fullscreen
            elif key == ord('f'):
                self.toggle_fullscreen()
            
            # Show help
            elif key == ord('h'):
                self.show_instructions()
            
            # Jump to percentage (0-9 keys)
            elif key >= ord('0') and key <= ord('9'):
                percentage = key - ord('0')
                self.jump_to_percentage(percentage)
                self.display_current_image()
            
            # Debug: show key code for unmapped keys
            else:
                print(f"Unknown key pressed: {key} ('{chr(key) if 32 <= key <= 126 else '?'}')")
        
        # Clean up
        cv2.destroyAllWindows()
        print("Viewer closed")

def main():
    # You can change the path here
    depth_path = "data/limo/mist/rosbag_1/results/"
    
    # Create and run viewer
    viewer = DepthImageViewer(depth_path)

if __name__ == "__main__":
    main()