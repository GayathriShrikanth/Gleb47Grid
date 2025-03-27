import cv2
import os
import numpy as np
import math

# Hardcoded image path with your specific path
IMAGE_PATH = "C:\\Users\\Srivalli\\OneDrive\\Desktop\\Grid\\gayatri\\image9.jpg"
# "C:\Users\Srivalli\OneDrive\Desktop\Grid\gayatri\3D_MPR_-_ARJUN_WAGHMARE_-_24-05-2024_00_19_39_-_for_MPR_Bone_0000.jpg"
# IMAGE_PATH = "C:\\Users\\Srivalli\\OneDrive\\Desktop\\Grid\\gayatri\\3D_MP5_-_ANKUSH_KATAKARI_-_07-10-2024_23_09_26_-_for_MPR_Bone_0000.jpg"
# Window name
WINDOW_NAME = "Threshold Adjustment"

# Global variables
original_image = None
output_path = None
show_contours = False
current_threshold = 128
all_contours = []
selected_contour = None
selected_points = []
point_selection_mode = False
display_img = None
contour_points = []  # Will store the points of the selected contour
current_path = []  # Will store the current path points
show_grid_lines = False
show_original_with_grid = False  # Flag to show original image with grid
highlight_grid_boxes = False  # Flag to highlight center box and right box

# Define colorful line colors
PATH_COLOR = (0, 0, 255)        # Red for path
TOP_LINE_COLOR = (255, 0, 127)   # Magenta for top horizontal line
BOTTOM_LINE_COLOR = (0, 165, 255) # Orange for bottom horizontal line
MIDDLE_LINE_COLOR = (0, 255, 0)  # Green for middle horizontal line
FIRST_VERT_COLOR = (255, 127, 0) # Blue-violet for first vertical line
SECOND_VERT_COLOR = (128, 0, 128) # Purple for second vertical line
CONTOUR_COLOR = (0, 165, 255)    # Yellow for contours
POINT_COLOR = (255, 100, 0)      # Blue for selected points
MARKER_COLOR = (255, 0, 255)     # Magenta for marker points

def on_threshold_change(threshold_value):
    """Callback function for threshold trackbar"""
    global current_threshold, show_contours, all_contours, display_img, current_path, show_original_with_grid, highlight_grid_boxes
    
    # Update the current threshold value
    current_threshold = threshold_value
    
    # Apply threshold
    _, thresholded = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Check if we should display the original image with grid
    if show_original_with_grid and len(current_path) > 0:
        # Create a copy of the original image but convert to BGR for colored overlay
        if len(original_image.shape) == 2:  # If grayscale
            display_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:  # If already BGR
            display_img = original_image.copy()
            
        # Draw grid on original image
        draw_grid_on_original()
        
        # Highlight boxes if enabled
        if highlight_grid_boxes:
            highlight_boxes()
    else:
        # Create display image from thresholded image
        display_img = thresholded.copy()
        
        # If contours are enabled, add them to the display
        if show_contours:
            # Convert to BGR to draw colored contours
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            
            # Find contours
            all_contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours in yellow
            cv2.drawContours(display_img, all_contours, -1, CONTOUR_COLOR, 2)
            
            # If points are selected, draw them and the path
            if len(selected_points) > 0:
                for point in selected_points:
                    cv2.circle(display_img, point, 3, POINT_COLOR, -1)  # Smaller blue circles
                    
                # If 2 points are selected, draw the path
                if len(selected_points) == 2 and selected_contour is not None:
                    current_path = draw_shortest_path()
                    
                    # If grid lines should be shown, draw them
                    if show_grid_lines and len(current_path) > 0:
                        draw_grid_lines()
    
    # Display the image
    cv2.imshow(WINDOW_NAME, display_img)

def find_closest_contour_point(click_point):
    """Find the closest point on any contour to the clicked point"""
    global all_contours, selected_contour, contour_points
    
    min_dist = float('inf')
    closest_point = None
    closest_contour = None
    
    for contour in all_contours:
        for i in range(len(contour)):
            point = tuple(contour[i][0])
            dist = math.sqrt((click_point[0] - point[0])**2 + (click_point[1] - point[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_point = point
                closest_contour = contour
    
    # If we're selecting the first point, store the contour and extract its points
    if len(selected_points) == 0 and closest_contour is not None:
        selected_contour = closest_contour
        contour_points = [tuple(point[0]) for point in selected_contour]
    
    return closest_point if min_dist < 20 else None  # Only return point if it's close enough (within 20 pixels)

def find_shortest_path():
    """Find the shortest path between two points along the contour"""
    global contour_points, selected_points
    
    if len(selected_points) != 2 or selected_contour is None:
        return []
    
    # Find indices of the closest contour points to the selected points
    indices = []
    for selected_point in selected_points:
        min_dist = float('inf')
        closest_idx = -1
        
        for i, point in enumerate(contour_points):
            dist = math.sqrt((selected_point[0] - point[0])**2 + (selected_point[1] - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        indices.append(closest_idx)
    
    # Find the shortest path (could be clockwise or counterclockwise along the contour)
    idx1, idx2 = indices
    total_points = len(contour_points)
    
    # Path 1: from idx1 to idx2
    if idx1 <= idx2:
        path1 = contour_points[idx1:idx2+1]
    else:
        path1 = contour_points[idx1:] + contour_points[:idx2+1]
    
    # Path 2: from idx2 to idx1 (the other way around)
    if idx2 <= idx1:
        path2 = contour_points[idx2:idx1+1]
    else:
        path2 = contour_points[idx2:] + contour_points[:idx1+1]
    
    # Return the shorter path
    return path1 if len(path1) <= len(path2) else path2

def draw_shortest_path():
    """Draw the shortest path between the two selected points and return the path"""
    global display_img, selected_points
    
    path = find_shortest_path()
    
    if len(path) >= 2:
        # Draw the path in red (thinner)
        for i in range(len(path) - 1):
            cv2.line(display_img, path[i], path[i+1], PATH_COLOR, 1)  # Thinner red line
    
    return path

def find_topmost_point(path):
    """Find the point with the highest y-coordinate (lowest value in image coordinates)"""
    if not path:
        return None
    
    # In image coordinates, y increases as you go down, so we find the minimum y
    topmost_point = min(path, key=lambda p: p[1])
    return topmost_point

def find_bottommost_left_point(path, topmost_point):
    """Find the bottom-most point to the left of the highest point in the path"""
    if not path or not topmost_point:
        return None
    
    # Filter points that are to the left of the topmost point (smaller x value)
    left_points = [p for p in path if p[0] < topmost_point[0]]
    
    if not left_points:
        return None
    
    # Find point with maximum y value (in image coordinates, higher y means lower position)
    bottommost_left_point = max(left_points, key=lambda p: p[1])
    return bottommost_left_point

def find_path_middle_horizontal_intersection(path, topmost_point, bottommost_left_point):
    """Find the intersection of middle horizontal line with the path, to the LEFT of first vertical line"""
    if not path or not topmost_point or not bottommost_left_point:
        return None
    
    # First vertical line x-coordinate
    first_vertical_x = topmost_point[0]
    
    # Calculate middle y-coordinate
    middle_y = (topmost_point[1] + bottommost_left_point[1]) // 2
    
    # Store all intersections of the path with the middle horizontal line
    intersections = []
    
    # Process each segment of the path
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        y1, y2 = p1[1], p2[1]
        x1, x2 = p1[0], p2[0]
        
        # Check if this segment crosses the middle horizontal line
        if (y1 <= middle_y <= y2) or (y2 <= middle_y <= y1):
            # If it's a horizontal segment exactly at middle_y
            if y1 == middle_y and y2 == middle_y:
                # Take the midpoint
                intersections.append(((x1 + x2) // 2, middle_y))
            else:
                # Calculate the precise intersection using linear interpolation
                try:
                    # t is the fractional distance along the segment
                    t = (middle_y - y1) / (y2 - y1)
                    # Calculate the x-coordinate at the intersection
                    x = int(x1 + t * (x2 - x1))
                    intersections.append((x, middle_y))
                except ZeroDivisionError:
                    # This should only happen if y1 == y2, which we already handled
                    pass
    
    # Debug: print all found intersections
    print(f"All middle horizontal line intersections: {intersections}")
    
    # Filter to only include points to the LEFT of the first vertical line
    left_intersections = [p for p in intersections if p[0] < first_vertical_x]
    
    if not left_intersections:
        print("Warning: No intersection found to the left of first vertical line!")
        return None
    
    # Pick the rightmost of the left intersections (i.e., the closest one to the vertical line)
    result = max(left_intersections, key=lambda p: p[0])
    print(f"Selected intersection point: {result}")
    
    return result

def draw_grid_lines():
    """Draw horizontal and vertical lines through specific points on the path"""
    global display_img, current_path
    
    if not current_path:
        return
    
    # Get image dimensions
    height, width = display_img.shape[:2]
    
    # Find the topmost point
    topmost_point = find_topmost_point(current_path)
    if not topmost_point:
        return
    
    # Find the bottom-most point to the left of the highest point
    bottommost_left_point = find_bottommost_left_point(current_path, topmost_point)
    if not bottommost_left_point:
        print("No points found to the left of the topmost point")
        return
    
    # Mark the topmost point with a smaller magenta circle
    cv2.circle(display_img, topmost_point, 3, MARKER_COLOR, -1)  # Smaller magenta circle
    
    # Mark the bottommost-left point with a smaller magenta circle
    cv2.circle(display_img, bottommost_left_point, 3, MARKER_COLOR, -1)  # Smaller magenta circle
    
    # Calculate the middle y-coordinate between the two horizontal lines
    middle_y = (topmost_point[1] + bottommost_left_point[1]) // 2
    print(f"Drawing middle horizontal line at y={middle_y}")
    
    # Find where the middle horizontal line intersects with the path to the LEFT of first vertical
    middle_intersection = find_path_middle_horizontal_intersection(current_path, topmost_point, bottommost_left_point)
    
    # Draw first horizontal line through topmost point (magenta, thin)
    cv2.line(display_img, (0, topmost_point[1]), (width, topmost_point[1]), TOP_LINE_COLOR, 1)
    
    # Draw second horizontal line through bottommost-left point (orange, thin)
    cv2.line(display_img, (0, bottommost_left_point[1]), (width, bottommost_left_point[1]), BOTTOM_LINE_COLOR, 1)
    
    # Draw the middle horizontal line (green and thicker for better visibility)
    cv2.line(display_img, (0, middle_y), (width, middle_y), MIDDLE_LINE_COLOR, 2)
    
    # Draw the first vertical line through the topmost point (blue-violet, thin)
    cv2.line(display_img, (topmost_point[0], 0), (topmost_point[0], height), FIRST_VERT_COLOR, 1)
    
    # If an intersection was found, draw the second vertical line through it
    if middle_intersection:
        cv2.circle(display_img, middle_intersection, 3, MARKER_COLOR, -1)  # Smaller magenta circle
        cv2.line(display_img, (middle_intersection[0], 0), (middle_intersection[0], height), SECOND_VERT_COLOR, 1)
        print(f"Drawing second vertical line at x={middle_intersection[0]}")
    else:
        print("No middle intersection point found")
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display_img, f"Top Point: ({topmost_point[0]}, {topmost_point[1]})", 
                (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display_img, f"Bottom-Left Point: ({bottommost_left_point[0]}, {bottommost_left_point[1]})", 
                (10, 60), font, 0.7, (255, 255, 255), 2)
    if middle_intersection:
        cv2.putText(display_img, f"Middle Intersection: ({middle_intersection[0]}, {middle_intersection[1]})", 
                    (10, 90), font, 0.7, (255, 255, 255), 2)
                    
def draw_grid_on_original():
    """Draw grid lines on the original image"""
    global display_img, current_path
    
    if not current_path:
        return
    
    # Get image dimensions
    height, width = display_img.shape[:2]
    
    # Find the topmost point
    topmost_point = find_topmost_point(current_path)
    if not topmost_point:
        return
    
    # Find the bottom-most point to the left of the highest point
    bottommost_left_point = find_bottommost_left_point(current_path, topmost_point)
    if not bottommost_left_point:
        print("No points found to the left of the topmost point")
        return
    
    # Calculate the middle y-coordinate between the two horizontal lines
    middle_y = (topmost_point[1] + bottommost_left_point[1]) // 2
    print(f"Drawing middle horizontal line at y={middle_y}")
    
    # Find where the middle horizontal line intersects with the path to the LEFT of first vertical
    middle_intersection = find_path_middle_horizontal_intersection(current_path, topmost_point, bottommost_left_point)
    
    # Draw the shortest path in red (thinner)
    if len(current_path) >= 2:
        for i in range(len(current_path) - 1):
            cv2.line(display_img, current_path[i], current_path[i+1], PATH_COLOR, 1)  # Thinner red line
    
    # Mark key points
    cv2.circle(display_img, topmost_point, 3, MARKER_COLOR, -1)  # Topmost point (magenta)
    cv2.circle(display_img, bottommost_left_point, 3, MARKER_COLOR, -1)  # Bottommost left point (magenta)
    
    # Draw grid lines with different colors
    # Top horizontal (magenta)
    cv2.line(display_img, (0, topmost_point[1]), (width, topmost_point[1]), TOP_LINE_COLOR, 1)
    
    # Bottom horizontal (orange)
    cv2.line(display_img, (0, bottommost_left_point[1]), (width, bottommost_left_point[1]), BOTTOM_LINE_COLOR, 1)
    
    # Draw the middle horizontal line (green for better visibility)
    cv2.line(display_img, (0, middle_y), (width, middle_y), MIDDLE_LINE_COLOR, 2)
    
    # First vertical (blue-violet)
    cv2.line(display_img, (topmost_point[0], 0), (topmost_point[0], height), FIRST_VERT_COLOR, 1)
    
    # Draw intersection point
    if middle_intersection:
        cv2.circle(display_img, middle_intersection, 3, MARKER_COLOR, -1)  # Middle intersection (magenta)
        
        # Second vertical line through the intersection (purple)
        cv2.line(display_img, (middle_intersection[0], 0), (middle_intersection[0], height), SECOND_VERT_COLOR, 1)
        print(f"Drawing second vertical line at x={middle_intersection[0]}")
    else:
        print("No middle intersection found, can't draw second vertical line")
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display_img, f"Top: ({topmost_point[0]}, {topmost_point[1]})", 
                (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display_img, f"Bottom-Left: ({bottommost_left_point[0]}, {bottommost_left_point[1]})", 
                (10, 60), font, 0.7, (255, 255, 255), 2)
    if middle_intersection:
        cv2.putText(display_img, f"Middle: ({middle_intersection[0]}, {middle_intersection[1]})", 
                    (10, 90), font, 0.7, (255, 255, 255), 2)

def highlight_boxes():
    """Highlight the box between middle and bottom horizontal lines, and a box of the same size to its right"""
    global display_img, current_path
    
    if not current_path:
        return
    
    # Get the key points from the grid
    topmost_point = find_topmost_point(current_path)
    bottommost_left_point = find_bottommost_left_point(current_path, topmost_point)
    
    if not topmost_point or not bottommost_left_point:
        return
    
    # Calculate middle y-coordinate
    middle_y = (topmost_point[1] + bottommost_left_point[1]) // 2
    
    # Find middle intersection
    middle_intersection = find_path_middle_horizontal_intersection(current_path, topmost_point, bottommost_left_point)
    
    if not middle_intersection:
        return
    
    # Define the center box coordinates between the two vertical lines, from middle to bottom horizontal
    center_box_top_left = (middle_intersection[0], middle_y)
    center_box_bottom_right = (topmost_point[0], bottommost_left_point[1])
    
    # Calculate the width of the center box
    box_width = center_box_bottom_right[0] - center_box_top_left[0]
    
    # Define the right box coordinates (same size, to the right of first vertical line)
    right_box_top_left = (topmost_point[0], middle_y)
    right_box_bottom_right = (topmost_point[0] + box_width, bottommost_left_point[1])
    
    # Create a copy of the image for overlay
    overlay = display_img.copy()
    
    # Fill the center box with translucent yellow
    cv2.rectangle(overlay, center_box_top_left, center_box_bottom_right, (0, 255, 255), -1)  # Yellow fill
    
    # Fill the right box with translucent yellow
    cv2.rectangle(overlay, right_box_top_left, right_box_bottom_right, (0, 255, 255), -1)  # Yellow fill
    
    # Apply the overlay with transparency
    alpha = 0.3  # Transparency factor (0.3 = 30% opacity)
    cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
    
    # Draw borders around the boxes for better visibility
    cv2.rectangle(display_img, center_box_top_left, center_box_bottom_right, (0, 200, 255), 1)  # Orange border
    cv2.rectangle(display_img, right_box_top_left, right_box_bottom_right, (0, 200, 255), 1)  # Orange border

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for point selection"""
    global point_selection_mode, selected_points, display_img
    
    if not point_selection_mode or not show_contours:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 2:
            # Find the closest point on the contour
            closest_point = find_closest_contour_point((x, y))
            
            if closest_point:
                selected_points.append(closest_point)
                
                # Update display
                on_threshold_change(current_threshold)
                
                if len(selected_points) == 2:
                    print("Two points selected. Press 'C' to clear points or continue.")

def main():
    global original_image, output_path, show_contours, point_selection_mode, selected_points
    global show_grid_lines, show_original_with_grid, highlight_grid_boxes
    
    # Read the image
    original_image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Error: Could not read the image at {IMAGE_PATH}")
        return
    
    # Prepare output path
    filename, extension = os.path.splitext(IMAGE_PATH)
    output_path = f"{filename}_thresholdFinal{extension}"
    
    # Create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Set mouse callback
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    
    # Initial threshold value
    initial_threshold = 128
    max_value = 255
    
    # Apply initial threshold
    _, thresholded = cv2.threshold(original_image, initial_threshold, max_value, cv2.THRESH_BINARY)
    
    # Display initial image
    cv2.imshow(WINDOW_NAME, thresholded)
    
    # Create trackbar
    cv2.createTrackbar("Threshold", WINDOW_NAME, initial_threshold, max_value, on_threshold_change)
    
    print("Controls:")
    print("- Adjust the slider to change threshold")
    print("- Press 'Enter' to toggle contour detection")
    print("- Press 'P' to enter point selection mode (after contours are shown)")
    print("- Press 'G' to show grid lines through the topmost point (after path is drawn)")
    print("- Press 'D' to superimpose grid on the original image (after grid lines are shown)")
    print("- Press 'H' to highlight boxes between middle and bottom lines")
    print("- Press 'C' to clear selected points")
    print("- Press 'S' to save the current image and exit")
    print("- Press 'ESC' to exit without saving")
    
    # Wait for key press
    while True:
        key = cv2.waitKey(100) & 0xFF
        
        # ESC key to exit without saving
        if key == 27:  # ESC key
            print("Exiting without saving")
            break
        
        # Enter key to toggle contours
        elif key == 13:  # Enter key
            show_contours = not show_contours
            if not show_contours:
                point_selection_mode = False
                selected_points = []
                show_grid_lines = False
                show_original_with_grid = False
                highlight_grid_boxes = False
            print(f"Contour detection {'enabled' if show_contours else 'disabled'}")
            on_threshold_change(current_threshold)  # Refresh display
        
        # 'P' key to enter point selection mode
        elif key == ord('p') or key == ord('P'):
            if show_contours:
                point_selection_mode = not point_selection_mode
                if point_selection_mode:
                    selected_points = []
                    show_grid_lines = False
                    show_original_with_grid = False
                    highlight_grid_boxes = False
                    print("Point selection mode enabled. Click on contour to select points (max 2).")
                else:
                    print("Point selection mode disabled.")
            else:
                print("Please enable contours first by pressing Enter.")
        
        # 'G' key to show grid lines
        elif key == ord('g') or key == ord('G'):
            if len(selected_points) == 2 and len(current_path) > 0:
                show_grid_lines = not show_grid_lines
                show_original_with_grid = False  # Turn off original view when toggling grid
                highlight_grid_boxes = False
                print(f"Grid lines {'enabled' if show_grid_lines else 'disabled'}")
                on_threshold_change(current_threshold)  # Refresh display
            else:
                print("Please select two points on the contour first.")
        
        # 'D' key to superimpose grid on original image
        elif key == ord('d') or key == ord('D'):
            if show_grid_lines and len(current_path) > 0:
                show_original_with_grid = not show_original_with_grid
                highlight_grid_boxes = False  # Reset highlight when toggling superimposed view
                print(f"Original image with grid {'enabled' if show_original_with_grid else 'disabled'}")
                on_threshold_change(current_threshold)  # Refresh display
            else:
                print("Please enable grid lines first by pressing G.")
                
        # 'H' key to highlight boxes
        elif key == ord('h') or key == ord('H'):
            if show_original_with_grid and len(current_path) > 0:
                highlight_grid_boxes = not highlight_grid_boxes
                print(f"Box highlighting {'enabled' if highlight_grid_boxes else 'disabled'}")
                on_threshold_change(current_threshold)  # Refresh display
            else:
                print("Please superimpose grid on original image first by pressing D.")
        
        # 'C' key to clear selected points
        elif key == ord('c') or key == ord('C'):
            selected_points = []
            show_grid_lines = False
            show_original_with_grid = False
            highlight_grid_boxes = False
            print("Selected points cleared.")
            on_threshold_change(current_threshold)  # Refresh display
        
        # 'S' key to save and exit
        elif key == ord('s') or key == ord('S'):
            # Save the current display image
            if display_img is not None:
                cv2.imwrite(output_path, display_img)
                print(f"Final image saved as {output_path}")
            break
    
    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()