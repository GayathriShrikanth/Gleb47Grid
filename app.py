import os
import cv2
import numpy as np
import math
import uuid
import json
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Create a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple) and hasattr(obj, '_fields'):  # For namedtuple
            return dict(zip(obj._fields, obj))
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)

# Configure app to use our custom JSON encoder
app.json_encoder = NumpyEncoder

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Define colorful line colors
PATH_COLOR = (0, 0, 255)        # Red for path
TOP_LINE_COLOR = (0, 255, 0)    # Magenta for top horizontal line
BOTTOM_LINE_COLOR = (0, 255, 0)  # Orange for bottom horizontal line
MIDDLE_LINE_COLOR = (0, 255, 0)  # Green for middle horizontal line
FIRST_VERT_COLOR = (0, 255, 0)  # Blue-violet for first vertical line
SECOND_VERT_COLOR = (0, 255, 0)  # Purple for second vertical line
CONTOUR_COLOR = (0, 165, 255)    # Orange for contours
POINT_COLOR = (255, 100, 0)      # Blue for selected points
MARKER_COLOR = (255, 0, 255)     # Magenta for marker points
OCHRE_COLOR = (34, 119, 204)     # Ochre color (BGR format)

# Session data storage
sessions = {}

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_data(session_id):
    """Get session data or create new if not exists"""
    if session_id not in sessions:
        sessions[session_id] = {
            'original_image': None,
            'processed_image': None,
            'threshold': 128,
            'show_contours': False,
            'all_contours': [],
            'selected_contour': None,
            'selected_points': [],
            'contour_points': [],
            'current_path': [],
            'show_grid_lines': False,
            'show_original_with_grid': False,
            'highlight_grid_boxes': False,
            'filename': None,
            'file_path': None,
            'processed_path': None,
            'step': 'upload'  # Steps: upload, threshold, contours, selection, grid, original, highlight
        }
    return sessions[session_id]

def find_topmost_point(path):
    """Find the point with the highest y-coordinate (lowest value in image coordinates)"""
    if not path:
        return None
    
    # In image coordinates, y increases as you go down, so we find the minimum y
    topmost_point = min(path, key=lambda p: p[1])
    return tuple(map(int, topmost_point))  # Convert to plain Python ints

def find_bottommost_left_point(path, topmost_point):
    """Find the bottom-most point to the left of the highest point in the path"""
    if not path or topmost_point is None:
        return None
    
    # Filter points that are to the left of the topmost point (smaller x value)
    left_points = [p for p in path if p[0] < topmost_point[0]]
    
    if not left_points:
        return None
    
    # Find point with maximum y value (in image coordinates, higher y means lower position)
    bottommost_left_point = max(left_points, key=lambda p: p[1])
    return tuple(map(int, bottommost_left_point))  # Convert to plain Python ints

def find_path_middle_horizontal_intersection(path, topmost_point, bottommost_left_point):
    """Find the intersection of middle horizontal line with the path, to the LEFT of first vertical line"""
    if not path or topmost_point is None or bottommost_left_point is None:
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
                intersections.append((int((x1 + x2) // 2), int(middle_y)))
            else:
                # Calculate the precise intersection using linear interpolation
                try:
                    # t is the fractional distance along the segment
                    t = (middle_y - y1) / (y2 - y1)
                    # Calculate the x-coordinate at the intersection
                    x = int(x1 + t * (x2 - x1))
                    intersections.append((int(x), int(middle_y)))
                except ZeroDivisionError:
                    # This should only happen if y1 == y2, which we already handled
                    pass
    
    # Filter to only include points to the LEFT of the first vertical line
    left_intersections = [p for p in intersections if p[0] < first_vertical_x]
    
    if not left_intersections:
        return None
    
    # Pick the rightmost of the left intersections (i.e., the closest one to the vertical line)
    result = max(left_intersections, key=lambda p: p[0])
    return tuple(map(int, result))  # Convert to plain Python ints

def find_shortest_path(contour_points, selected_points):
    """Find the shortest path between two points along the contour"""
    if len(selected_points) != 2 or not contour_points:
        return []
    
    # Find indices of the closest contour points to the selected points
    indices = []
    for selected_point in selected_points:
        min_dist = float('inf')
        closest_idx = -1
        
        for i, point in enumerate(contour_points):
            # Ensure point is tuple of integers
            point = tuple(map(int, point))
            selected_point = tuple(map(int, selected_point))
            
            dist = math.sqrt((selected_point[0] - point[0])**2 + (selected_point[1] - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        indices.append(closest_idx)
    
    # Find the shortest path (could be clockwise or counterclockwise along the contour)
    idx1, idx2 = indices
    total_points = len(contour_points)
    
    # Make sure indices are integers
    idx1, idx2 = int(idx1), int(idx2)
    
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
    
    # Convert all points to plain Python tuples with integer coordinates
    path1 = [tuple(map(int, p)) for p in path1]
    path2 = [tuple(map(int, p)) for p in path2]
    
    # Return the shorter path
    return path1 if len(path1) <= len(path2) else path2

def process_image(session_id):
    """Process the image with current settings"""
    data = get_session_data(session_id)
    
    if data['original_image'] is None:
        return None
    
    # Create a copy of the original image
    if data['show_original_with_grid'] and len(data['current_path']) > 0:
        # Convert grayscale to BGR if needed
        if len(data['original_image'].shape) == 2:
            display_img = cv2.cvtColor(data['original_image'], cv2.COLOR_GRAY2BGR)
        else:
            display_img = data['original_image'].copy()
            
        # Draw grid on original
        draw_grid_on_original(display_img, data)
        
        # Highlight boxes if enabled
        if data['highlight_grid_boxes']:
            highlight_boxes(display_img, data)
    else:
        # Apply threshold
        _, thresholded = cv2.threshold(data['original_image'], data['threshold'], 255, cv2.THRESH_BINARY)
        
        # Create display image
        display_img = thresholded.copy()
        
        # If contours are enabled, add them to the display
        if data['show_contours']:
            # Convert to BGR to draw colored contours
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            
            # Find contours
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            data['all_contours'] = contours
            
            # Draw contours
            cv2.drawContours(display_img, contours, -1, CONTOUR_COLOR, 1)
            
            # If points are selected, draw them and the path
            if len(data['selected_points']) > 0:
                for point in data['selected_points']:
                    point = tuple(map(int, point))  # Ensure point is integer tuple
                    cv2.circle(display_img, point, 2, POINT_COLOR, -1)
                    
                # If 2 points are selected, draw the path
                if len(data['selected_points']) == 2 and data['selected_contour'] is not None:
                    path = find_shortest_path(data['contour_points'], data['selected_points'])
                    data['current_path'] = path
                    
                    # Draw path
                    if len(path) >= 2:
                        for i in range(len(path) - 1):
                            # Ensure points are integer tuples
                            pt1 = tuple(map(int, path[i]))
                            pt2 = tuple(map(int, path[i+1]))
                            cv2.line(display_img, pt1, pt2, PATH_COLOR, 1)
                    
                    # If grid lines should be shown, draw them
                    if data['show_grid_lines'] and len(path) > 0:
                        draw_grid_lines(display_img, data)
    
    # Save the processed image
    processed_filename = f"processed_{uuid.uuid4().hex}.jpg"
    data['processed_path'] = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    cv2.imwrite(data['processed_path'], display_img)
    data['processed_image'] = display_img
    
    return processed_filename

def draw_grid_lines(display_img, data):
    """Draw horizontal and vertical lines through specific points on the path"""
    path = data['current_path']
    
    if not path:
        return
    
    # Get image dimensions
    height, width = display_img.shape[:2]
    
    # Find the topmost point
    topmost_point = find_topmost_point(path)
    if not topmost_point:
        return
    
    # Find the bottom-most point to the left of the highest point
    bottommost_left_point = find_bottommost_left_point(path, topmost_point)
    if not bottommost_left_point:
        return
    
    # Mark the topmost point with a smaller magenta circle
    cv2.circle(display_img, topmost_point, 2, MARKER_COLOR, -1)
    
    # Mark the bottommost-left point with a smaller magenta circle
    cv2.circle(display_img, bottommost_left_point, 2, MARKER_COLOR, -1)
    
    # Calculate the middle y-coordinate between the two horizontal lines
    middle_y = int((topmost_point[1] + bottommost_left_point[1]) // 2)
    
    # Find where the middle horizontal line intersects with the path to the LEFT of first vertical
    middle_intersection = find_path_middle_horizontal_intersection(path, topmost_point, bottommost_left_point)
    
    # Draw first horizontal line through topmost point (magenta)
    cv2.line(display_img, (0, topmost_point[1]), (width, topmost_point[1]), TOP_LINE_COLOR, 1)
    
    # Draw second horizontal line through bottommost-left point (orange)
    cv2.line(display_img, (0, bottommost_left_point[1]), (width, bottommost_left_point[1]), BOTTOM_LINE_COLOR, 1)
    
    # Draw the middle horizontal line (green and slightly thicker for better visibility)
    cv2.line(display_img, (0, middle_y), (width, middle_y), MIDDLE_LINE_COLOR, 1)
    
    # Draw the first vertical line through the topmost point (blue-violet)
    cv2.line(display_img, (topmost_point[0], 0), (topmost_point[0], height), FIRST_VERT_COLOR, 1)
    
    # If an intersection was found, draw the second vertical line through it
    if middle_intersection:
        cv2.circle(display_img, middle_intersection, 2, MARKER_COLOR, -1)
        cv2.line(display_img, (middle_intersection[0], 0), (middle_intersection[0], height), SECOND_VERT_COLOR, 1)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display_img, f"Top: ({topmost_point[0]}, {topmost_point[1]})", 
                (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display_img, f"Bottom-Left: ({bottommost_left_point[0]}, {bottommost_left_point[1]})", 
                (10, 60), font, 0.7, (255, 255, 255), 2)
    if middle_intersection:
        cv2.putText(display_img, f"Middle: ({middle_intersection[0]}, {middle_intersection[1]})", 
                    (10, 90), font, 0.7, (255, 255, 255), 2)

def draw_grid_on_original(display_img, data):
    """Draw grid lines on the original image"""
    path = data['current_path']
    
    if not path:
        return
    
    # Get image dimensions
    height, width = display_img.shape[:2]
    
    # Find the topmost point
    topmost_point = find_topmost_point(path)
    if not topmost_point:
        return
    
    # Find the bottom-most point to the left of the highest point
    bottommost_left_point = find_bottommost_left_point(path, topmost_point)
    if not bottommost_left_point:
        return
    
    # Calculate the middle y-coordinate between the two horizontal lines
    middle_y = int((topmost_point[1] + bottommost_left_point[1]) // 2)
    
    # Find where the middle horizontal line intersects with the path to the LEFT of first vertical
    middle_intersection = find_path_middle_horizontal_intersection(path, topmost_point, bottommost_left_point)
    
    # Draw the shortest path in red (thinner)
    if len(path) >= 2:
        for i in range(len(path) - 1):
            # Ensure points are integer tuples
            pt1 = tuple(map(int, path[i]))
            pt2 = tuple(map(int, path[i+1]))
            cv2.line(display_img, pt1, pt2, PATH_COLOR, 1)
    
    # Mark key points
    cv2.circle(display_img, topmost_point, 2, MARKER_COLOR, -1)
    cv2.circle(display_img, bottommost_left_point, 2, MARKER_COLOR, -1)
    
    # Draw grid lines with different colors
    # Top horizontal (magenta)
    cv2.line(display_img, (0, topmost_point[1]), (width, topmost_point[1]), TOP_LINE_COLOR, 1)
    
    # Bottom horizontal (orange)
    cv2.line(display_img, (0, bottommost_left_point[1]), (width, bottommost_left_point[1]), BOTTOM_LINE_COLOR, 1)
    
    # Draw the middle horizontal line (green)
    cv2.line(display_img, (0, middle_y), (width, middle_y), MIDDLE_LINE_COLOR, 1)
    
    # First vertical (blue-violet)
    cv2.line(display_img, (topmost_point[0], 0), (topmost_point[0], height), FIRST_VERT_COLOR, 1)
    
    # Draw intersection point
    if middle_intersection:
        cv2.circle(display_img, middle_intersection, 2, MARKER_COLOR, -1)
        
        # Second vertical line through the intersection (purple)
        cv2.line(display_img, (middle_intersection[0], 0), (middle_intersection[0], height), SECOND_VERT_COLOR, 1)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display_img, f"Top: ({topmost_point[0]}, {topmost_point[1]})", 
                (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display_img, f"Bottom-Left: ({bottommost_left_point[0]}, {bottommost_left_point[1]})", 
                (10, 60), font, 0.7, (255, 255, 255), 2)
    if middle_intersection:
        cv2.putText(display_img, f"Middle: ({middle_intersection[0]}, {middle_intersection[1]})", 
                    (10, 90), font, 0.7, (255, 255, 255), 2)


def highlight_boxes(display_img, data):
    """Highlight the boxes between the two vertical lines, under the middle and bottom horizontal lines"""
    path = data['current_path']
    
    if not path:
        return
    
    # Get the key points from the grid
    topmost_point = find_topmost_point(path)
    bottommost_left_point = find_bottommost_left_point(path, topmost_point)
    
    if not topmost_point or not bottommost_left_point:
        return
    
    # Calculate middle y-coordinate
    middle_y = int((topmost_point[1] + bottommost_left_point[1]) // 2)
    
    # Find middle intersection
    middle_intersection = find_path_middle_horizontal_intersection(path, topmost_point, bottommost_left_point)
    
    if not middle_intersection:
        return
    
    # Define the center box coordinates between the two vertical lines, from middle to bottom horizontal
    center_box_top_left = (int(middle_intersection[0]), int(middle_y))
    center_box_bottom_right = (int(topmost_point[0]), int(bottommost_left_point[1]))
    
    # Define the bottom box coordinates between the two vertical lines, below the bottom horizontal
    # Get the height of existing box to maintain proportions
    box_height = bottommost_left_point[1] - middle_y
    
    bottom_box_top_left = (int(middle_intersection[0]), int(bottommost_left_point[1]))
    bottom_box_bottom_right = (int(topmost_point[0]), int(bottommost_left_point[1] + box_height))
    
    # Create a copy of the image for overlay
    overlay = display_img.copy()
    
    # Fill the center box with translucent ochre
    cv2.rectangle(overlay, center_box_top_left, center_box_bottom_right, OCHRE_COLOR, -1)
    
    # Fill the bottom box with translucent ochre
    cv2.rectangle(overlay, bottom_box_top_left, bottom_box_bottom_right, OCHRE_COLOR, -1)
    
    # Apply the overlay with transparency
    alpha = 0.3  # Transparency factor (0.3 = 30% opacity)
    cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
    
    # Draw borders around the boxes for better visibility
    cv2.rectangle(display_img, center_box_top_left, center_box_bottom_right, (0, 200, 255), 1)  # Orange border
    cv2.rectangle(display_img, bottom_box_top_left, bottom_box_bottom_right, (0, 200, 255), 1)  # Orange border

def find_closest_contour_point(contours, click_point):
    """Find the closest point on any contour to the clicked point"""
    min_dist = float('inf')
    closest_point = None
    closest_contour = None
    
    for contour in contours:
        for i in range(len(contour)):
            # Convert to regular Python tuple to avoid NumPy types
            point = tuple(map(int, contour[i][0]))
            dist = math.sqrt((click_point[0] - point[0])**2 + (click_point[1] - point[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_point = point
                closest_contour = contour
    
    return closest_point, closest_contour if min_dist < 50 else (None, None)

@app.route('/')
def index():
    """Main page"""
    # Create a new session
    session_id = str(uuid.uuid4())
    return render_template('index.html', session_id=session_id)

@app.route('/upload/<session_id>', methods=['POST'])
def upload_file(session_id):
    """Handle file uploads"""
    app.logger.info(f"Received upload request for session {session_id}")
    
    if 'file' not in request.files:
        app.logger.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        app.logger.error("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    app.logger.info(f"Processing file: {file.filename}")
    
    if file and allowed_file(file.filename):
        # Get session data
        data = get_session_data(session_id)
        
        try:
            # Save the file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Ensure directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            app.logger.info(f"Saving file to: {file_path}")
            file.save(file_path)
            
            # Read the image in grayscale with error handling
            app.logger.info("Reading image with OpenCV")
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # Check if image was loaded successfully
            if image is None:
                app.logger.error(f"Failed to load image from {file_path}")
                return jsonify({'error': 'Failed to load image. The file may be corrupted or in an unsupported format.'}), 400
            
            app.logger.info(f"Image loaded successfully, dimensions: {image.shape}")
            
            # Update session data
            data['original_image'] = image
            data['filename'] = unique_filename
            data['file_path'] = file_path
            data['step'] = 'threshold'
            
            # Process and save the image
            app.logger.info("Processing image")
            processed_filename = process_image(session_id)
            
            if processed_filename is None:
                app.logger.error("Failed to process image")
                return jsonify({'error': 'Failed to process image'}), 500
            
            app.logger.info(f"Image processed successfully: {processed_filename}")
            
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'processed': processed_filename,
                'step': data['step']
            })
        except Exception as e:
            app.logger.error(f"Exception during file upload: {str(e)}")
            return jsonify({'error': f'An error occurred during upload: {str(e)}'}), 500
    
    app.logger.error(f"File type not allowed: {file.filename}")
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/threshold/<session_id>', methods=['POST'])
def adjust_threshold(session_id):
    """Adjust the threshold value"""
    data = get_session_data(session_id)
    
    if data['original_image'] is None:
        return jsonify({'error': 'No image uploaded'}), 400
    
    # Get threshold value from request
    threshold = int(request.form.get('threshold', 128))
    data['threshold'] = threshold
    
    # Process the image
    processed_filename = process_image(session_id)
    
    return jsonify({
        'success': True,
        'threshold': threshold,
        'processed': processed_filename,
        'step': data['step']
    })

@app.route('/toggle_contours/<session_id>', methods=['POST'])
def toggle_contours(session_id):
    """Toggle contour detection"""
    data = get_session_data(session_id)
    
    if data['original_image'] is None:
        return jsonify({'error': 'No image uploaded'}), 400
    
    # Toggle contour detection
    data['show_contours'] = not data['show_contours']
    data['step'] = 'contours' if data['show_contours'] else 'threshold'
    
    # Reset selection if contours are disabled
    if not data['show_contours']:
        data['selected_points'] = []
        data['selected_contour'] = None
        data['contour_points'] = []
        data['current_path'] = []
        data['show_grid_lines'] = False
        data['show_original_with_grid'] = False
        data['highlight_grid_boxes'] = False
    
    # Process the image
    processed_filename = process_image(session_id)
    
    return jsonify({
        'success': True,
        'show_contours': data['show_contours'],
        'processed': processed_filename,
        'step': data['step']
    })

@app.route('/select_point/<session_id>', methods=['POST'])
def select_point(session_id):
    """Select a point on the contour"""
    data = get_session_data(session_id)
    
    if not data['show_contours']:
        return jsonify({'error': 'Contours not enabled'}), 400
    
    if len(data['selected_points']) >= 2:
        return jsonify({'error': 'Already selected two points'}), 400
    
    # Get coordinates from request
    x = int(request.form.get('x', 0))
    y = int(request.form.get('y', 0))
    
    # Find closest contour point
    closest_point, closest_contour = find_closest_contour_point(data['all_contours'], (x, y))
    
    if closest_point is None:
        return jsonify({'error': 'No contour point found near click'}), 400
    
    # Store the point
    data['selected_points'].append(closest_point)
    
    # If this is the first point, store the contour
    if len(data['selected_points']) == 1 and closest_contour is not None:
        data['selected_contour'] = closest_contour
        # Convert to regular Python tuples to avoid NumPy types
        data['contour_points'] = [tuple(map(int, point[0])) for point in closest_contour]
    
    data['step'] = 'selection'
    
    # If two points are selected, calculate the path
    if len(data['selected_points']) == 2:
        data['step'] = 'grid'
    
    # Process the image
    processed_filename = process_image(session_id)
    
    # Convert points to regular Python types for JSON serialization
    points_json = [list(map(int, point)) for point in data['selected_points']]
    
    return jsonify({
        'success': True,
        'points': points_json,
        'processed': processed_filename,
        'step': data['step'],
        'point_count': len(data['selected_points'])
    })

@app.route('/toggle_grid/<session_id>', methods=['POST'])
def toggle_grid(session_id):
    """Toggle grid lines"""
    data = get_session_data(session_id)
    
    if len(data['selected_points']) != 2 or len(data['current_path']) == 0:
        return jsonify({'error': 'Need to select two points first'}), 400
    
    # Toggle grid lines
    data['show_grid_lines'] = not data['show_grid_lines']
    data['step'] = 'grid'
    
    # Reset original view and highlighting when toggling grid
    data['show_original_with_grid'] = False
    data['highlight_grid_boxes'] = False
    
    # Process the image
    processed_filename = process_image(session_id)
    
    return jsonify({
        'success': True,
        'show_grid_lines': data['show_grid_lines'],
        'processed': processed_filename,
        'step': data['step']
    })

@app.route('/toggle_original/<session_id>', methods=['POST'])
def toggle_original(session_id):
    """Toggle original image with grid overlay"""
    data = get_session_data(session_id)
    
    if not data['show_grid_lines'] or len(data['current_path']) == 0:
        return jsonify({'error': 'Grid lines must be enabled first'}), 400
    
    # Toggle original image with grid
    data['show_original_with_grid'] = not data['show_original_with_grid']
    data['step'] = 'original' if data['show_original_with_grid'] else 'grid'
    
    # Reset highlighting when toggling original view
    data['highlight_grid_boxes'] = False
    
    # Process the image
    processed_filename = process_image(session_id)
    
    return jsonify({
        'success': True,
        'show_original_with_grid': data['show_original_with_grid'],
        'processed': processed_filename,
        'step': data['step']
    })

@app.route('/toggle_highlight/<session_id>', methods=['POST'])
def toggle_highlight(session_id):
    """Toggle box highlighting"""
    data = get_session_data(session_id)
    
    if not data['show_original_with_grid'] or len(data['current_path']) == 0:
        return jsonify({'error': 'Original image with grid must be shown first'}), 400
    
    # Toggle box highlighting
    data['highlight_grid_boxes'] = not data['highlight_grid_boxes']
    data['step'] = 'highlight' if data['highlight_grid_boxes'] else 'original'
    
    # Process the image
    processed_filename = process_image(session_id)
    
    return jsonify({
        'success': True,
        'highlight_grid_boxes': data['highlight_grid_boxes'],
        'processed': processed_filename,
        'step': data['step']
    })

@app.route('/reset/<session_id>', methods=['POST'])
def reset_session(session_id):
    """Reset the session"""
    data = get_session_data(session_id)
    
    # Reset session data
    data['selected_points'] = []
    data['selected_contour'] = None
    data['contour_points'] = []
    data['current_path'] = []
    data['show_grid_lines'] = False
    data['show_original_with_grid'] = False
    data['highlight_grid_boxes'] = False
    data['step'] = 'contours' if data['show_contours'] else 'threshold'
    
    # Process the image
    processed_filename = process_image(session_id)
    
    return jsonify({
        'success': True,
        'processed': processed_filename,
        'step': data['step']
    })

@app.route('/save/<session_id>', methods=['POST'])
def save_image(session_id):
    """Save the processed image"""
    data = get_session_data(session_id)
    
    if data['processed_image'] is None:
        return jsonify({'error': 'No processed image to save'}), 400
    
    # Generate a download filename
    if data['filename']:
        base_name = os.path.splitext(data['filename'])[0]
        download_filename = f"processed_{base_name}.jpg"
    else:
        download_filename = f"processed_{uuid.uuid4().hex}.jpg"
    
    # Copy the processed image to a download location if needed
    download_path = os.path.join(app.config['PROCESSED_FOLDER'], download_filename)
    cv2.imwrite(download_path, data['processed_image'])
    
    # Return success with the download URL
    return jsonify({
        'success': True,
        'download_url': url_for('download_file', filename=download_filename)
    })

@app.route('/downloads/<filename>')
def download_file(filename):
    """Handle file downloads"""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

# Add a route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Add a placeholder route
@app.route('/static/placeholder.jpg')
def serve_placeholder():
    # Create a placeholder image if it doesn't exist
    placeholder_path = os.path.join('static', 'placeholder.jpg')
    if not os.path.exists(placeholder_path):
        # Create a simple placeholder image
        placeholder = np.ones((300, 400, 3), dtype=np.uint8) * 200  # Light gray
        cv2.putText(placeholder, "Upload an image", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(placeholder_path, placeholder)
    
    return send_from_directory('static', 'placeholder.jpg')

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
