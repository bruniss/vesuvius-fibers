import sys
import numpy as np
from typing import List
from scipy import ndimage
from scipy.interpolate import splprep, splev
import cc3d
from skimage import measure, morphology
from sklearn.decomposition import PCA
from magicgui import magicgui
from magicgui.widgets import Label, Container
import napari
import napari.layers
from napari.layers import Labels
from napari.utils import notifications
from qtpy.QtCore import Qt
import logging
import colorsys
import random

selected_label_points = set()


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

viewer = napari.Viewer()

@magicgui(
    call_button='run cc3d',
    connectivity={"choices": ["6", "18", "26"]}
)
def run_cc3d(
    target_image: "napari.layers.Labels", 
    connectivity="6"
) -> "napari.layers.Labels":

    if target_image is not None:
        labels_data = target_image.data  
        connectivity = int(connectivity)
        labeled_components = cc3d.connected_components(labels_data, connectivity=connectivity)
        
        # Create the custom name with the specified format
        layer_name = f"{target_image}_{connectivity}-cc"
        
        # Return the layer with the custom name
        return napari.layers.Labels(labeled_components, name=layer_name)

@magicgui(call_button='remove small components',
          min_size={'widget_type': 'SpinBox', 'min': 0, 'max': 1000000},
          connectivity={"choices": ["6", "18", "26"]})
def remove_small_components(
    target_image: "napari.layers.Labels", 
    min_size = 500,
    connectivity="6"
) -> "napari.layers.Labels":
    if target_image is not None:
        connectivity = int(connectivity)
        labels_data = target_image.data.copy() 
        
      
        cc3d.dust(labels_data, 
                  threshold=min_size,
                  connectivity=connectivity,
                  in_place=True)
    
        layer_name = f"{target_image.name}_clean"
        return napari.layers.Labels(labels_data, name=layer_name)
    
@magicgui(auto_call=True)
def set_centerxy(
    center_x: int, 
    center_y: int
) -> np.ndarray:  
    return np.array([center_x, center_y]) 

@magicgui(call_button="extract class from image")
          #extract_all={'widget_type': 'CheckBox', 'value': False}
def extract_label_from_image(
    target_image: "napari.layers.Image", 
    label_value: int, 
    label_name: str,
    #extract_all: bool
) -> None:  
    binary_mask = (target_image.data == label_value).astype(int)
    viewer.add_labels(binary_mask, name=label_name) 
    

@magicgui(
    call_button="find fibers",
    fiber_type={"choices": ["vertical", "horizontal"]},
    direction={"choices": ["inside", "outside"]},
    src_label_id={'widget_type': 'SpinBox', 'min': 0, 'max': 1000000}
)
def find_fibers(
    src_labels: "napari.layers.Labels",
    target_labels: "napari.layers.Labels",
    src_label_id: int,
    max_steps: int,
    fiber_type="vertical",
    direction="inside"
) -> None:
    print(f"Starting find_fibers with: fiber_type={fiber_type}, direction={direction}")
    
    existing_layers = {layer.name: layer for layer in viewer.layers}
    found_labels_layer = existing_layers.get("found labels", None)
    
    # Initialize or copy the labels data
    if found_labels_layer is not None:
        labels_data = found_labels_layer.data.copy()
    else:
        labels_data = np.zeros_like(target_labels.data, dtype=np.uint8)
    
    # Find new fibers
    new_fibers = find_labels_relative_to_fiber(
        target_labels=target_labels.data, 
        source_labels=src_labels.data, 
        current_label=src_label_id,
        center_xy=set_centerxy(),
        direction=direction,
        max_steps=max_steps
    )
    print(f"Fibers found: {new_fibers}")

    # Add the new fibers to labels_data
    for fiber in new_fibers:
        labels_data[target_labels.data == fiber] = fiber
    
    # Also add the source label
    labels_data[src_labels.data == src_label_id] = src_label_id

    # Update or add the labels layer
    if found_labels_layer is not None:
        found_labels_layer.data = labels_data
        found_labels_layer.refresh()
        print("Updated 'found labels' layer.")
    else:
        viewer.add_labels(labels_data, name="found labels")
        print("Added 'found labels' layer.")

    if not new_fibers:
        print("No new fibers found or added.")

    
@magicgui(
    call_button="Create z nodes for label",
    target_label_list={"widget_type": "LineEdit"}
)
def create_nodes_from_label(
    src_labels: "napari.layers.Labels",
    target_label_list: str,
    interval: int
) -> "List[napari.layers.Points]":
    
    try:
        label_list = [int(label.strip()) for label in target_label_list.split(',')]
    except ValueError:
        notifications.show_error("Invalid input for target labels. Please enter comma-separated integers.")
        return []
    
    if not label_list:
        notifications.show_error("No valid labels provided")
        
    created_layers = []
    
    for lbl in label_list:
        
        nodes = []
        tgt_lbl = (src_labels.data == lbl).astype(np.uint8)
        
        if not np.any(tgt_lbl):
            notifications.show_error(f"Label ID {lbl} does not exist in the source labels.")
            continue
        
        z_indices = np.where(tgt_lbl)[0]
        min_z = z_indices.min()
        max_z = z_indices.max()
        print(f"Label {lbl} spans from Z={min_z} to Z={max_z}")

        for current_z in range(min_z, max_z + 1, interval):
            # Define the range for the current chunk
            chunk_start = current_z
            chunk_end = min(current_z + interval, tgt_lbl.shape[0])  # Ensure we don't go out of bounds

            # Extract the chunk for the current interval
            chunk = tgt_lbl[chunk_start:chunk_end, :, :]
            print(f"Processing Z-slices {chunk_start} to {chunk_end - 1}")

            # Calculate the centroid within the chunk
            centroid = ndimage.center_of_mass(chunk)

            if np.isnan(centroid[0]):
                print(f"No voxels found in Z-slices {chunk_start} to {chunk_end - 1}. Skipping.")
                continue  # Skip if no voxels are found in this chunk

            # Adjust the Z-coordinate based on the chunk's position in the volume
            global_centroid_z = centroid[0] + chunk_start
            global_centroid_y = centroid[1]
            global_centroid_x = centroid[2]

            nodes.append([global_centroid_z, global_centroid_y, global_centroid_x])
            print(f"Added centroid at (Z={global_centroid_z}, Y={global_centroid_y}, X={global_centroid_x})")

        if not nodes:
            notifications.show_error("No centroids were created. Check the interval and label distribution.")
            

        points = np.array(nodes)
        print(f"Total points created: {len(points)}")

        
        layer_name = f"Label_{lbl}_Nodes"
        points_layer = viewer.add_points(
            points,
            name=layer_name,
            size=5,
            face_color='red',
            border_color='black'
        )
        created_layers.append(points_layer)

    return created_layers

@magicgui(
    call_button="connect nodes in specified order",
    target_points_list={"widget_type": "LineEdit"},
    z_tolerance={"widget_type": "SpinBox", "min": 0, "max": 100, "step": 1, "value": 5}
)
def create_spline_from_nodes(
    target_points_list: str,
    z_tolerance: int = 5
) -> "napari.layers.Shapes":
    all_points = {}
    all_splines = []
    
    try:
        pointset_list = [int(pointset.strip()) for pointset in target_points_list.split(',')]
    except ValueError:
        notifications.show_error("Invalid input for target points. Please enter comma-separated integers.")
        return None
    
    if len(pointset_list) < 2:
        notifications.show_error("At least two point sets are required to create splines.")
        return None
        
    for tgt_pointset in pointset_list:
        pts_name = f"Label_{tgt_pointset}_Nodes"
        if pts_name in viewer.layers:
            current_points = viewer.layers[pts_name]
            if isinstance(current_points, napari.layers.Points):
                all_points[pts_name] = current_points.data.tolist()  # Convert to list for easier manipulation
            else:
                notifications.show_error(f"Selected points layer {pts_name} is not a valid points layer -- skipped")
        else:
            notifications.show_error(f"Layer {pts_name} not found -- skipped")
    
    # Sort points by z-coordinate for each layer
    for layer_name in all_points:
        all_points[layer_name].sort(key=lambda p: p[0])
    
    # Find the overall z range
    min_z = min(p[0] for points in all_points.values() for p in points)
    max_z = max(p[0] for points in all_points.values() for p in points)
    
    # Create splines
    for current_z in np.arange(min_z, max_z + 1, 1):
        spline_points = []
        for tgt_pointset in pointset_list:
            layer_name = f"Label_{tgt_pointset}_Nodes"
            if layer_name in all_points and all_points[layer_name]:
                # Find the closest point within the z_tolerance
                closest_point = min(
                    (p for p in all_points[layer_name] if abs(p[0] - current_z) <= z_tolerance),
                    key=lambda p: abs(p[0] - current_z),
                    default=None
                )
                if closest_point:
                    spline_points.append(closest_point[1:])  # Only append y and x coordinates
                    all_points[layer_name].remove(closest_point)  # Remove the used point
                else:
                    break  # If we can't find a point for this layer, stop the spline here
        
        if len(spline_points) >= 2:
            points = np.array(spline_points)
            
            # Fit the spline (always open)
            tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=min(3, len(points) - 1))
            
            # Generate points along the spline
            u_new = np.linspace(0, 1, 100)
            y_new, x_new = splev(u_new, tck)
            
            # Create the spline points with the constant z value
            spline_points = np.column_stack((np.full(len(x_new), current_z), y_new, x_new))
            
            # Add the spline to the list of all splines
            all_splines.append(spline_points)
    
    if all_splines:
        spline_layer = viewer.add_shapes(
            all_splines,
            shape_type='path',
            edge_width=2,
            edge_color='cyan',
            name='interp splines from nodes'
        )
        return spline_layer
    else:
        print("No splines were created.")
        return None
    
       
def extract_surface_voxels(label_volume, component_label, center_xy, side='front', is_vertical=True):
    if side not in ['front', 'back']:
        notifications.show_error("side must be either 'front' or 'back'")
    
    component_mask = label_volume == component_label
    bounds = ndimage.find_objects(component_mask)[0]
    
    surface_voxels = []
    
    if is_vertical:
        for z in range(bounds[0].start, bounds[0].stop):
            slice_mask = component_mask[z, bounds[1].start:bounds[1].stop, bounds[2].start:bounds[2].stop]
            
            if not np.any(slice_mask):
                continue
            
            # Use dilation method for boundary detection
            dilated = ndimage.binary_dilation(slice_mask)
            boundaries = dilated ^ slice_mask
            boundary_coords = np.array(np.where(boundaries)).T
            
            if len(boundary_coords) == 0:
                continue
            
            # Adjust coordinates to original image space
            boundary_coords[:, 0] += bounds[1].start
            boundary_coords[:, 1] += bounds[2].start
            
            distances = np.linalg.norm(boundary_coords[:, [1, 0]] - center_xy, axis=1)
            selected_index = np.argmin(distances) if side == 'front' else np.argmax(distances)
            
            y, x = boundary_coords[selected_index]
            surface_voxels.append((z, y, x))
    
    else:  # Horizontal fibers
        orientation = determine_fiber_orientation(component_mask, bounds, center_xy)
        
        for z in range(bounds[0].start, bounds[0].stop):
            slice_mask = component_mask[z, bounds[1].start:bounds[1].stop, bounds[2].start:bounds[2].stop]
            
            if not np.any(slice_mask):
                continue
            
            y_coords, x_coords = np.where(slice_mask)
            
            if len(y_coords) == 0:
                continue
            
            # Adjust coordinates to original image space
            y_coords += bounds[1].start
            x_coords += bounds[2].start
            
            if orientation == 'x':
                if side == 'front':
                    for y in np.unique(y_coords):
                        x = np.min(x_coords[y_coords == y])
                        surface_voxels.append((z, y, x))
                else:  # 'back'
                    for y in np.unique(y_coords):
                        x = np.max(x_coords[y_coords == y])
                        surface_voxels.append((z, y, x))
            
            elif orientation == 'y':
                if side == 'front':
                    for x in np.unique(x_coords):
                        y = np.min(y_coords[x_coords == x])
                        surface_voxels.append((z, y, x))
                else:  # 'back'
                    for x in np.unique(x_coords):
                        y = np.max(y_coords[x_coords == x])
                        surface_voxels.append((z, y, x))
            
            else:  # 'unknown'
                dilated = ndimage.binary_dilation(slice_mask)
                boundaries = dilated ^ slice_mask
                boundary_coords = np.array(np.where(boundaries)).T
                if len(boundary_coords) == 0:
                    continue
                boundary_coords[:, 0] += bounds[1].start
                boundary_coords[:, 1] += bounds[2].start
                distances = np.linalg.norm(boundary_coords[:, [1, 0]] - center_xy, axis=1)
                selected_index = np.argmin(distances) if side == 'front' else np.argmax(distances)
                y, x = boundary_coords[selected_index]
                surface_voxels.append((z, y, x))
    
    
    print(f"Number of surface voxels: {len(surface_voxels)}")
    print(f"First few surface voxels: {surface_voxels[:5]}")
    return np.array(surface_voxels)

def find_labels_relative_to_fiber(target_labels, source_labels, current_label, center_xy, direction='in', max_steps=10):
    
    if direction not in ['inside', 'outside']:
        notifications.show_error("direction must be either 'inside' or 'outside'")
    
    if direction == 'inside':
        side = 'front'
    
    if direction == 'outside':
        side = 'back'

    surface_voxels = extract_surface_voxels(source_labels, current_label, center_xy, side=side)
    
    all_intersected_labels = set()

    for voxel in surface_voxels:
        z, y, x = voxel
        voxel_xy = np.array([x, y])
        direction_xy = center_xy - voxel_xy
        norm_xy = np.linalg.norm(direction_xy)
        
        if norm_xy == 0:
            print('normxy is 0')
            continue

        unit_dir_xy = direction_xy / norm_xy

        # Flip the direction if we're casting rays outward
        if direction == 'outside':
            unit_dir_xy = -unit_dir_xy
            print('casting rays out')

        for step in range(1, max_steps + 1):
            new_x = x + unit_dir_xy[0] * step
            new_y = y + unit_dir_xy[1] * step
            new_x_int = int(round(new_x))
            new_y_int = int(round(new_y))

            if (new_x_int < 0 or new_x_int >= target_labels.shape[2] or
                new_y_int < 0 or new_y_int >= target_labels.shape[1]):
                break

            target_label = target_labels[z, new_y_int, new_x_int]
            if target_label != 0:
                all_intersected_labels.add(target_label)
                break

    return list(all_intersected_labels)

def determine_fiber_orientation(component_mask, bounds, center_xy):
    # Extract voxel coordinates from the component mask
    voxel_coords = np.array(np.where(component_mask)).T
    
    # Adjust coordinates based on bounds
    voxel_coords[:, 0] += bounds[0].start
    voxel_coords[:, 1] += bounds[1].start
    voxel_coords[:, 2] += bounds[2].start
    
    # Project voxel coordinates onto the x-y plane
    xy_coords = voxel_coords[:, [2, 1]]  # X, Y
    
    # Subtract the center to make PCA relative to the center
    centered_xy = xy_coords - center_xy
    
    # Check the number of samples
    n_samples, n_features = centered_xy.shape
    if n_samples < 2:
        print("Orientation: unknown (insufficient samples)")  # Added debug print
        return 'unknown'
    
    try:
        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(centered_xy)
        
        # The principal component is the first eigenvector
        principal_component = pca.components_[0]
        
        # Determine the angle of the principal component
        angle = np.arctan2(principal_component[1], principal_component[0])  # In radians
        
        # Convert angle to degrees
        angle_deg = np.degrees(angle) % 180  # Angle between 0 and 180 degrees
        
        # Define thresholds to classify orientation
        if (0 <= angle_deg < 45) or (135 <= angle_deg < 180):
            orientation = 'x'
        elif (45 <= angle_deg < 135):
            orientation = 'y'
        else:
            orientation = 'unknown'
        
        print(f"Detected orientation: {orientation} (angle: {angle_deg:.2f} degrees)")  # Added debug print
        return orientation
    except Exception as e:
        print(f"PCA failed for component with error: {e}")
        print("Orientation: unknown (PCA failed)")  # Added debug print
        return 'unknown'


found_vt_labels = set()
found_hz_labels = set()

# Container for widgets on the right (vertical layout)
right_container = Container(widgets=[extract_label_from_image, set_centerxy,  remove_small_components, run_cc3d, find_fibers], layout="vertical", labels=False)
bottom_container = Container(widgets=[create_nodes_from_label, create_spline_from_nodes], layout="vertical", labels=False)

viewer.window.add_dock_widget(right_container, area="right",tabify=True)
viewer.window.add_dock_widget(bottom_container, area="bottom",tabify=True)

#viewer.window.add_dock_widget(set_pred_layers)
#viewer.window.add_dock_widget(run_cc3d)
#viewer.window.add_dock_widget(remove_small_components)


napari.run()