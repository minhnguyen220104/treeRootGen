import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from matplotlib.colors import ListedColormap
from scipy.ndimage import rotate
import os
from scipy.ndimage import convolve

def create_geometry(square_size, box_size, air_thickness):
    # Initialize the square room with air (represented by 0)
    geometry = np.zeros((square_size, square_size), dtype=int)

    # Set the soil box region
    box_start = air_thickness

    # Soil region is represented by 1
    geometry[box_start:box_size, square_size-box_size:box_size] = 1  

    return geometry, box_start, box_size
def get_material():
    '''materials = {
        "Concrete": {"type": "Dielectric", "permittivity": 5.24, "conductivity": 0.001},
        "Brick": {"type": "Dielectric", "permittivity": 3.91, "conductivity": 0.002},
        "Plasterboard": {"type": "Dielectric", "permittivity": 2.73, "conductivity": 0.0005},
        "Wood": {"type": "Dielectric", "permittivity": 1.99, "conductivity": 0.0002},
        "Glass": {"type": "Dielectric", "permittivity": 6.31, "conductivity": 0.00001},
        "Aluminum": {"type": "Metallic", "permittivity": 1, "conductivity": 3.77e7},
        "Copper": {"type": "Metallic", "permittivity": 1, "conductivity": 5.8e7},
        "Gold": {"type": "Metallic", "permittivity": 1, "conductivity": 4.1e7},
        "Silver": {"type": "Metallic", "permittivity": 1, "conductivity": 6.3e7},
        "Iron": {"type": "Metallic", "permittivity": 1, "conductivity": 1e7},
        "Dry Soil": {"type": "Nonmetallic", "permittivity": 4.0, "conductivity": 0.001},
        "Ice": {"type": "Nonmetallic", "permittivity": 3.2, "conductivity": 0.00001},
    }'''
    materials = {
        "Root": {"type": "Dielectric", "permittivity": 24, "conductivity": 0.00063}
    }
    
    variance_factor = 0.15
    material = random.choice(list(materials.keys()))

    if materials[material]["type"] == "Metallic":
        permittivity = materials[material]["permittivity"]
    else:
        permittivity = materials[material]["permittivity"] * random.uniform(1 - variance_factor, 1 + variance_factor)
    conductivity = materials[material]["conductivity"] * random.uniform(1 - variance_factor, 1 + variance_factor)
    return material, materials[material]["type"], round(permittivity, 3), round(conductivity, 6)

# def add_shape(i, geometry, box_start, box_end, air_thickness, shape = "circle"):
#     obj_mat,obj_type,permittivity_object, conductivity_object = get_material()
#     objair_gap = 25  # Gap between object and wall
#     # shape = random.choice(["rectangle", "triangle", "circle"])
#     rect_width = random.randint(20, 40)
#     rect_height = random.randint(20, 40)
#     # rect_y = random.randint(air_start + wall_thickness + objwall_gap, air_end - rect_height - square_size//4)
#     # # rect_x = random.randint(air_start + int(6*square_size/22), air_end - rect_width - int(6*square_size/22))
#     # rect_x = 200

#     if i == 0:
#         rect_y = random.randint(box_start+(box_end-air_thickness)//2, box_end - rect_height)
#         rect_x = random.randint(rect_width, square_size - rect_width - int(6*square_size/22))
#     elif i == 1:
#         rect_y = random.randint(box_start + objair_gap, box_start+(box_end-air_thickness)//2)
#         rect_x = random.randint(square_size//2, square_size - rect_width)
#     elif i == 2:
#         rect_y = random.randint(box_start + objair_gap, box_start+(box_end-air_thickness)//2)
#         rect_x = random.randint(rect_width, square_size//2 - rect_width)

#     # rotation_angle = random.randint(0, 360)
#     rotation_angle = 0

#     # Define a blank canvas for the shape
#     shape_canvas = np.zeros_like(geometry)

#     if shape == "rectangle":
#         shape_canvas[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width] = int(i) + 2

#     elif shape == "triangle":
#         for y in range(rect_height):
#             for x in range(rect_width - y):
#                 shape_canvas[rect_y + y, rect_x + x] = int(i) + 2

#     elif shape == "circle":
#         radius = min(rect_width, rect_height) // 2
#         x_center, y_center = rect_x + radius, rect_y + radius
#         for y in range(-radius, radius):
#             for x in range(-radius, radius):
#                 if x**2 + y**2 <= radius**2:
#                     # shape_canvas[y_center + y, x_center + x] = int(i) + 2
#                     shape_canvas[y_center + y, x_center + x] = 2

#     # Rotate the shape
#     rotated_shape = rotate(shape_canvas, angle=rotation_angle, reshape=False, order=0)


#     # Define a 5x5 kernel for checking neighborhood
#     kernel = np.ones((10, 10), dtype=int)

#     # Create a mask where nonzero values exist in a 5x5 neighborhood
#     nonzero_mask = convolve((geometry > 0).astype(int), kernel, mode='constant', cval=0) > 0

#     # Valid positions are where geometry is 0 and no nonzero values exist in a 5x5 region
#     valid_positions = (geometry == 0) & (~nonzero_mask)

#     # Add the shape only in valid positions
#     geometry[valid_positions & (rotated_shape > 1)] = rotated_shape[valid_positions & (rotated_shape > 1)]

#     return obj_mat,obj_type,permittivity_object, conductivity_object, shape, geometry

import numpy as np
import random
from scipy.ndimage import binary_dilation

def add_shape(i, geometry, square_size, box_start, box_end, air_thickness, shape="circle",objair_gap=25, min_spacing=10, max_tries=400):
    """
    Place one shape into the soil box with:
      - no overlap,
      - at least `objair_gap` from air,
      - at least `min_spacing` pixels from any *existing* shape.
    Labels:
      Air=0, Soil=1, Shapes>=2 (label = 2+i)
    """

    # Soil box columns from how create_geometry() filled it:
    # geometry[box_start:box_end, square_size-box_end:box_end] = 1
    col_min_raw = square_size - box_end
    col_max_raw = box_end

    # Apply clearance from air boundaries (shrink the allowed box)
    row_min = box_start + objair_gap
    row_max = box_end   - objair_gap
    col_min = col_min_raw + objair_gap
    col_max = col_max_raw - objair_gap

    if row_min >= row_max or col_min >= col_max:
        raise ValueError("Not enough room inside the box after applying objair_gap.")

    # Sizes
    rect_width  = random.randint(20, 30)
    rect_height = random.randint(20, 30)
    label = int(i) + 2

    # --- Build a circular structuring element for spacing ---
    def disk(radius):
        r = int(radius)
        y, x = np.ogrid[-r:r+1, -r:r+1]
        return (x*x + y*y) <= (r*r)

    selem = disk(max(1, min_spacing))

    # Anything not soil (air=0 or shapes>=2) is "blocked"
    blocked = (geometry != 1)
    # Dilate blocked region to keep a spacing buffer from existing shapes/air
    blocked_dilated = binary_dilation(blocked, structure=selem)

    # Candidate mask generators
    def sample_circle_mask():
        r = min(rect_width, rect_height) // 2
        cy = random.randint(row_min + r, row_max - 1 - r)
        cx = random.randint(col_min + r, col_max - 1 - r)
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        circle = (yy*yy + xx*xx) <= (r*r)
        oy, ox = np.where(circle)
        rr = oy + (cy - r)
        cc = ox + (cx - r)
        return rr, cc

    def sample_rectangle_mask():
        top  = random.randint(row_min, row_max - rect_height)
        left = random.randint(col_min, col_max - rect_width)
        rr, cc = np.mgrid[top:top+rect_height, left:left+rect_width]
        rr = rr.ravel()
        cc = cc.ravel()
        return rr, cc

    # Feasibility check: must be on soil (==1), not in dilated blocked area
    def can_place(rr, cc):
        return np.all(geometry[rr, cc] == 1) and np.all(~blocked_dilated[rr, cc])

    # Try to place
    for _ in range(max_tries):
        if shape == "circle":
            rr, cc = sample_circle_mask()
        elif shape == "rectangle":
            rr, cc = sample_rectangle_mask()
        else:  # default to circle if unknown
            rr, cc = sample_circle_mask()

        if can_place(rr, cc):
            geometry[rr, cc] = label
            obj_mat, obj_type, eps_obj, sig_obj = get_material()
            return obj_mat, obj_type, eps_obj, sig_obj, shape, geometry

    raise RuntimeError(
        "Could not place the shape with the requested spacing; try reducing min_spacing/objair_gap or sizes."
    )


def visualize_geometry(geometry, box_color, air_color, r1_color, r2_color, r3_color):
    cmap = ListedColormap([
        air_color, 
        box_color,
        r1_color,
        r2_color, 
        r3_color
    ])
    plt.figure(figsize=(10, 10))
    plt.imshow(geometry, cmap=cmap, origin='lower')
    plt.title('Heterogenous Geometry with Circle Tree Roots')
    plt.axis('off')
    plt.show()

def save_image(filename, numobjects, geometry, square_size, box_color, air_color, r1_color, r2_color, r3_color):
    cmap = ListedColormap([
        air_color, 
        box_color, 
        r1_color,
        r2_color,
        r3_color
    ][:numobjects + 2])
    plt.figure(figsize=(10, 10))
    plt.imshow(geometry, cmap=cmap)
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.axis('off')    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, format='png', dpi=square_size / 10, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_base(filename, geometry, square_size, box_color, air_color):
    cmap = ListedColormap([air_color,box_color])    
    plt.figure(figsize=(10, 10))
    plt.imshow(geometry, cmap=cmap)
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.axis('off')    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, format='png', dpi=square_size / 10, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_parameters(filename, **params):
    if os.path.exists(filename):
        existing_data = np.load(filename, allow_pickle=True)
        all_params = list(existing_data['params'])
    else:
        all_params = []

    all_params.append(params)

    with open(filename, 'wb') as f:
        np.savez(f, params=all_params)

if __name__ == '__main__':
    # Predefined colors
    box_color = [1, 1, 0]   # Box color
    air_color = [1, 1, 1]    # Air color
    r1_color = [0, 0, 1]      # Root color
    r2_color = [0, 0, 1]      # Root color
    r3_color = [0, 0, 1]      # Root color

    # Argument parsing
    parser = argparse.ArgumentParser(description='Generate and visualize heterogenous geometries with tree roots.')
    parser.add_argument('--start', type=int, default=0, help='Starting index for geometry generation')
    parser.add_argument('--end', type=int, default=10, help='Ending index for geometry generation')
    args = parser.parse_args()

    args.n = args.end + 1 - args.start

    for i in range(args.n):
        square_size = 200
        # wall_thickness = random.randint(15, 30)
        air_thickness = 100

        # Define wall materials with permittivity and conductivity
        box_materials = {
            "Concrete": {"permittivity": 5.24, "conductivity": 0.001},
            "Brick": {"permittivity": 3.91, "conductivity": 0.002},
        }

        # Variance factor for permittivity
        variance_factor = 0.1

        # Randomly select a wall material
        box_material = random.choice(list(box_materials.keys()))

        # Get the base permittivity and conductivity
        base_permittivity = box_materials[box_material]["permittivity"]
        conductivity = box_materials[box_material]["conductivity"]

        # Add variability to permittivity
        variance = base_permittivity * variance_factor
        permittivity = round(random.uniform(base_permittivity - variance, base_permittivity + variance), 2)

        # # Print the results
        # print(f"Wall Material: {wall_material}")
        # print(f"Permittivity: {permittivity_wall}")
        # print(f"Conductivity: {conductivity}")

        if not os.path.exists('./Geometry_ge'):
            os.makedirs('./Geometry_ge')
        if not os.path.exists('./Geometry_ge/Roots'):
            os.makedirs('./Geometry_ge/Roots')
        if not os.path.exists('./Geometry_ge/HeteSoil'):
            os.makedirs('./Geometry_ge/HeteSoil')

        filename = f'./Geometry_ge/Roots/roots{i + args.start}.png'
        base = f'./Geometry_ge/HeteSoil/hetesoil{i + args.start}.png'
        params_filename = f'./Geometry_ge/root_hete_{args.start}_{args.end}.npz'

        geometry, box_start, box_end = create_geometry(square_size, square_size, air_thickness)

        save_base(base, geometry, square_size, box_color, air_color)

        per_obj_arr = []
        shape_arr = []
        con_arr = []
        mat_arr = []
        num_objects = random.randint(1, 3)
        for j in range(num_objects):
            obj_mat,obj_type,per_obj, con_obj, shape, geometry = add_shape(j, geometry, square_size, box_start, box_end, air_thickness)
            per_obj_arr.append(per_obj)
            shape_arr.append(shape)
            con_arr.append(con_obj)
            mat_arr.append(obj_type)

        save_image(filename,num_objects, geometry, square_size, box_color, air_color, r1_color, r2_color, r3_color)

        save_parameters(
            params_filename,
            shape=shape_arr,
            square_size=square_size,
            air_thickness=air_thickness,
            wall_color=box_color,
            air_color=air_color,
            object_color=[r1_color, r2_color, r3_color],
            conductivity_object = con_arr,
            permittivity_object=per_obj_arr,   
            material = mat_arr, 
            permittivity_box=permittivity,
            conductivity_box=conductivity,
            box_material=box_material,
        )
