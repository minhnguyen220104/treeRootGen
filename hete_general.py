import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from matplotlib.colors import ListedColormap
from scipy.ndimage import rotate
import os
from scipy.ndimage import convolve
from scipy.ndimage import binary_dilation

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

# def add_shape(i, geometry, square_size, box_start, box_end, air_thickness,
#               shape="circle", objair_gap=25, min_spacing=20, max_tries=400,
#               fixed_row=True):
#     """
#     Place one shape into the soil box with:
#       - no overlap,
#       - at least `objair_gap` from air,
#       - at least `min_spacing` pixels from any *existing* shape,
#       - Shapes aligned at the same row if fixed_row=True.

#     Labels:
#       Air=0, Soil=1, Shapes>=2 (label = 2+i)

#     Args:
#       fixed_row (bool): 
#         True  -> all shapes share the same row (randomly chosen once per run).
#         False -> each shape gets its own random row.
#     """

#     # Soil box columns from how create_geometry() filled it:
#     col_min_raw = square_size - box_end
#     col_max_raw = box_end

#     # Apply clearance from air boundaries
#     row_min = box_start + 2*objair_gap
#     row_max = box_end   - objair_gap
#     col_min = col_min_raw + objair_gap
#     col_max = col_max_raw - objair_gap

#     if row_min >= row_max or col_min >= col_max:
#         raise ValueError("Not enough room inside the box after applying objair_gap.")

#     # Sizes
#     rect_width  = random.randint(20, 25)
#     rect_height = random.randint(20, 25)
#     label = int(i) + 2

#     # --- Adaptive spacing ---
#     obj_diag = int(np.hypot(rect_width, rect_height))
#     min_spacing = max(min_spacing, obj_diag // 2)

#     # --- Choose row ---
#     if fixed_row:
#         if not hasattr(add_shape, "_shared_row") or add_shape._shared_row is None:
#             add_shape._shared_row = random.randint(row_min, row_max - 1)
#         shared_row = add_shape._shared_row
#     else:
#         shared_row = random.randint(row_min, row_max - 1)

#     # --- Build structuring element for spacing ---
#     def disk(radius):
#         r = int(radius)
#         y, x = np.ogrid[-r:r+1, -r:r+1]
#         return (x*x + y*y) <= (r*r)

#     selem = disk(max(1, min_spacing))

#     blocked = (geometry != 1)
#     blocked_dilated = binary_dilation(blocked, structure=selem)

#     # Helpers
#     def clamp(v, lo, hi):
#         return max(lo, min(hi, v))

#     def sample_circle_mask():
#         r = min(rect_width, rect_height) // 2
#         cy = clamp(shared_row, row_min + r, row_max - 1 - r)
#         cx = random.randint(col_min + r, col_max - 1 - r)
#         yy, xx = np.ogrid[-r:r+1, -r:r+1]
#         circle = (yy*yy + xx*xx) <= (r*r)
#         oy, ox = np.where(circle)
#         rr = oy + (cy - r)
#         cc = ox + (cx - r)
#         return rr, cc

#     def sample_rectangle_mask():
#         center = clamp(shared_row, row_min, row_max - 1)
#         top = clamp(center - rect_height // 2, row_min, row_max - rect_height)
#         left = random.randint(col_min, col_max - rect_width)
#         rr, cc = np.mgrid[top:top+rect_height, left:left+rect_width]
#         return rr.ravel(), cc.ravel()

#     def can_place(rr, cc):
#         H, W = geometry.shape
#         if (rr.min() < 0) or (cc.min() < 0) or (rr.max() >= H) or (cc.max() >= W):
#             return False
#         return np.all(geometry[rr, cc] == 1) and np.all(~blocked_dilated[rr, cc])

#     # Try to place
#     for _ in range(max_tries):
#         if shape == "circle":
#             rr, cc = sample_circle_mask()
#         elif shape == "rectangle":
#             rr, cc = sample_rectangle_mask()
#         else:
#             rr, cc = sample_circle_mask()

#         if can_place(rr, cc):
#             geometry[rr, cc] = label
#             obj_mat, obj_type, eps_obj, sig_obj = get_material()
#             return obj_mat, obj_type, eps_obj, sig_obj, shape, geometry

#     raise RuntimeError("Could not place the shape with the requested spacing.")



def add_shape(i, geometry, square_size, box_start, box_end, air_thickness,
              shape="circle", objair_gap=25, min_spacing=50, max_tries=400):
    """
    Place one shape into the soil box with:
      - no overlap,
      - at least `objair_gap` from air,
      - at least `min_spacing` pixels from any *existing* shape,
      - X-position constrained by `i`:
          i==0 -> left band   [col_min, col_min+50]
          i==1 -> middle band [col_min+50+min_spacing, col_max-50-min_spacing]
          i==2 -> right band  [col_max-50, col_max]
    Labels: Air=0, Soil=1, Shapes>=2 (label = 2+i)
    """

    # Soil box boundaries
    col_min_raw = square_size - box_end
    col_max_raw = box_end

    H, W = geometry.shape
    row_min = max(0, box_start + 2*objair_gap)
    row_max = min(H, box_end - objair_gap)
    col_min = max(0, col_min_raw + objair_gap)
    col_max = min(W, col_max_raw - objair_gap)

    if row_min >= row_max or col_min >= col_max:
        raise ValueError("Not enough room inside the box after applying objair_gap.")

    # Shape sizes
    rect_width  = random.randint(10, 15)
    rect_height = random.randint(10, 15)
    label = int(i) + 2

    # Adaptive spacing
    obj_diag = int(np.hypot(rect_width, rect_height))
    min_spacing = max(min_spacing, obj_diag // 2)


    # Disk kernel for spacing
    def disk(radius):
        r = int(max(1, radius))
        y, x = np.ogrid[-r:r+1, -r:r+1]
        return (x*x + y*y) <= (r*r)

    selem = disk(min_spacing)
    blocked = (geometry != 1)
    blocked_dilated = binary_dilation(blocked, structure=selem)

    # Sector logic
    def sector_bounds_for_i(i):
        if i == 0:
            sx_lo = col_min + 50 + min_spacing
            sx_hi = col_max - 50 - min_spacing
        elif i == 1:
            sx_lo = col_min
            sx_hi = min(col_min + 50, col_max - 1)
        else:
            sx_lo = max(col_min, col_max - 50)
            sx_hi = col_max - 1
        return sx_lo, sx_hi

    # Sampling functions
    def sample_circle_mask():
        r = min(rect_width, rect_height) // 2
        cy = random.randint(row_min + r, row_max - 1 - r - 50)
        sx_lo, sx_hi = sector_bounds_for_i(i)
        cx_lo = max(sx_lo + r, col_min + r)
        cx_hi = min(sx_hi - r, col_max - 1 - r)
        if cx_lo > cx_hi:
            raise RuntimeError("No horizontal room in the chosen sector for the circle.")
        cx = random.randint(cx_lo, cx_hi)
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        circle = (yy*yy + xx*xx) <= (r*r)
        oy, ox = np.where(circle)
        rr = oy + (cy - r)
        cc = ox + (cx - r)
        return rr, cc

    def sample_rectangle_mask():
        top = random.randint(row_min, row_max - rect_height)
        sx_lo, sx_hi = sector_bounds_for_i(i)
        left_lo = max(sx_lo, col_min)
        left_hi = min(sx_hi - (rect_width - 1), col_max - rect_width)
        if left_lo > left_hi:
            raise RuntimeError("No horizontal room in the chosen sector for the rectangle.")
        left = random.randint(left_lo, left_hi)
        rr, cc = np.mgrid[top:top+rect_height, left:left+rect_width]
        return rr.ravel(), cc.ravel()

    def can_place(rr, cc):
        if (rr.min() < 0) or (cc.min() < 0) or (rr.max() >= H) or (cc.max() >= W):
            return False
        return np.all(geometry[rr, cc] == 1) and np.all(~blocked_dilated[rr, cc])

    # Try placements
    for _ in range(max_tries):
        if shape == "circle":
            rr, cc = sample_circle_mask()
        elif shape == "rectangle":
            rr, cc = sample_rectangle_mask()
        else:
            rr, cc = sample_circle_mask()

        if can_place(rr, cc):
            geometry[rr, cc] = label
            obj_mat, obj_type, eps_obj, sig_obj = get_material()
            return obj_mat, obj_type, eps_obj, sig_obj, shape, geometry

    raise RuntimeError("Could not place the shape with the requested spacing and sector.")




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
        square_size = 300
        # wall_thickness = random.randint(15, 30)
        air_thickness = 150

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

        #add_shape._shared_row = None

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
