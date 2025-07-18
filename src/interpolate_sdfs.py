import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt as distance
from skimage import measure
import trimesh
import os
from itertools import combinations_with_replacement
from helpers import load_config
import time

# load the configuration file
config = load_config()

corner_stls = config["corner_stls"]
resolution = config["resolution"]
project_dir = config["project_dir"]
samples_per_dim = config["samples_per_dim"]
epsilon = config["epsilon"]
project_dir = f'{project_dir}_r{resolution}_n{samples_per_dim}'
print(project_dir)


def gen_1d_bary_weights(resolution):
    w1_list = []
    w2_list = []
    step = 1 / resolution
    for w1 in range(resolution+1):
        w2 = resolution - w1
        w1_list.append(w1*step)
        w2_list.append(w2*step)

    w1_array = np.array(w1_list)
    w2_array = np.array(w2_list)

    return w1_array, w2_array


def interpolate_sdfs_1d(sdfs, w1, w2):
    return w1*sdfs[0]+w2*sdfs[1]


sdfs = [binary_fill_holes(np.load(f"{project_dir}/corner_{i}.npy")) for i in range(len(corner_stls))]
t0 = time.time()
sdfs = [distance(~sdf) - distance(sdf) for sdf in sdfs] # convert binary representation to SDFs here!

#np.save('sdfs.npy', sdfs)
#print(sdfs.shape)
#breakpoint()

t1 = time.time()
print(f'Timings :: SDF (distance transform): {t1-t0} (s)')

# generate barycentric coordinates
num_stls = len(config["corner_stls"])

w1, w2 = gen_1d_bary_weights(samples_per_dim)
np.save(f'{project_dir}/w1.npy', w1)
np.save(f'{project_dir}/w2.npy', w2)

npys_dir = f"{project_dir}/npys"
#sdfs_dir = f"{project_dir}/sdfs"
os.makedirs(f"{npys_dir}", exist_ok=True) 
#os.makedirs(f"{sdfs_dir}", exist_ok=True)

# process each combination of barycentric coordinates
count = 0
print(f'Total samples: {w1.shape}')
for i in range(w1.shape[0]):
    print(count)
    interpolated_sdf = interpolate_sdfs_1d(sdfs, w1[i], w2[i])

    vertices, faces, normals, values = measure.marching_cubes(interpolated_sdf, level=0.0, step_size=8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # export to a temporary STL file
    temp_stl_path = os.path.join(output_directory, 'recon.stl')
    mesh.export(temp_stl_path)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(temp_stl_path)

    ms.apply_filter('apply_coord_laplacian_smoothing', stepsmoothnum=smooth_iter)

    smooth_stl_path = os.path.join(output_directory, npy_file.replace('.npy', '.stl'))
    ms.save_current_mesh(smooth_stl_path)

    # delete the temporary STL file
    os.remove(temp_stl_path)
