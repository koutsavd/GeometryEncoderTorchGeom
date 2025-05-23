import open3d as o3d
import trimesh
from trimesh.sample import sample_surface
import networkx as nx
import numpy as np
from vedo import Mesh, show
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix
from pathlib import Path
from scipy import sparse
from tqdm import tqdm
from joblib import Parallel, delayed



# Load mesh using trimesh
def mesh_visualize(mesh):

# Convert to Open3D
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Optionally compute normals
    o3d_mesh.compute_vertex_normals()

    # Show
    o3d.visualization.draw_geometries([o3d_mesh])

def visualize_point_cloud(points: np.ndarray):

    points = points - np.mean(points, axis=0)
    scale = np.linalg.norm(points, axis=1).max()
    points = points / scale
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)


    # Optionally add some color (uniform gray)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # Launch Open3D viewer
    o3d.visualization.draw_geometries([pcd])


# def read_preproc_save(path, n_points, n_features):
#     k = 40
#
#     # Load and sample mesh
#     mesh = trimesh.load(path)
#     points = mesh.sample(n_points)
#     center_point = mesh.sample(n_points)
#     points = points - points.mean(axis=0)
#     scale = np.linalg.norm(points, axis=1).max()
#     points = points / scale
#     centroid = np.mean(points, axis=0)
#
#     # Build k-NN graph
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
#     distances, indices = nbrs.kneighbors(points)
#
#     G = nx.Graph()
#     for i, point in enumerate(points):
#         G.add_node(i, pos=point)
#     for i in range(n_points):
#         for neighbor_index in indices[i]:
#             if i != neighbor_index:
#                 G.add_edge(i, neighbor_index)
#
#     # Compute node features
#     curvatures = []
#     normals = []
#     distance_to_centroid = []
#     for i in range(n_points):
#         nbr_idx = indices[i]
#         nbr_pts = points[nbr_idx]
#         pca = PCA(n_components=3)
#         pca.fit(nbr_pts)
#         curvature = pca.explained_variance_[2] / sum(pca.explained_variance_)
#         curvatures.append(curvature)
#         normals.append(pca.components_[-1])
#         distance_to_centroid.append(np.linalg.norm(points[i] - centroid))
#
#     node_features = np.zeros((n_points, n_features), dtype=np.float32)
#     for i in range(n_points):
#         node_features[i, 0:3] = points[i]
#         node_features[i, 3] = distance_to_centroid[i]
#         node_features[i, 4] = curvatures[i]
#         node_features[i, 5:8] = normals[i]
#
#     # Build adjacency matrix
#     A = nx.to_scipy_sparse_array(G, format='coo')
#
#     # Compute edge features
#     edge_features = []
#     for i, j in zip(A.row, A.col):
#         vec = points[j] - points[i]
#         length = np.linalg.norm(vec)
#         dx, dy, dz = vec / (length + 1e-8)
#         ni = normals[i]
#         nj = normals[j]
#         cos_angle = np.clip(np.dot(ni, nj), -1.0, 1.0)
#         angle = np.arccos(cos_angle) / np.pi
#         edge_features.append([length, dx, dy, dz, angle])
#     edge_features = np.array(edge_features, dtype=np.float32)
#
#     # Save everything
#     export_path = Path("Graphs")
#     export_path.mkdir(parents=True, exist_ok=True)
#     filename = Path(path).stem
#     save_path = export_path / f"{filename}.npz"
#     adj_path = export_path / f"{filename}_adj.npz"
#
#     np.savez_compressed(save_path, x=node_features, e=edge_features, center_point=center_point, scale=scale)
#     sparse.save_npz(adj_path, A)

def read_preproc_save(path, n_features):
    mesh = trimesh.load(path)
    print(f"Loaded {len(mesh.vertices)} vertices from {path}. Mesh is watertight: {mesh.is_watertight}")
    points = mesh.vertices
    n_points = len(points)

    if n_points < 1000:
        print(f"Skipping {path.name} due to small size ({n_points} points)")
        return

    centroid = np.mean(points, axis=0)
    scale = np.linalg.norm(points - centroid, axis=1).max()
    points_scaled = (points - centroid) / scale

    # Automatically determine flow direction from bounding box
    extents = mesh.bounding_box.extents  # dx, dy, dz
    max_axis = np.argmax(extents)
    flow_vector = np.eye(3)[max_axis]  # [1,0,0], [0,1,0], or [0,0,1]

    G = nx.Graph()
    for i, pt in enumerate(points):
        G.add_node(i, pos=pt)
    for edge in mesh.edges:
        G.add_edge(edge[0], edge[1])

    curvatures, normals, dists, dots, topo = [], [], [], [], []
    for i in range(n_points):
        nbrs = list(G.neighbors(i))
        nbr_pts = points[nbrs]
        pt = points[i]
        if len(nbr_pts) < 3:
            curvature = 0.0
            normal = np.array([0, 0, 1])
            topo_label = 0
        else:
            pca = PCA(n_components=3).fit(nbr_pts)
            eigs = pca.explained_variance_
            curvature = eigs[2] / (np.sum(eigs) + 1e-8)
            normal = pca.components_[-1]
            if eigs[2] < eigs[1] and eigs[1] < eigs[0]:
                topo_label = 0  # flat
            elif eigs[1] > eigs[0] and eigs[1] > eigs[2]:
                topo_label = 1  # ridge
            else:
                topo_label = 2  # saddle

        curvatures.append(curvature)
        normals.append(normal)
        dists.append(np.linalg.norm(pt - centroid))
        dots.append(np.dot(normal, flow_vector))
        topo.append(topo_label)

    curvatures = (np.array(curvatures) - np.min(curvatures)) / (np.ptp(curvatures) + 1e-8)
    dists = (np.array(dists) - np.min(dists)) / (np.ptp(dists) + 1e-8)

    node_features = np.zeros((n_points, n_features), dtype=np.float32)
    for i in range(n_points):
        node_features[i, 0:3] = points_scaled[i]
        node_features[i, 3] = dists[i]
        node_features[i, 4] = curvatures[i]
        node_features[i, 5:8] = normals[i]
        node_features[i, 8] = dots[i]
        node_features[i, 9] = topo[i]

    A = nx.to_scipy_sparse_array(G, format='coo')

    export_path = Path("Graphs")
    export_path.mkdir(parents=True, exist_ok=True)
    base = Path(path).stem
    np.savez_compressed(export_path / f"{base}.npz", x=node_features, center_point=centroid, scale=scale)
    sparse.save_npz(export_path / f"{base}_adj.npz", A)


if __name__ == "__main__":
    p = Path("/Users/koutsavd/PycharmProjects/Geometry_GNN/car")
    # files = list(p.rglob("*.off"))
    # for file in tqdm(files, desc="Processing OFF files"):
    #     read_preproc_save(file, 4096, 8)
    files = list(p.rglob("*.off"))
    for f in files:
        read_preproc_save(f, 10)
        # Parallel(n_jobs=8)(delayed(read_preproc_save)(f, 10) for f in files)