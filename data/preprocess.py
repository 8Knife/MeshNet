# import pymeshlab
import trimesh
import numpy as np
from pathlib import Path
from rich.progress import track



if __name__ == '__main__':
    root = Path('/home/gt/tmp/MeshNet2/MeshNet/dataset/ModelNet40')
    new_root = Path('/home/gt/tmp/MeshNet2/MeshNet/dataset/ModelNet40_processed')
    max_faces = 1024
    shape_list = sorted(list(root.glob('*/*/*.obj')))
    print("num obj:", len(shape_list))

    for shape_dir in track(shape_list):
        out_dir = new_root / shape_dir.relative_to(root).with_suffix('.npz')
        # if out_dir.exists():
        #     continue
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        # load mesh
        mesh = trimesh.load_mesh(str(shape_dir), process=False)

        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()

        if faces.shape[0] != max_faces:
            print("Model with more than {} faces ({}): {}".format(max_faces, faces.shape[0], out_dir))
            continue

        # move to center
        center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
        vertices -= center

        # normalize
        max_len = np.max(vertices[:, 0] ** 2 + vertices[:, 1] ** 2 + vertices[:, 2] ** 2)
        vertices /= np.sqrt(max_len)

        # recompute mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # face normal
        face_normal = mesh.face_normals

        # get neighbors
# region
        # faces_contain_this_vertex = []
        # for i in range(len(vertices)):
        #     faces_contain_this_vertex.append(set([]))
        # centers = []
        # corners = []
        # for i in range(len(faces)):
        #     [v1, v2, v3] = faces[i]
        #     x1, y1, z1 = vertices[v1]
        #     x2, y2, z2 = vertices[v2]
        #     x3, y3, z3 = vertices[v3]
        #     centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
        #     corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
        #     faces_contain_this_vertex[v1].add(i)
        #     faces_contain_this_vertex[v2].add(i)
        #     faces_contain_this_vertex[v3].add(i)
        #
        # neighbors = []
        # for i in range(len(faces)):
        #     [v1, v2, v3] = faces[i]
        #     n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
        #     n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
        #     n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
        #     neighbors.append([n1, n2, n3])
        #
        # centers = np.array(centers)
        # corners = np.array(corners)
        # faces = np.concatenate([centers, corners, face_normal], axis=1)
        # neighbors = np.array(neighbors)
# endregion
        # ---------------------------------------
        # 1. 计算每个三角面的 center 和 corner
        # ---------------------------------------

        centers = []  # 每个face中心点 (x,y,z)
        corners = []  # 每个face三个顶点坐标 (x1,y1,z1,x2,y2,z2,x3,y3,z3)

        for i in range(len(faces)):
            # faces[i] = [v1,v2,v3] 三角面三个顶点index
            v1, v2, v3 = faces[i]

            # 取顶点坐标
            x1, y1, z1 = vertices[v1]
            x2, y2, z2 = vertices[v2]
            x3, y3, z3 = vertices[v3]

            # -----------------------------
            # face center
            # -----------------------------
            centers.append([
                (x1 + x2 + x3) / 3,
                (y1 + y2 + y3) / 3,
                (z1 + z2 + z3) / 3
            ])

            # -----------------------------
            # face corners
            # -----------------------------
            corners.append([
                x1, y1, z1,
                x2, y2, z2,
                x3, y3, z3
            ])

        # 转成 numpy
        centers = np.array(centers)  # (F,3)
        corners = np.array(corners)  # (F,9)

        # ---------------------------------------
        # 2. 计算 face neighbors（替代 find_neighbor）
        # ---------------------------------------

        # 初始化邻居列表
        neighbors = [[] for _ in range(len(faces))]

        # trimesh已经计算好共享边的face
        for f1, f2 in mesh.face_adjacency:
            neighbors[f1].append(f2)
            neighbors[f2].append(f1)

        # MeshNet要求每个face必须有3个neighbor
        neighbors_fixed = []

        for i, neigh in enumerate(neighbors):

            # 如果邻居少于3个，用自己补齐
            if len(neigh) < 3:
                neigh = neigh + [i] * (3 - len(neigh))

            # 如果多于3个，截断
            else:
                neigh = neigh[:3]

            neighbors_fixed.append(neigh)

        neighbors = np.array(neighbors_fixed)  # (F,3)

        # ---------------------------------------
        # 3. 构造MeshNet输入特征
        # ---------------------------------------

        faces = np.concatenate([
            centers,  # (F,3)
            corners,  # (F,9)
            face_normal  # (F,3)
        ], axis=1)

        np.savez(str(out_dir), faces=faces, neighbors=neighbors)
