import os
import pickle
import json
import time
import tqdm
import trimesh
import trimesh.visual
import trimesh.creation
import numpy as np
import pyrender

import utils


class GenerateSwapFeature:
    def __init__(self, model_dir='UHM_models',
                 model_name='head_model_global_align_no_mouth_and_eyes.pkl',
                 template_name='mean_nme_fcolor_b.ply'):
        infile = open(os.path.join(model_dir, model_name), 'rb')
        uhm_dict = pickle.load(infile)
        infile.close()

        self.model_dir = model_dir
        self.template_name = template_name
        self._components = uhm_dict['Eigenvectors'].shape[1]
        self._faces = uhm_dict['Trilist']
        self._mu = uhm_dict['Mean']
        self._eigenvectors = uhm_dict['Eigenvectors']
        self._eigenvalues = uhm_dict['EigenValues']

        colored = trimesh.load(os.path.join(self.model_dir, self.template_name),
                               process=False)
        features = utils.extract_feature_and_contour_from_colour(colored)
        key = list(features.keys())[11]
        self._feature_idx = features[key]['feature']
        self._contour_idx = features[key]['contour']

    @property
    def faces(self):
        return self._faces

    def save_feature_indices(self, filename):
        feature_idx_list = self._feature_idx[0].tolist()
        with open(filename, 'w') as outfile:
            json.dump(feature_idx_list, outfile)

    def generate_random_face(self, weight=1.):
        w = weight * np.random.normal(size=self._components) * \
            self._eigenvalues ** 0.5
        w = np.expand_dims(w, axis=1)

        verts = self._mu + self._eigenvectors @ w
        return verts.reshape(-1, 3)

    def generate_and_swap(self, visualize=False):
        verts0 = self.generate_random_face(weight=1)
        feature_verts0 = verts0[self._feature_idx]
        contour_verts0 = verts0[self._contour_idx]

        verts1 = self.generate_random_face(weight=1.5)
        feature_verts1 = verts1[self._feature_idx]
        contour_verts1 = verts1[self._contour_idx]

        t = time.time()
        # compute lest square transformation to fit feature on face
        hom_contour_verts0 = self.to_homogeneous(contour_verts0)
        hom_contour_verts1 = self.to_homogeneous(contour_verts1)
        transform, _, _, _ = np.linalg.lstsq(hom_contour_verts1,
                                             hom_contour_verts0,
                                             rcond=None)

        # apply transformation
        hom_feature_verts1 = self.to_homogeneous(feature_verts1)
        hom_feature_verts1 = hom_feature_verts1 @ transform
        feature_verts1 = self.to_cartesian(hom_feature_verts1)

        # deform feature0 on feature1
        distances = self.compute_minimum_distances(feature_verts1,
                                                   contour_verts0)
        max_dist = max(distances)
        displacement_weights = np.tanh(3 * (distances / max_dist))
        # displacement_weights = self.smooth_step(distances / max_dist, 0.1)
        displacement_weights = np.expand_dims(displacement_weights, axis=1)
        displacement = feature_verts1 - feature_verts0
        feature_verts_def = feature_verts0 + displacement * displacement_weights

        # assemble final face with new feature
        combined_verts = verts0.copy()
        combined_verts[self._feature_idx] = feature_verts_def

        print(time.time() - t)

        if visualize:
            scene = pyrender.Scene()

            original = pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(verts0 - [4, 0, 0], self._faces, process=False,
                                vertex_colors=[255, 219, 172, 180]))
            target_feature = self.create_pyrender_pcl(
                feature_verts1 - [4, 0, 0])
            scene.add(original)
            scene.add(target_feature)

            final = pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(combined_verts - [2, 0, 0], self._faces,
                                process=False,
                                vertex_colors=[255, 219, 172, 255]))
            scene.add(final)

            colours = np.tile(np.array([255, 219, 172, 255]),
                              [verts0.shape[0], 1])
            feature_colours = trimesh.visual.interpolate(displacement_weights,
                                                         'plasma')
            colours[self._feature_idx] = feature_colours
            final_cmap = pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(combined_verts, self._faces,
                                process=False, vertex_colors=colours))
            scene.add(final_cmap)

            final_transparent = pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(combined_verts + [2, 0, 0], self._faces,
                                process=False,
                                vertex_colors=[255, 219, 172, 180]))
            original_feature = self.create_pyrender_pcl(
                feature_verts0 + [2, 0, 0], [0., 1., 0.])
            scene.add(final_transparent)
            scene.add(original_feature)

            pyrender.Viewer(scene, use_raymond_lighting=True)

        return verts0, combined_verts

    @staticmethod
    def to_homogeneous(x):
        return np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)

    @staticmethod
    def to_cartesian(x):
        x = x[:, :3] / x[:, 3:4]
        return x[:, :3]

    @staticmethod
    def compute_minimum_distances(feature, contour):
        distances = []
        for f in feature:
            distances.append(min(np.sum((contour - f) ** 2, axis=1)))
        return np.array(distances)

    @staticmethod
    def smooth_step(x, end_step=1.):
        return np.where(x > end_step, 1,
                        3 * (x / end_step) ** 2 - 2 * (x / end_step) ** 3)

    @staticmethod
    def create_pyrender_pcl(points, colour=None):
        sm = trimesh.creation.uv_sphere(radius=0.003, count=[3, 3])
        if colour is None:
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
        else:
            sm.visual.vertex_colors = colour
        tfs = np.tile(np.eye(4), (len(points), 1, 1))
        tfs[:, :3, 3] = points
        return pyrender.Mesh.from_trimesh(sm, poses=tfs)


if __name__ == '__main__':
    swap_generator = GenerateSwapFeature()
    # swap_generator.save_feature_indices(os.path.join('..', 'static', 'data',
    #                                                  'feature_idx.json'))
    for i in tqdm.tqdm(range(10)):
        v0, v1 = swap_generator.generate_and_swap(visualize=True)
        fs = swap_generator.faces
        m1 = trimesh.Trimesh(v0, fs, process=False)
        m2 = trimesh.Trimesh(v1, fs, process=False)
