from typing import Any, Tuple, List
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import load_model
from phase_I.run_attention import find_tfl_lights
from phase_III.SFM import calc_TFL_dist


class FrameContainer(object):

    def __init__(self, img_path):
        self.img = plt.imread(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []


class TFLManager:

    def __init__(self, pkl_path, prev_frame_id, curr_frame_id):

        with open(pkl_path, 'rb') as pkl_file:

            data = pickle.load(pkl_file, encoding='latin1')
            self.focal = data['flx']
            self.pp = data['principle_point']
            self.ego_motions = {k: v for k, v in data.items() if 'egomotion' == k[:9]}
            self.prev_frame = None
            self.net = load_model("model.h5")


    def run_on_frame(self, image_path, index):

        print(index)
        current_frame = FrameContainer(image_path)

        '''part1'''
        candidate, auxiliary = self.detect_candidates(image_path)

        '''part2'''
        traffic, auxiliary_traffic = self.get_tfl_lights(candidate, auxiliary, image_path)
        current_frame.traffic_light = traffic

        '''part3'''
        dists = []
        if self.prev_frame:
            current_frame.EM = self.ego_motions['egomotion_' + str(index - 1) + '-' + str(index)]
            dists = self.find_dist_of_tfl(current_frame)

        self.visual_images(image_path, current_frame, auxiliary_traffic, candidate, auxiliary, dists)
        self.prev_frame = current_frame


    def detect_candidates(self, image_path: Any):

        x_red, y_red, x_green, y_green = find_tfl_lights(np.array(Image.open(image_path)), some_threshold=42)
        x_coord = x_red + x_green
        y_coord = y_red + y_green
        candidate = np.array([[x_coord[i], y_coord[i]] for i in range(len(x_coord))])
        auxiliary = ['r' if i < len(x_red) else 'g' for i in range(len(x_coord))]

        return candidate, auxiliary


    def get_tfl_lights(self, candidate, auxiliary, image_path) -> Tuple[list, list]:

        traffic = []
        auxiliary_traffic = []
        image = np.array(Image.open(image_path))
        image_pad = np.pad(image, 40, self.pad_with_zeros)[:, :, 40:43]

        for i, coord in enumerate(candidate):
            crop_image = self.get_crop_image(image_pad, coord)
            result = self.network(crop_image)

            if result:
                traffic.append(coord)
                auxiliary_traffic.append(auxiliary[i])

        return traffic, auxiliary_traffic


    def find_dist_of_tfl(self, current_frame):

        curr_container = calc_TFL_dist(self.prev_frame, current_frame, self.focal, self.pp)

        return np.array(curr_container.traffic_lights_3d_location)[:, 2]


    def get_cord_tfl(self, current_frame, image_path) -> Tuple[list, list, Any]:

        image = np.array(Image.open(image_path))
        curr_p = current_frame.traffic_light
        x_cord = [p[0] for p in curr_p]
        y_cord = [p[1] for p in curr_p]

        return x_cord, y_cord, image


    def get_crop_image(self, image, coord):

        crop_image = image[coord[1]:coord[1] + 81, coord[0]:coord[0] + 81]

        return crop_image


    def network(self, image) -> bool:

        crop_shape = (81, 81)
        predictions = self.net.predict(image.reshape([-1] + list(crop_shape) + [3]))

        return predictions[0][1] > 0.97


    def pad_with_zeros(self, vector, pad_width, iaxis, kwargs):

        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0


    def visual_images(self, image_path, current_frame, auxiliary_traffic, candidate, auxiliary, dist):

        fig, (ax_source_lights, ax_is_tfl, ax_dist) = plt.subplots(3, 1, figsize=(12, 30))
        ax_source_lights.imshow(current_frame.img)
        x_, y_ = candidate[:, 1], candidate[:, 0]
        ax_source_lights.scatter(y_, x_, c=auxiliary, s=1)
        ax_source_lights.set_title('source_light')

        ax_is_tfl.imshow(current_frame.img)
        x_, y_ = np.array(current_frame.traffic_light)[:, 1], np.array(current_frame.traffic_light)[:, 0]
        ax_is_tfl.scatter(y_, x_, c=auxiliary_traffic, s=1)
        ax_is_tfl.set_title('traffic_light')

        ax_dist.set_title('dist of tfl')

        if dist != list():
            x_cord, y_cord, image_dist = self.get_cord_tfl(current_frame, image_path)
            ax_dist.imshow(image_dist)

            for i in range(len(x_cord)):
                ax_dist.text(x_cord[i], y_cord[i], r'{0:.1f}'.format(dist[i]), color='r')

        fig.show()

