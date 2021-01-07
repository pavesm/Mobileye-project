
import matplotlib.pyplot as plt
from phase_IV.TFL_manager import TFLManager

class Controller:

    def __init__(self):

        with open("data/frames.pls", "r+") as data:

            read_lines = data.readlines()
            images_path = [line[:-1] for line in read_lines]
            self.path_images = images_path[2:]
            prev_frame_id = int(images_path[1])
            curr_frame_id = int(images_path[1]) + len(self.path_images) - 1
            self.tfl_manager = TFLManager(images_path[0], prev_frame_id, curr_frame_id)


    def run(self) -> None:

        for i in range(len(self.path_images)):
            self.tfl_manager.run_on_frame(self.path_images[i], i)

        plt.show(block=True)


def main():

    controller = Controller()
    controller.run()


if __name__ == '__main__':
    main()
