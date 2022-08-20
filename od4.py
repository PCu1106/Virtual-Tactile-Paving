import signal
import numpy as np
import threading
from libs.darknet.yolo_device import YoloDevice
from libs.darknet import utils
from libs.deep_sort.wrapper import DeepSortWrapper
from libs.deep_sort.wrapper import DEEP_SORT_ENCODER_MODEL_PATH_PERSON




# Note: this program can only run on Linux
if __name__ == '__main__':
    # Define the variables
    yolo = YoloDevice(
        video_url="rtsp://192.168.10.100/video0.sdp",
        gpu=False,
        gpu_id=0,
        display_message=True,
        config_file=utils.CONFIG_FILE_YOLO_V4,
        names_file=utils.NAMES_COCO,
        thresh=0.25,
        weights_file=utils.WEIGHTS_YOLO_V4_COCO,
        output_dir="./output",
        # use_polygon=False,
        # vertex=[(0, 0), (0, 216), (384, 261), (384, 0)],
        target_classes=["person"],
        draw_bbox=True,
        draw_polygon=True,
    )

    run = True


    def job():
        global object_id
        while (True):
            object_id = int(input("Please enter thr ID that you want to track"))
            print("thread object id:", object_id)


    t = threading.Thread(target=job)
    t.start()


    def on_data(frame_id, img, bboxes, img_path):
        """
        When objects are detected, this function will be called.

        Args:
            frame_id:
                the frame number
            img:
                a numpy array which stored the frame
            bboxes:
                A list contains several `libs.darknet.yolo_device.ExtendedBoundingBox` object. The list contains
                all detected objects in a single frame.
            img_path:
                The path of the stored frame. If `output_dir` is None, this parameter will be None too.
        """
        print("===========================================")
        # minid = bboxes.begin().get_class_id()
        # for det3 in bboxes:
        #     if det3.get_class_id() < minid:
        #         minid = det3.get_class_id()
        # print(minid)
        print("main object id", object_id)
        # print(object_id)

        for det in bboxes:

            # You can push these variables to IoTtalk sever
            class_name = det.get_class_name()
            confidence = det.get_confidence()
            center_x, center_y = det.get_center()
            width, height = det.get_width(), det.get_height()
            circus1 = max(det.get_width(), det.get_height())

            print(class_name, det.get_class_id(), det.get_obj_id())
            #print(class_name, confidence, center_x, center_y,width,height, det.get_class_id(), det.get_obj_id())
            for det2 in bboxes:
                if det.get_obj_id() == object_id and det2.get_obj_id() != object_id:   
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    circus2 = max(det2.get_width(), det2.get_height())
                    center_x2, center_y2 = det.get_center()
                    a = np.array((center_x, center_y))
                    b = np.array((center_x2, center_y2))
                    dist = np.linalg.norm(a - b)
                    if dist <= circus1 or dist <= circus2:
                        print("Obstacle Dectect!!!!!")


        print("=========================================")


    # `yolo.set_listener()` must be before `yolo.start()`
    yolo.set_listener(on_data)
    yolo.enable_tracking(True)
    yolo.add_deep_sort_tracker("person", DeepSortWrapper(encoder_model_path=DEEP_SORT_ENCODER_MODEL_PATH_PERSON))
    yolo.start()


    def signal_handler(sig, frame):
        """
        A simple handler to handle ctrl + c
        """
        global run
        run = False
        # Use `yolo.stop()` to stop the darknet program
        yolo.stop()
        # Use `yolo.join()` to wait for darknet closing
        yolo.join()


    print("press ctrl + c to exit.")
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()

    while run:
        pass

    print("Exit.")
