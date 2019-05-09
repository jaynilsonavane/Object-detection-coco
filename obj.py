from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()

#COCO MODEL
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

#YOLO MODEL
#detector.setModelTypeAsYOLOv3()
#detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
#detector.loadModel()

#custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True, car=True, bus=True)

#video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join(execution_path, "traffic.mp4"),
#                                output_file_path=os.path.join(execution_path, "traffic_custom_detected")
#                               , frames_per_second=20, log_progress=True)
#print(video_path)
video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "VID_20190509_125542.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic_detectedp")
                                , frames_per_second=20, log_progress=True)
print(video_path)
