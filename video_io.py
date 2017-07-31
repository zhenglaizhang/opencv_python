import cv2

video_capture = cv2.VideoCapture('myInputVid.avi')
fps = video_capture.get(cv2.CAP_PROP_FPS)
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

video_writer = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
success, frame = video_capture.read()
while success:  # Loop until there are no more frames.
    video_writer.write(frame)
success, frame = video_capture.read()
