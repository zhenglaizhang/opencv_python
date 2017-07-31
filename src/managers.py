import cv2
import numpy as np
import time


class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture
        self._chanel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFileName = None
        self._videoFileName = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = int(0)
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._chanel

    @channel.setter
    def channel(self, value):
        if self._chanel != value:
            self._chanel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def is_writing_image(self):
        return self._imageFileName is not None

    @property
    def is_writing_video(self):
        return self._videoFileName is not None

    def enter_frame(self):
        """Capture the next frame, if any"""
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exit_frame(self):
        """Draw to the window, write to files, and release the frame."""
        if self.frame is None:
            self._enteredFrame = False
            return

        # update the FPS estimate
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            time_elapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / time_elapsed
        self._framesElapsed += 1

        # draw to window if any
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirrored_frame = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirrored_frame)
            else:
                self.previewWindowManager.show(self._frame)

        # write to file if any
        if self.is_writing_image:
            cv2.imwrite(self._imageFileName, self._frame)
            self._imageFileName = None

        self._write_video_frame()

        # release the frame
        self._frame = None
        self._enteredFrame = False

    def write_image(self, filename):
        self._imageFileName = filename

    def start_writing_video(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        self._videoFileName = filename
        self._videoEncoding = encoding

    def stop_writing_video(self):
        self._videoFileName = None
        self._videoEncoding = None
        self._videoWriter = None

    def _write_video_frame(self):
        if not self.is_writing_video:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:  # capture's FPS is unknown
                if self._framesElapsed < 20:
                    # wait until more frames elapse so that the estimate is more stable
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(
                self._videoFileName,
                self._videoEncoding,
                fps,
                size
            )
        self._videoWriter.write(self._frame)


class WindowManager(object):
    def __init__(self, window_name, key_press_callback=None):
        self.key_press_callback = key_press_callback
        self._window_name = window_name
        self._is_window_created = False

    @property
    def is_window_created(self):
        return self._is_window_created

    def create_window(self):
        cv2.namedWindow(self._window_name)
        self._is_window_created = True

    def show(self, frame):
        cv2.imshow(self._window_name, frame)

    def destory_window(self):
        cv2.destroyWindow(self._window_name)
        self._is_window_created = False

    def process_events(self):
        key_code = cv2.waitKey(1)
        if self.key_press_callback is not None and key_code != -1:
            key_code &= 0xFF    # discard any non-ASCII info encoded by GTK
            self.key_press_callback(key_code)
