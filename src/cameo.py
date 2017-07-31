import cv2
from managers import WindowManager, CaptureManager

class Cameo(object):
    def __init__(self):
        self._window_manager = WindowManager('Cameo', self.on_key_press)
        self._capture_manager = CaptureManager(cv2.VideoCapture(0), self._window_manager, True)

    def run(self):
        """Run the main loop"""
        self._window_manager.create_window()
        while self._window_manager.is_window_created:
            self._capture_manager.enter_frame()
            frame = self._capture_manager.frame

            # filter the frame

            self._capture_manager.exit_frame()
            self._window_manager.process_events()

    def on_key_press(self, key_code):
        """Handle a key press.
        space       -> take a snapshot
        tab         -> start/stop recording a screen cast
        escape      -> quit
        """
        if key_code == 32:  # space
            self._capture_manager.write_image('screenshot.png')
        elif key_code == 9:     # tab
            if not self._capture_manager.is_writing_video:
                self._capture_manager.start_writing_video('screencast.avi')
            else:
                self._capture_manager.stop_writing_video()
        elif key_code == 27:    # escape
            self._window_manager.destory_window()

if __name__ == '__main__':
    Cameo().run()
