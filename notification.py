import time
# from winotify import Notification, audio
from windows_toasts import Toast, WindowsToaster, ToastButton, ToastAudio, AudioSource, ToastDisplayImage
from pathlib import Path 
import threading
import os


class WindowNotify:

    '''
    # ******** winotify library ***********

    def __init__(self, app_id):
        self.app_id = app_id

    def _send(self, title, msg, action_label=None, action_link=None):
        toast = Notification (
            app_id = self.app_id,
            title = title,
            msg = msg,
            duration = "long"
        )

        toast.set_audio(audio.Default, loop=False)
        
        if action_label and action_link:
            toast.add_actions(label = action_label, launch = action_link)

        toast.show()

    def _send_async(self, title, msg, action_label=None, action_link=None):
        thread = threading.Thread (
            target = self._send,
            args = (title, msg, action_label, action_link),
            daemon = True
        )

        thread.start()

    '''


    # ******** windows_toasts library ***********

    def __init__(self, app_id):
        self.app_id = app_id
        self.icon_path = Path(os.path.abspath("wafer_icon.png"))

    def _send(self, title, msg, action_label=None, action_link=None):
        toaster = WindowsToaster(self.app_id) 
        toast = Toast()
        toast.text_fields = [title, msg]
        toast.audio = ToastAudio(AudioSource.Default, looping=False)
        
        if self.icon_path.exists():
            toast.AddImage(ToastDisplayImage.fromPath(self.icon_path))

        if action_label and action_link:
            toast.AddAction(ToastButton(action_label, action_link))

        toaster.show_toast(toast)


    def _send_async(self, title, msg, action_label=None, action_link=None):
        thread = threading.Thread (
            target = self._send,
            args = (title, msg, action_label, action_link),
            daemon = True
        )

        thread.start()

    
        
    # ********************* any cases of notofication *************************

    def started(self):
        self._send_async (
            title = "Automatic Monitor Started",
            msg = "Monitoring is now automatically running in the background."
        )

    def stopped(self):
        self._send (
            title = "Automatic Monitor Stopped",
            msg = "Automatic monitor has been stopped."
        )

    def detected(self):
        self._send (
            title = "Tomato Cat Detected",
            msg = "Tomamto cat is showing on the screen now."
        )

    def alert(self, title, msg, action_label=None, action_link=None):
        self._send_async (
            title = title,
            msg = msg,
            action_label = action_label,
            action_link = action_link
        )


