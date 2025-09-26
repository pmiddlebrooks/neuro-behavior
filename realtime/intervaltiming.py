import serial
import argparse
import time
import os
import cv2
import numpy as np
from datetime import datetime
from pypylon import pylon


DEFAULT_COM_PORT = "COM4"
DEFAULT_BAUDRATE = 9600
DEFAULT_OUTPUT = "D:\\RealtimePython\\data"
DEFAULT_CAMERA_SETTINGS = r"D:\RealtimePython\camera_settings\acA3088-57um_25100094_rig2_cropped_binned_30fps.pfs"
DEFAULT_CAMERA_SERIAL_NUMBER = "25100094"


class BaslerCamera():
    def __init__(self, serial_number, camera_settings):
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        
        num_devices = len(devices)
        device_to_use = None

        print(f"Found {len(devices)} available cameras:")
        for device in devices:
            name = device.GetFriendlyName()
            dev_serial_number = device.GetSerialNumber()
            
            print(f"{name}: {dev_serial_number}")
            device_to_use = device if dev_serial_number == serial_number else None

        if device_to_use is None:
            raise RuntimeError(f"Camera with serial number {serial_number} not found")
        
        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device_to_use))

        self.camera.StartGrabbing()

        self.width = self.camera.Width.GetValue()
        self.height = self.camera.Height.GetValue()

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        self.fps = self.camera.ResultingFrameRate.GetValue()
        self.frame_duration_ms = 1e3 / self.fps

    def _cam_read_convert(self):
        while True:
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result)
                img_np = image.GetArray()

                grab_result.Release()
                
                return img_np

    def get_frames(self, num_frames=1):
        if num_frames == 1:
            return self._cam_read_convert()

        frames = np.zeros((num_frames, self.height, self.width))

        for i in range(num_frames):
            im = self._cam_read_convert()
            frames[i] = im[..., 0]

        return frames

    def end_connection(self):
        self.camera.StopGrabbing()
        self.camera.Close()


def initialize_serial(com_port, baud_rate):
    ser = serial.Serial(com_port, baud_rate, timeout=1)
    return ser


def initialize_logging(save_dir, camera):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M")
    session_dir = os.path.join(save_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    
    # Create log file
    log_path = os.path.join(session_dir, "interval_log.txt")
    log = open(log_path, "w")
    
    # Create video file
    vid_path = os.path.join(session_dir, 'video.mp4')
    fourCC = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(vid_path, fourCC, camera.fps, (camera.width, camera.height), isColor=False)
    
    return log, video_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--com_port", type=str, default=DEFAULT_COM_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera_specs_path", type=str, default=DEFAULT_CAMERA_SETTINGS,
                        help="Path to Basler specifications")
    parser.add_argument("--camera_serial_number", type=str, default=DEFAULT_CAMERA_SERIAL_NUMBER,
                        help="Serial number of Basler camera")
    args = parser.parse_args()

    # Initialize camera
    print("Initializing camera...")
    camera = BaslerCamera(args.camera_serial_number, args.camera_specs_path)
    
    # Initialize serial connection
    ser = initialize_serial(args.com_port, args.baudrate)
    
    # Initialize logging and video recording
    log, video_out = initialize_logging(args.output, camera)

    print("Interval timing session started. Press Ctrl+C to quit.")
    log.write("Session start\n")
    log.flush()
    
    # Send start signal to Arduino
    print("Sending start signal to Arduino...")
    ser.write(b'1')  # Send any byte to start the task
    time.sleep(0.1)  # Give Arduino time to process

    try:
        while True:
            # Capture and record video frame
            img_np = camera.get_frames()
            video_out.write(img_np[..., 0])  # Write grayscale frame
            
            # Arduino runs the schedule; Python can optionally poke or read back.
            # If you want to log Arduino messages: uncomment the following lines and
            # have Arduino Serial.print status lines.
            # if ser.in_waiting:
            #     line = ser.readline().decode(errors='ignore').strip()
            #     if line:
            #         print(line)
            #         log.write(line + "\n")
            #         log.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ser.close()
        log.write("Session end\n")
        log.close()
        video_out.release()
        camera.end_connection()


if __name__ == "__main__":
    main()


