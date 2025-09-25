import cv2
import torch
import numpy as np
import argparse
import serial
from pypylon import pylon
from background_subtraction import FindRatBoundingBox
from make_video import ModelOptimizer
import time
import keyboard
from datetime import datetime
import os

DEFAULT_CAMERA_SETTINGS = r"D:\RealtimePython\camera_settings\acA3088-57um_25100094_rig2_cropped_binned_30fps.pfs"
DEFAULT_CAMERA_SERIAL_NUMBER = "25100094"
DEFAULT_COM_PORT = "COM4"
DEFAULT_BAUDRATE = 9600
DEFAULT_MODEL_PATH = r"D:\RealtimePython\models\rig2_model_per_frame_scripted.pt"
DEFAULT_OUTPUT = "D:\RealtimePython\data"
DOWNSAMPLE_FACTOR = 1
IMG_SIZE = 224
NUM_FRAMES_BG = 500
CLASS_NAMES = [
    'Body Groom', 'Face/Head/Paw Groom', 'Itch', 'Left Turn', 'Locomotion', 'Other', 'Rear', 'Right Turn'
]

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
            
            print(f"{name}: {serial_number}")
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
        camera.StopGrabbing()
        camera.Close()

class Cropper():
    def __init__(self, bg, frame_height, frame_width, downsample_factor):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.bounding_box_finder = FindRatBoundingBox(bg, downsample_factor=downsample_factor)
    
    def crop_frame(self, frame_bgr):
        cropped_small = self.bounding_box_finder(frame_bgr)

        resized_frame = cv2.resize(cropped_small, (224, 224), interpolation=cv2.INTER_AREA)

        return resized_frame

def get_background(cam, num_frames):
    frames = cam.get_frames(num_frames)

    bg_avg = frames.mean(axis=0)
    
    return bg_avg

def initialize_logging(save_dir, cam, beh, duration):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M")
    session_dir = os.path.join(save_dir, timestamp)

    os.mkdir(session_dir)

    vid_path = os.path.join(session_dir, 'video.mp4')
    log_path = os.path.join(session_dir, 'log.txt')
    session_info_path = os.path.join(session_dir, 'session_info.txt')

    fourCC = cv2.VideoWriter_fourcc(*'mp4v')

    video_out = cv2.VideoWriter(vid_path, fourCC, cam.fps, (cam.width, cam.height), isColor=False)

    log = open(log_path, "w")

    with open(session_info_path, "w") as file:
        file.write(f"Behavior: {beh}\nDuration: {duration} ms")

    return video_out, log

def end_connections(serial, vid_out, camera, logger):
    serial.close()
    vid_out.release()

    camera.camera.StopGrabbing()
    camera.camera.Close()
    logger.close()
    
def initialize_serial_model(com_port, baud_rate, model_path):
    ser = serial.Serial(com_port, baud_rate, timeout=1)

    if ser.is_open:
        print("Serial opened successfully")

    model = ModelOptimizer(model_path, device='cuda')
    
    return ser, model
        
def detect_and_communicate(model, camera, cropper, ser, logger, video_out, beh_duration, chosen_beh):
    beh_idx = CLASS_NAMES.index(chosen_beh)
    beh_duration_frames = int(np.ceil((beh_duration/camera.frame_duration_ms - 2)))

    frame_count = 0
    num_frames_criteria = 0

    while True:
        img_np = camera.get_frames()
        cropped_frame = cropper.crop_frame(img_np)

        video_out.write(img_np[..., 0])

        frame_idx = frame_count % 3
        model.preprocess_frame_gpu(cropped_frame, frame_idx)

        if frame_count >= 2:
            pred_idx = model.run_inference()
            print(f"Predicted: {CLASS_NAMES[pred_idx]}")
            if pred_idx == beh_idx:
                num_frames_criteria += 1
                if num_frames_criteria == beh_duration_frames:
                    ser.write(b'1')
                    logger.write(f"{frame_count}\n")
                    num_frames_criteria = 0
            else:
                num_frames_criteria = 0
                

        if keyboard.is_pressed("q"):
            break

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--behavior", type=str, default='Locomotion',
                        help="Behavior to reinforce")
    parser.add_argument("--duration", type=int, default=200,
                        help="Minimum behavior duration to reinforce")
    parser.add_argument("--camera_specs_path", type=str, default=DEFAULT_CAMERA_SETTINGS,
                        help="Path to Basler specifications")
    parser.add_argument("--camera_serial_number", type=str, default=DEFAULT_CAMERA_SERIAL_NUMBER,
                        help="Serial number of Basler camera")
    parser.add_argument("--com_port", type=str, default=DEFAULT_COM_PORT,
                        help="COM port for arduino communication")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE,
                        help="Baudrate for arduino communication, must be set on arduino as well")

    args = parser.parse_args()

    print(f"Starting real-time behavior reinforcement for behavior {args.behavior} with duration {args.duration} ms")
    camera = BaslerCamera(args.camera_serial_number, args.camera_specs_path)

    bg_avg = get_background(camera, NUM_FRAMES_BG)
    
    cropper = Cropper(bg_avg, camera.height, camera.width, DOWNSAMPLE_FACTOR)
    serial, model = initialize_serial_model(args.com_port, args.baudrate, DEFAULT_MODEL_PATH)

    video_out, log = initialize_logging(DEFAULT_OUTPUT, camera, args.behavior, args.duration)

    print("Press and hold \"q\" at anytime to end program")
    detect_and_communicate(model, camera, cropper, serial, log, video_out, args.duration, args.behavior)
    end_connections(serial, video_out, camera, log)
    
if __name__ == "__main__":
    main()
