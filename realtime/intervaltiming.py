import serial
import argparse
import time
import os
from datetime import datetime


DEFAULT_COM_PORT = "COM4"
DEFAULT_BAUDRATE = 9600
DEFAULT_OUTPUT = "D:\\RealtimePython\\data"


def initialize_serial(com_port, baud_rate):
    ser = serial.Serial(com_port, baud_rate, timeout=1)
    return ser


def initialize_logging(save_dir):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M")
    session_dir = os.path.join(save_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    log_path = os.path.join(session_dir, "interval_log.txt")
    log = open(log_path, "w")
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--com_port", type=str, default=DEFAULT_COM_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ser = initialize_serial(args.com_port, args.baudrate)
    log = initialize_logging(args.output)

    print("Interval timing session started. Press Ctrl+C to quit.")
    log.write("Session start\n")
    log.flush()
    
    # Send start signal to Arduino
    print("Sending start signal to Arduino...")
    ser.write(b'1')  # Send any byte to start the task
    time.sleep(0.1)  # Give Arduino time to process

    try:
        while True:
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


if __name__ == "__main__":
    main()


