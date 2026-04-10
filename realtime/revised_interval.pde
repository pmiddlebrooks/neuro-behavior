// revised_interval.pde
// Processing sketch to log Revised_Interval_Timing_Code events from Arduino to a CSV file.
//
// Workflow:
//  1. Upload and run Revised_Interval_Timing_Code_Processing.ino on the Arduino.
//  2. Start this Processing sketch.
//  3. Set serialPortIndex below to match your Arduino's port.
//  4. Sketch opens serial, sends start byte, then streams CSV lines to a timestamped file.
//
// Handshake: Processing sends 'S' after opening the port; Arduino waits for that before starting.
// CSV format: timestamp_ms,event,value,  (events: B, REWARD, ERROR, SYNC)
//   SYNC: value 1 = sync line high (on), 2 = low (off); one row per edge
//
// Press 'q' or 'Q' to stop logging and exit.

import processing.serial.*;

Serial rigPort;
PrintWriter logFile;

int serialPortIndex = 1;

/**
 * get_log_file_name
 *
 * Goal:
 *  Build a timestamped CSV filename. Format: revised_interval_YYYYMMDD_HHMMSS.csv
 */
String get_log_file_name() {
  return String.format("revised_interval_%04d%02d%02d_%02d%02d%02d.csv",
                       year(), month(), day(), hour(), minute(), second());
}

void setup() {
  size(500, 200);
  background(0);
  fill(255);
  textAlign(LEFT, TOP);

  println("Available serial ports:");
  String[] ports = Serial.list();
  for (int i = 0; i < ports.length; i++) {
    println(i + ": " + ports[i]);
  }

  if (ports.length == 0) {
    println("No serial ports found. Connect the Arduino and restart.");
    exit();
    return;
  }

  if (serialPortIndex < 0 || serialPortIndex >= ports.length) {
    println("serialPortIndex is out of range. Set it to the correct port index.");
    exit();
    return;
  }

  String fileName = get_log_file_name();
  logFile = createWriter(fileName);
  println("Logging to: " + fileName);

  text("Logging to: " + fileName + "\nPress 'q' to quit.", 10, 10);

  String portName = ports[serialPortIndex];
  println("Opening port: " + portName);
  rigPort = new Serial(this, portName, 9600);
  rigPort.bufferUntil('\n');
  rigPort.clear();

  delay(2000);
  rigPort.write('S');
  rigPort.write('\n');
}

void draw() {
}

void serialEvent(Serial p) {
  try {
    String line = p.readStringUntil('\n');
    if (line != null) {
      line = trim(line);
      if (line.length() == 0) return;
      println(line);
      if (logFile != null) {
        logFile.println(line);
        logFile.flush();
      }
    }
  } catch (Exception ex) {
    println("serialEvent exception: " + ex.getMessage());
  }
}

void keyPressed() {
  if (key == 'q' || key == 'Q') {
    println("Stopping logging and exiting.");
    if (logFile != null) {
      logFile.flush();
      logFile.close();
    }
    if (rigPort != null) {
      rigPort.stop();
    }
    exit();
  }
}
