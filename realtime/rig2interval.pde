// rig2interval.pde
// Processing sketch to log rig2interval events from Arduino to a CSV file.
//
// Workflow:
//  1. Upload and run rig2intervalProcessing.ino on the Arduino.
//  2. Start this Processing sketch.
//  3. Select the correct serial port index below (serialPortIndex).
//  4. The sketch opens the serial port, sends a start byte to the Arduino,
//     and then streams every CSV line from the Arduino to a timestamped file.
//
// Handshake:
//  - Processing sends a single byte ('S') after opening the serial port.
//  - Arduino waits for at least one byte before starting and sending data.
//
// Controls:
//  - Press 'q' or 'Q' to stop logging, close the file and exit.

import processing.serial.*;

Serial rigPort;
PrintWriter logFile;

// Set this to the index of the correct serial port shown in the console.
int serialPortIndex = 0;

/**
 * get_log_file_name
 *
 * Goal:
 *  Build a timestamped CSV filename for logging Arduino data.
 *  Format: rig2interval_YYYYMMDD_HHMMSS.csv
 */
String get_log_file_name() {
  return String.format("rig2interval_%04d%02d%02d_%02d%02d%02d.csv",
                       year(), month(), day(), hour(), minute(), second());
}

void setup() {
  size(500, 200);
  background(0);
  fill(255);
  textAlign(LEFT, TOP);

  // List available serial ports for convenience.
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
    println("serialPortIndex is out of range. Adjust it to match the desired port.");
    exit();
    return;
  }

  String portName = ports[serialPortIndex];
  println("Opening port: " + portName);
  rigPort = new Serial(this, portName, 9600);

  // Clear any existing data.
  rigPort.clear();

  // Give Arduino time to reset after opening the port.
  delay(2000);

  // Send handshake byte to tell Arduino we are ready.
  rigPort.write('S');
  rigPort.write('\n');

  // Create a new timestamped CSV file.
  String fileName = get_log_file_name();
  logFile = createWriter(fileName);
  println("Logging to: " + fileName);

  text("Logging to: " + fileName + "\nPress 'q' to quit.", 10, 10);
}

void draw() {
  // Nothing needed here; all logging is handled in serialEvent.
}

/**
 * serialEvent
 *
 * Goal:
 *  Receive complete lines from the Arduino and append them to the CSV log file.
 */
void serialEvent(Serial p) {
  String line = p.readStringUntil('\n');
  if (line != null) {
    line = trim(line);
    if (line.length() == 0) {
      return;
    }
    // Pass-through: Arduino already sends properly formatted CSV lines.
    logFile.println(line);
    logFile.flush();
  }
}

/**
 * keyPressed
 *
 * Goal:
 *  Allow the user to gracefully stop logging and close resources.
 *  Press 'q' or 'Q' to quit.
 */
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

