#include <SPI.h>
#include <SD.h>

// Pins
const int SOLENOID_PORT = 13;
const int BEAM_BREAK_PORT = 2;   // active low beam sensor
const int LED_PORT = 10;         // reward cue LED

// Timing (ms)
const int SOLENOID_ON_TIME = 50;       // duration solenoid stays open

// SD card
const int chipSelect = 53;
File dataFile;

// State
bool ledOn = false;
bool givingReward = false;
bool waitingForBeam = false;

// Beam and serial
int lastBeamState = 0;
int beamState = 0;

// Timers
unsigned long rewardStartTime = 0;     // solenoid open timestamp

char buffer[100];

void logLine(const char* tag, int value) {
  sprintf(buffer, "%lu %s %d", millis(), tag, value);
  dataFile.println(buffer);
  dataFile.flush();
}

void writeLEDOn() { logLine("L", 1); }
void writeLEDOff() { logLine("L", 0); }
void writeRewardOn() { logLine("R", 1); }
void writeRewardOff() { logLine("R", 0); }
void writeBeam(int state) { logLine("B", state); }
void writeState(const char* tag) { sprintf(buffer, "%lu S %s", millis(), tag); dataFile.println(buffer); dataFile.flush(); }

void turnOnLED() { digitalWrite(LED_PORT, HIGH); ledOn = true; writeLEDOn(); }
void turnOffLED() { digitalWrite(LED_PORT, LOW); ledOn = false; writeLEDOff(); }

void openSolenoid() { 
  digitalWrite(SOLENOID_PORT, HIGH); 
  givingReward = true; 
  rewardStartTime = millis(); 
  writeRewardOn(); 
}
void closeSolenoid() { 
  digitalWrite(SOLENOID_PORT, LOW); 
  givingReward = false; 
  writeRewardOff(); 
}

void setup() {
  pinMode(SOLENOID_PORT, OUTPUT);
  pinMode(LED_PORT, OUTPUT);
  pinMode(BEAM_BREAK_PORT, INPUT);

  digitalWrite(SOLENOID_PORT, LOW);
  digitalWrite(LED_PORT, LOW);

  Serial.begin(9600);

  if (!SD.begin(chipSelect)) {
    while (1);
  }
  dataFile = SD.open("log.txt", FILE_WRITE);

  writeState("HOTKEY_MODE_READY");
  Serial.println("Hotkey mode ready. Commands:");
  Serial.println("1 - Manual solenoid click");
  Serial.println("2 - Turn on reward light and wait for beam break");
  Serial.println("3 - Turn off reward light and stop waiting");
}

void loop() {
  // Read beam (active low -> invert)
  beamState = !digitalRead(BEAM_BREAK_PORT);
  if (beamState != lastBeamState) {
    writeBeam(beamState);
    lastBeamState = beamState;

    // If waiting for beam break and beam is broken, deliver reward
    if (waitingForBeam && beamState) {
      waitingForBeam = false;
      turnOffLED();
      openSolenoid();
      writeState("BEAM_BREAK_REWARD_DELIVERED");
    }
  }

  // Handle serial commands
  if (Serial.available() > 0) {
    char command = Serial.read();
    // Flush any remaining characters (newline, carriage return, etc.)
    while (Serial.available() > 0) {
      Serial.read();
    }
    
    switch (command) {
      case '1':
        // Manual solenoid click
        // Ensure LED is off and cancel any pending wait before manual click
        if (ledOn) {
          turnOffLED();
        }
        if (waitingForBeam) {
          waitingForBeam = false;
        }
        openSolenoid();
        writeState("MANUAL_SOLENOID_CLICK");
        break;
        
      case '2':
        // Turn on reward light and wait for beam break
        if (!waitingForBeam) {
          waitingForBeam = true;
          turnOnLED();
          writeState("REWARD_LIGHT_ON_WAITING");
        }
        break;
        
      case '3':
        // Turn off reward light and stop waiting
        if (waitingForBeam) {
          waitingForBeam = false;
          turnOffLED();
          writeState("REWARD_LIGHT_OFF_STOPPED");
        }
        break;
        
      default:
        // Ignore unknown commands
        break;
    }
  }

  // Handle ongoing solenoid pulse - matches rig2interval.ino logic
  if (givingReward) {
    if (millis() - rewardStartTime >= SOLENOID_ON_TIME) {
      closeSolenoid();
    }
  }
}
