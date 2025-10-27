#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <RTClib.h>

// Pins
const int SOLENOID_PORT = 12;
const int BEAM_BREAK_PORT = 2;   // active low beam sensor
const int LED_PORT = 10;         // reward cue LED

// Timing (ms)
const int SOLENOID_ON_TIME = 30; //50;       // duration solenoid stays open
const int SOLENOID_OFF_TIME = 300;     // spacing between clicks in a burst
const int SOLENOID_REPEATS = 1;        // additional clicks after first
const unsigned long INTERTRIAL_INTERVAL = 3 * 1000; // time to wait between trials
const unsigned long REWARD_WINDOW = 2 * 1000;       // LED on window to wait for port entry

// SD card
const int chipSelect = 53;
File dataFile;
File configFile;
RTC_DS1307 rtc;

char logFileName[50];
char configFileName[50];

// State
bool ledOn = false;
bool givingReward = false;
bool inBurst = false;
bool inITI = false;              // true when counting ITI
bool trialReady = false;         // LED on, waiting for beam break

// Beam and serial
int lastBeamState = 0;
int beamState = 0;

// Timers
unsigned long rewardStartTime = 0;     // solenoid open timestamp
unsigned long lastPulseEndTime = 0;    // timestamp when last solenoid pulse ended
unsigned long itiStartTime = 0;        // when ITI started
unsigned long ledOnTime = 0;           // when LED turned on (start of reward window)

// Counters
int pulsesGivenInBurst = 0;            // counts pulses after the initial click

char buffer[100];

void logLine(const char* tag, int value) {
  sprintf(buffer, "%lu,%s,%d,", millis(), tag, value);
  dataFile.println(buffer);
  dataFile.flush();
}

void writeLEDOn() { logLine("L", 1); }
void writeLEDOff() { logLine("L", 0); }
void writeRewardOn() { logLine("R", 1); }
void writeRewardOff() { logLine("R", 0); }
void writeBeam(int state) { logLine("B", state); }
void writeState(const char* tag) { sprintf(buffer, "%lu,S,0,%s", millis(), tag); dataFile.println(buffer); dataFile.flush(); }

void turnOnLED() { digitalWrite(LED_PORT, HIGH); ledOn = true; writeLEDOn(); }
void turnOffLED() { digitalWrite(LED_PORT, LOW); ledOn = false; writeLEDOff(); }

void openSolenoid() { digitalWrite(SOLENOID_PORT, HIGH); givingReward = true; rewardStartTime = millis(); writeRewardOn(); }
void closeSolenoid() { digitalWrite(SOLENOID_PORT, LOW); givingReward = false; writeRewardOff(); lastPulseEndTime = millis(); }

void startITI() { 
  inITI = true; 
  itiStartTime = millis(); 
  trialReady = false; 
  turnOffLED(); // Ensure LED is off during ITI
  writeState("ITI_START"); 
}
void endITIAndArmTrial() { 
  inITI = false; 
  trialReady = true; 
  turnOnLED(); 
  ledOnTime = millis(); 
  writeState("TRIAL_ARMED"); 
}

void setup() {
  pinMode(SOLENOID_PORT, OUTPUT);
  pinMode(LED_PORT, OUTPUT);
  pinMode(BEAM_BREAK_PORT, INPUT);

  digitalWrite(SOLENOID_PORT, LOW);
  digitalWrite(LED_PORT, LOW);

  Serial.begin(9600);
  Wire.begin();
  
  if (!SD.begin(chipSelect)) {
    while (1);
  }
  
  // Create folder and filename with timestamp
  if (!SD.exists("interval_task")) {
    SD.mkdir("interval_task");
  }
  
  DateTime now = rtc.now();
  
  // Create config filename
  sprintf(configFileName, "interval_task/config_%04d%02d%02d_%02d%02d.txt", 
          now.year(), now.month(), now.day(), now.hour(), now.minute());
  
  // Create CSV log filename
  sprintf(logFileName, "interval_task/log_%04d%02d%02d_%02d%02d.csv", 
          now.year(), now.month(), now.day(), now.hour(), now.minute());
  
  // Write config file with timing constants
  configFile = SD.open(configFileName, FILE_WRITE);
  configFile.println("TIMING_CONSTANTS:");
  sprintf(buffer, "SOLENOID_ON_TIME: %d ms", SOLENOID_ON_TIME);
  configFile.println(buffer);
  sprintf(buffer, "SOLENOID_OFF_TIME: %d ms", SOLENOID_OFF_TIME);
  configFile.println(buffer);
  sprintf(buffer, "SOLENOID_REPEATS: %d", SOLENOID_REPEATS);
  configFile.println(buffer);
  sprintf(buffer, "INTERTRIAL_INTERVAL: %d ms", INTERTRIAL_INTERVAL);
  configFile.println(buffer);
  sprintf(buffer, "REWARD_WINDOW: %d ms", REWARD_WINDOW);
  configFile.println(buffer);
  configFile.close();
  
  // Write CSV header
  dataFile = SD.open(logFileName, FILE_WRITE);
  dataFile.println("timestamp_ms,event,value,state_label");
  
  // Auto-start without waiting for Python
  writeState("AUTO_START");
  
  // Begin first trial: LED on only (no initial click)
  endITIAndArmTrial();
}

void loop() {
  // Read beam (active low -> invert)
  beamState = !digitalRead(BEAM_BREAK_PORT);
  if (beamState != lastBeamState) {
    writeBeam(beamState);
    lastBeamState = beamState;

    // Beam broken (mouse enters port)
    if (beamState) {
      // If beam broken during ITI, mark that we need to restart ITI when beam is unbroken
      if (inITI) {
        writeState("BEAM_BROKEN_DURING_ITI");
      }
      // If beam broken during trial (LED on), trigger burst
      else if (trialReady) {
        trialReady = false;
        turnOffLED();  // Turn off LED when beam is broken
        inBurst = true;
        pulsesGivenInBurst = 0;
        writeState("BURST_START");
        // Immediately deliver the first pulse upon beam break
        openSolenoid();
      }
    }
    // Beam unbroken (mouse leaves port)
    else {
      // If beam leaves after burst or single pulse, begin ITI
      if (!givingReward && !inBurst && !inITI && !trialReady) {
        startITI();
      }
      // If beam leaves during burst, wait for burst to complete before starting ITI
      else if (inBurst && !givingReward) {
        writeState("BEAM_LEAVE_DURING_BURST");
      }
      // If beam was broken during ITI and now unbroken, restart ITI
      else if (inITI) {
        itiStartTime = millis();
        writeState("ITI_RESTART_AFTER_BEAM_UNBROKEN");
      }
    }
  }

  // Manage ITI
  if (inITI) {
    if (millis() - itiStartTime >= INTERTRIAL_INTERVAL) {
      endITIAndArmTrial(); // LED on only, no initial click
    }
  }

  // If trial is armed (LED on), wait for beam within REWARD_WINDOW
  if (trialReady) {
    if (millis() - ledOnTime >= REWARD_WINDOW) {
      // window expired, turn off LED and start a new ITI
      turnOffLED();
      trialReady = false;
      writeState("REWARD_WINDOW_EXPIRED");
      startITI();
    }
  }

  // Handle ongoing solenoid pulse
  if (givingReward) {
    if (millis() - rewardStartTime >= SOLENOID_ON_TIME) {
      closeSolenoid();
    }
  }

  // Handle burst pulses after initial click
  if (inBurst && !givingReward) {
    unsigned long sinceLastPulse = millis() - lastPulseEndTime;
    if (pulsesGivenInBurst < SOLENOID_REPEATS) {
      if (sinceLastPulse >= SOLENOID_OFF_TIME) {
        openSolenoid();
        pulsesGivenInBurst++;
      }
    } else {
      // Burst complete
      inBurst = false;
      writeState("BURST_COMPLETE");
      // If beam is now not broken, begin ITI; otherwise wait until it leaves then ITI will start via beam transition branch
      if (!beamState) {
        startITI();
      }
    }
  }
}


