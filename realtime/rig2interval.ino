#include <SPI.h>
#include <SD.h>
#include <string.h>

// Pins
const int SOLENOID_PORT = 12;
const int BEAM_BREAK_PORT = 2;   // active low beam sensor
const int LED_PORT = 10;         // reward cue LED
const int ITI_LED_PORT = 9;      // inter-trial interval LED

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

char logFileName[50];
char configFileName[50];

// State
bool ledOn = false;
bool givingReward = false;
bool inBurst = false;
bool inITI = false;              // true when counting ITI
bool trialReady = false;         // LED on, waiting for beam break
bool itiLedOn = false;           // ITI indicator LED state

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
  if (dataFile) {
    sprintf(buffer, "%lu,%s,%d,", millis(), tag, value);
    dataFile.println(buffer);
    dataFile.flush();
  }
}

void writeLEDOn() { logLine("L", 1); }
void writeLEDOff() { logLine("L", 0); }
void writeITIOn() { logLine("I", 1); }
void writeITIOff() { logLine("I", 0); }
void writeRewardOn() { logLine("R", 1); }
void writeRewardOff() { logLine("R", 0); }
void writeBeam(int state) { logLine("B", state); }
void writeState(const char* tag) { 
  if (dataFile) {
    sprintf(buffer, "%lu,S,0,%s", millis(), tag); 
    dataFile.println(buffer); 
    dataFile.flush(); 
  }
}

void turnOnLED() { digitalWrite(LED_PORT, HIGH); ledOn = true; writeLEDOn(); }
void turnOffLED() { digitalWrite(LED_PORT, LOW); ledOn = false; writeLEDOff(); }
void turnOnITI() { digitalWrite(ITI_LED_PORT, HIGH); itiLedOn = true; writeITIOn(); }
void turnOffITI() { digitalWrite(ITI_LED_PORT, LOW); itiLedOn = false; writeITIOff(); }

void openSolenoid() { digitalWrite(SOLENOID_PORT, HIGH); givingReward = true; rewardStartTime = millis(); writeRewardOn(); }
void closeSolenoid() { digitalWrite(SOLENOID_PORT, LOW); givingReward = false; writeRewardOff(); lastPulseEndTime = millis(); }

void startITI() { 
  inITI = true; 
  itiStartTime = millis(); 
  trialReady = false; 
  turnOffLED(); // Ensure LED is off during ITI
  if (!itiLedOn) { turnOnITI(); }
  writeState("ITI_START"); 
}
void endITIAndArmTrial() { 
  inITI = false; 
  trialReady = true; 
  if (itiLedOn) { turnOffITI(); }
  turnOnLED(); 
  ledOnTime = millis(); 
  writeState("TRIAL_ARMED"); 
}

void setup() {
  pinMode(SOLENOID_PORT, OUTPUT);
  pinMode(LED_PORT, OUTPUT);
  pinMode(BEAM_BREAK_PORT, INPUT);
  pinMode(ITI_LED_PORT, OUTPUT);

  digitalWrite(SOLENOID_PORT, LOW);
  digitalWrite(LED_PORT, LOW);
  digitalWrite(ITI_LED_PORT, LOW);

  Serial.begin(9600);
  delay(100); // Give serial time to initialize
  
  if (!SD.begin(chipSelect)) {
    Serial.println("SD card initialization failed!");
    while (1);
  }
  Serial.println("SD card initialized successfully.");
  
  // Create folder and filename with timestamp
  if (!SD.exists("interval_task")) {
    SD.mkdir("interval_task");
  }
  
  // Use compile-time date and time for unique filename
  // __DATE__ format: "MMM DD YYYY" (e.g., "Jan 15 2024")
  // __TIME__ format: "HH:MM:SS" (e.g., "14:30:45")
  char monthStr[4] = "";
  int day = 1, year = 2024;
  int hour = 0, minute = 0, second = 0;
  
  // Parse __DATE__: "MMM DD YYYY"
  if (sscanf(__DATE__, "%3s %d %d", monthStr, &day, &year) != 3) {
    Serial.println("Warning: Could not parse __DATE__, using defaults");
    year = 2024; monthStr[0] = 'J'; monthStr[1] = 'a'; monthStr[2] = 'n'; day = 1;
  }
  
  // Convert month string to number
  int month = 1;
  if (strcmp(monthStr, "Jan") == 0) month = 1;
  else if (strcmp(monthStr, "Feb") == 0) month = 2;
  else if (strcmp(monthStr, "Mar") == 0) month = 3;
  else if (strcmp(monthStr, "Apr") == 0) month = 4;
  else if (strcmp(monthStr, "May") == 0) month = 5;
  else if (strcmp(monthStr, "Jun") == 0) month = 6;
  else if (strcmp(monthStr, "Jul") == 0) month = 7;
  else if (strcmp(monthStr, "Aug") == 0) month = 8;
  else if (strcmp(monthStr, "Sep") == 0) month = 9;
  else if (strcmp(monthStr, "Oct") == 0) month = 10;
  else if (strcmp(monthStr, "Nov") == 0) month = 11;
  else if (strcmp(monthStr, "Dec") == 0) month = 12;
  
  // Parse __TIME__: "HH:MM:SS"
  if (sscanf(__TIME__, "%d:%d:%d", &hour, &minute, &second) != 3) {
    Serial.println("Warning: Could not parse __TIME__, using defaults");
    hour = 0; minute = 0; second = 0;
  }
  
  // Create filenames with date and time: YYYYMMDD_HHMMSS
  sprintf(configFileName, "interval_task/config_%04d%02d%02d_%02d%02d%02d.txt", 
          year, month, day, hour, minute, second);
  sprintf(logFileName, "interval_task/log_%04d%02d%02d_%02d%02d%02d.csv", 
          year, month, day, hour, minute, second);
  
  Serial.print("Config file: ");
  Serial.println(configFileName);
  Serial.print("Log file: ");
  Serial.println(logFileName);
  
  // Write config file with timing constants
  configFile = SD.open(configFileName, FILE_WRITE);
  if (configFile) {
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
    configFile.flush();
    configFile.close();
    Serial.println("Config file created and saved.");
  } else {
    Serial.println("Warning: Could not open config file - continuing without config file.");
  }
  
  // Write CSV header
  dataFile = SD.open(logFileName, FILE_WRITE);
  if (dataFile) {
    dataFile.println("timestamp_ms,event,value,state_label");
    dataFile.flush(); // Ensure header is written immediately
    Serial.println("Log file created. Starting task...");
  } else {
    Serial.println("Warning: Could not open log file - continuing without logging.");
    Serial.println("Task will run but data will not be saved.");
  }
  
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
      // If beam broken during ITI, mark error and turn off ITI indicator
      if (inITI) {
        if (itiLedOn) { turnOffITI(); }
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
        if (!itiLedOn) { turnOnITI(); }
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


