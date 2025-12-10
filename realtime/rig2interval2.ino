#include <SPI.h>
#include <SD.h>
#include <string.h>

// Pins
const int SOLENOID_PORT = 12;
const int BEAM_BREAK_PORT = 2;   // active low beam sensor
const int LED_PORT = 10;         // reward cue LED
const int ERROR_LED_PORT = 9;      // error indicator LED

// Timing (ms)
const int SOLENOID_ON_TIME = 30; //50;       // duration solenoid stays open
const int SOLENOID_OFF_TIME = 300;     // spacing between clicks in a burst
const int SOLENOID_REPEATS = 1;        // additional clicks after first
const unsigned long INTERVAL_DURATION = 4 * 1000; // time to wait between trials
const unsigned long REWARD_WINDOW = 1 * 1000;       // LED on window to wait for port entry
const unsigned long ERROR_DURATION = 4 * 1000;       // error timeout duration

// SD card
const int chipSelect = 53;
File logFile;
File dataFile;

char csvFileName[20]; // Use char array instead of String for better SD card compatibility

// State
bool ledOn = false;
bool givingReward = false;
bool inBurst = false;
bool inITI = false;              // true when counting ITI
bool trialReady = false;         // LED on, waiting for beam break
bool errorLedOn = false;          // error indicator LED state
bool inErrorTimeout = false;      // true when in error timeout period

// Beam and serial
int lastBeamState = 0;
int beamState = 0;

// Timers
unsigned long rewardStartTime = 0;     // solenoid open timestamp
unsigned long lastPulseEndTime = 0;    // timestamp when last solenoid pulse ended
unsigned long itiStartTime = 0;        // when ITI started
unsigned long ledOnTime = 0;           // when LED turned on (start of reward window)
unsigned long errorStartTime = 0;      // when error timeout started

// Counters
int pulsesGivenInBurst = 0;            // counts pulses after the initial click

char buffer[100];

void setup() {
  pinMode(SOLENOID_PORT, OUTPUT);
  pinMode(LED_PORT, OUTPUT);
  pinMode(BEAM_BREAK_PORT, INPUT);
  pinMode(ERROR_LED_PORT, OUTPUT);
  pinMode(chipSelect, OUTPUT);

  digitalWrite(SOLENOID_PORT, LOW);
  digitalWrite(LED_PORT, LOW);
  digitalWrite(ERROR_LED_PORT, LOW);

  Serial.begin(9600);
  delay(100); // Give serial time to initialize
  
  // Seed random number generator with analog noise (like JS_SD.ino line 80)
  randomSeed(analogRead(0));
  
  // Initialize SD card - try both methods (JS_SD.ino uses SD.begin() without parameter)
  if (!SD.begin()) {
    // If that fails, try with chipSelect (for Mega)
    if (!SD.begin(chipSelect)) {
      Serial.println("SD card initialization failed!");
      while (1);
    } else {
      Serial.println("SD card initialized with chipSelect.");
    }
  } else {
    Serial.println("SD card initialized successfully.");
  }
  
  // Parse compile-time date for log.txt
  char monthStr[4] = "";
  int day = 1, year = 2024;
  
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
  
  // Write date to log.txt file
  logFile = SD.open("log.txt", FILE_WRITE);
  if (logFile) {
    sprintf(buffer, "Date: %04d-%02d-%02d", year, month, day);
    logFile.println(buffer);
    logFile.flush();
    logFile.close();
    Serial.print("log.txt created with date: ");
    Serial.println(buffer);
  } else {
    Serial.println("Warning: Could not open log.txt - continuing without log file.");
  }
  
  // Add a small delay to ensure SD card is fully ready
  delay(100);
  
  // Generate random filename for CSV data file (exactly like JS_SD.ino line 81-82)
  unsigned long randomNum = random(100000, 999999); // 6-digit random number
  sprintf(csvFileName, "%lu_data.csv", randomNum);
  
  Serial.print("CSV file will be: ");
  Serial.println(csvFileName);
  
  // Test SD card write access first
  File testFile = SD.open("_test.txt", FILE_WRITE);
  if (testFile) {
    testFile.println("test");
    testFile.flush();
    testFile.close();
    SD.remove("_test.txt");
    Serial.println("SD card write test: PASSED");
  } else {
    Serial.println("WARNING: SD card write test FAILED!");
  }
  
  // Check if file already exists and remove it (in case of previous failed run)
  if (SD.exists(csvFileName)) {
    Serial.print("File already exists, removing: ");
    Serial.println(csvFileName);
    SD.remove(csvFileName);
    delay(50); // Give SD card time to process removal
  }
  
  // Create the CSV file by writing a test line (like JS_SD.ino creates file in loop)
  dataFile = SD.open(csvFileName, FILE_WRITE);
  if (dataFile) {
    // Write a test line to ensure file is created
    dataFile.println("0,INIT,0,SETUP_COMPLETE");
    dataFile.flush(); // Ensure data is written immediately
    dataFile.close();
    Serial.println("CSV file created successfully.");
    
    // Verify file was created by checking if it exists
    if (SD.exists(csvFileName)) {
      Serial.println("CSV file verified on SD card.");
    } else {
      Serial.println("WARNING: CSV file created but not found on SD card!");
    }
  } else {
    Serial.print("ERROR: Could not create CSV file: ");
    Serial.println(csvFileName);
    Serial.print("Filename length: ");
    Serial.println(strlen(csvFileName));
    Serial.println("Attempting to list files on SD card...");
    File root = SD.open("/");
    if (root) {
      int fileCount = 0;
      while (true) {
        File entry = root.openNextFile();
        if (!entry) break;
        Serial.print("  File found: ");
        Serial.println(entry.name());
        entry.close();
        fileCount++;
      }
      root.close();
      Serial.print("Total files: ");
      Serial.println(fileCount);
    } else {
      Serial.println("ERROR: Could not open root directory!");
    }
  }
  
  // Auto-start without waiting for Python
  writeState("AUTO_START");
  
  // Begin first trial: LED on only (no initial click)
  endITIAndArmTrial();
}

void logLine(const char* tag, int value) {
  dataFile = SD.open(csvFileName, FILE_WRITE);
  if (dataFile) {
    sprintf(buffer, "%lu,%s,%d,", millis(), tag, value);
    dataFile.println(buffer);
    dataFile.flush();
    dataFile.close();
  } else {
    // Only print error once to avoid spam
    static bool errorPrinted = false;
    if (!errorPrinted) {
      Serial.print("ERROR: Could not open CSV file for writing: ");
      Serial.println(csvFileName);
      errorPrinted = true;
    }
  }
}

void writeLEDOn() { logLine("L", 1); }
void writeLEDOff() { logLine("L", 0); }
void writeErrorOn() { logLine("E", 1); }
void writeErrorOff() { logLine("E", 0); }
void writeRewardOn() { logLine("R", 1); }
void writeRewardOff() { logLine("R", 0); }
void writeBeam(int state) { logLine("B", state); }
void writeState(const char* tag) { 
  dataFile = SD.open(csvFileName, FILE_WRITE);
  if (dataFile) {
    sprintf(buffer, "%lu,S,0,%s", millis(), tag); 
    dataFile.println(buffer); 
    dataFile.flush();
    dataFile.close();
  } else {
    // Only print error once to avoid spam
    static bool errorPrinted = false;
    if (!errorPrinted) {
      Serial.print("ERROR: Could not open CSV file for writing: ");
      Serial.println(csvFileName);
      errorPrinted = true;
    }
  }
}

void turnOnLED() { digitalWrite(LED_PORT, HIGH); ledOn = true; writeLEDOn(); }
void turnOffLED() { digitalWrite(LED_PORT, LOW); ledOn = false; writeLEDOff(); }
void turnOnError() { digitalWrite(ERROR_LED_PORT, HIGH); errorLedOn = true; writeErrorOn(); }
void turnOffError() { digitalWrite(ERROR_LED_PORT, LOW); errorLedOn = false; writeErrorOff(); }

void openSolenoid() { digitalWrite(SOLENOID_PORT, HIGH); givingReward = true; rewardStartTime = millis(); writeRewardOn(); }
void closeSolenoid() { digitalWrite(SOLENOID_PORT, LOW); givingReward = false; writeRewardOff(); lastPulseEndTime = millis(); }

void startErrorTimeout() {
  inErrorTimeout = true;
  errorStartTime = millis();
  turnOffLED(); // Turn off reward light
  if (!errorLedOn) { turnOnError(); }
  writeState("ERROR_TIMEOUT_START");
}

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


void loop() {
  // Read beam (active low -> invert)
  beamState = !digitalRead(BEAM_BREAK_PORT);
  if (beamState != lastBeamState) {
    writeBeam(beamState);
    lastBeamState = beamState;

    // Beam broken (mouse enters port)
    if (beamState) {
      // If beam broken during ITI, start error timeout
      if (inITI) {
        inITI = false; // Exit ITI state
        startErrorTimeout();
        writeState("BEAM_BROKEN_DURING_ITI");
      }
      // If beam broken during error timeout, restart error timeout
      else if (inErrorTimeout) {
        startErrorTimeout(); // Restart the error timeout
        writeState("BEAM_BROKEN_DURING_ERROR_TIMEOUT");
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
      if (!givingReward && !inBurst && !inITI && !trialReady && !inErrorTimeout) {
        startITI();
      }
      // If beam leaves during burst, wait for burst to complete before starting ITI
      else if (inBurst && !givingReward) {
        writeState("BEAM_LEAVE_DURING_BURST");
      }
    }
  }

  // Manage ITI
  if (inITI) {
    if (millis() - itiStartTime >= INTERVAL_DURATION) {
      endITIAndArmTrial(); // LED on only, no initial click
    }
  }

  // If trial is armed (LED on), wait for beam within REWARD_WINDOW
  if (trialReady) {
    if (millis() - ledOnTime >= REWARD_WINDOW) {
      // window expired, start error timeout
      trialReady = false;
      startErrorTimeout();
      writeState("REWARD_WINDOW_EXPIRED");
    }
  }

  // Manage error timeout
  if (inErrorTimeout) {
    if (millis() - errorStartTime >= ERROR_DURATION) {
      // Error timeout complete, turn off error light and start new ITI
      inErrorTimeout = false;
      turnOffError();
      writeState("ERROR_TIMEOUT_COMPLETE");
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
      if (!beamState && !inErrorTimeout) {
        startITI();
      }
    }
  }
}

