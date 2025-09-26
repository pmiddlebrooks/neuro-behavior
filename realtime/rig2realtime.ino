#include <SPI.h>
#include <SD.h>

const int SOLENOID_PORT = 13;
const int BEAM_BREAK_PORT = 2;
const int LED_PORT = 10;
const int SOLENOID_ON_TIME = 50; // ms
const int SOLENOID_OFF_TIME = 500; // ms = sec * ms / sec
const int SOLENOID_REPEATS = 2;
const int INTERTRIAL_INTERVAL = 3 * 1000; // ms = sec * ms / sec ||| minimum duration of time between rewards
const int chipSelect = 53; //  SD card

File dataFile;

bool criteriaMet = false;
bool LEDOn = false;
bool givingReward = false;
bool inRewardBurst = false;

char serialCode;
char buffer[100];

int rewardsGivenInGroup = 0;
int rewardStartTime = 0;
int currTime = 0;
int lastRewardTime = 0;
int elapsedReward = 0;
int timeSinceLastReward = 0;
int timeSincePortEntry = 0;
int lastPortEntryTime = 0;
int lastBeamState = 0;
int beamState = 0;

void writeBeamStateChange(int beamState) {
  sprintf(buffer, "%d B %d", millis(), beamState);
  dataFile.println(buffer);
  dataFile.flush();
}

void writeBehaviorIdentified() {
  sprintf(buffer, "%d A", millis());
  dataFile.println(buffer);
  dataFile.flush();
}

void writeSolenoidOpened() {
  sprintf(buffer, "%d R 1", millis());
  dataFile.println(buffer);
  dataFile.flush();
}

void writeLEDOn() {
  sprintf(buffer, "%d L", millis());
  dataFile.println(buffer);
  dataFile.flush();
}

void writeRewardOff() {
  sprintf(buffer, "%d R 0", millis());
  dataFile.println(buffer);
  dataFile.flush();
}

void turnOnLED() {
  digitalWrite(LED_PORT, HIGH);
  LEDOn = true;
}

void turnOffLED() {
  digitalWrite(LED_PORT, LOW);
  LEDOn = false;
}

void openSolenoid() {
  digitalWrite(SOLENOID_PORT, HIGH);
  givingReward = true;
  rewardStartTime = millis();
}

void closeSolenoid() {
  digitalWrite(SOLENOID_PORT, LOW);
  givingReward = false;
}

void setup() {
  pinMode(SOLENOID_PORT, OUTPUT);
  pinMode(LED_PORT, OUTPUT);
  pinMode(BEAM_BREAK_PORT, INPUT);

  Serial.begin(9600);

  if (!SD.begin(chipSelect)) {
    Serial.println("SD initialization failed!");
    while (1);
  }
  Serial.println("SD initialized.");

  dataFile = SD.open("log.txt", FILE_WRITE);
  
  // Wait for Python to send start signal
  sprintf(buffer, "%d S WAITING_FOR_START", millis());
  dataFile.println(buffer);
  dataFile.flush();
  
  while (Serial.available() == 0) {
    delay(10);
  }
  // Clear any buffered data
  while (Serial.available() > 0) {
    Serial.read();
  }
  
  sprintf(buffer, "%d S START_SIGNAL_RECEIVED", millis());
  dataFile.println(buffer);
  dataFile.flush();
}

void loop() {
  beamState = !digitalRead(BEAM_BREAK_PORT);

  if (lastBeamState != beamState) {
    writeBeamStateChange(beamState);

    if (criteriaMet && beamState) {
      inRewardBurst = true;
      criteriaMet = false;
      lastPortEntryTime = millis();
    }
    lastBeamState = beamState;
  }

  if (inRewardBurst) {
    if (rewardsGivenInGroup == SOLENOID_REPEATS) {
      rewardsGivenInGroup = 0;
      inRewardBurst = false;
    }
    else if (rewardsGivenInGroup == 0) {
      timeSincePortEntry = millis() - lastPortEntryTime;
      if (timeSincePortEntry >= SOLENOID_OFF_TIME) {
        openSolenoid();
        writeSolenoidOpened();
        rewardsGivenInGroup++;
      }
    }
    else {
      timeSinceLastReward = millis() - lastRewardTime;
      if (timeSinceLastReward >= SOLENOID_OFF_TIME) {
        openSolenoid();
        writeSolenoidOpened();
        rewardsGivenInGroup++;
      }
    }
  }
  else {
    if (Serial.available() > 0) {
      while (Serial.available() > 1) {
        Serial.read();
      }
      serialCode = Serial.read();
      criteriaMet = serialCode == '1';
      if (criteriaMet) {
        currTime = millis();

        writeBehaviorIdentified();

        timeSinceLastReward = currTime - lastRewardTime;

        if (timeSinceLastReward >= INTERTRIAL_INTERVAL) {
          turnOnLED();
          writeLEDOn();
          // No initial solenoid click - only during beam break burst
        }
        else {
          criteriaMet = false;
        }
      }
    }
  }

  if (givingReward) {
    elapsedReward = millis() - rewardStartTime;
    if (elapsedReward >= SOLENOID_ON_TIME) {
      closeSolenoid();
      turnOffLED();
      writeRewardOff();
      lastRewardTime = millis();
    }
  }
}
