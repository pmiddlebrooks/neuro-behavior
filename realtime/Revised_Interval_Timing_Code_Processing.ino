// Revised_Interval_Timing_Code_Processing.ino
// Same interval timing task as Revised_Interval_Timing_Code.ino, but streams
// logged events over Serial so a Processing sketch can write the CSV to disk.
//
// Handshake: Arduino waits for at least one byte from the host (Processing)
// before starting the task. Processing sends 'S' (or any byte) after opening
// the serial port.
//
// CSV format (one line per event): timestamp_ms,event,value,
// Events: B (beam), REWARD, ERROR

#include <string.h>

// Pins
const int SOLENOID_PORT = 12;
const int BEAM_BREAK_PORT = 2;   // active low
const int LED_PORT = 10;         // reward confirmation LED

// Timing (ms)
const int SOLENOID_ON_TIME = 30;   // solenoid stays open
const int SOLENOID_OFF_TIME = 300; // wait btwn pulses
const int SOLENOID_REPEATS = 1;   // additional pulses

const unsigned long INTERVAL_DURATION = 3000;   // min wait time after confirmed leave
const unsigned long MIN_LEAVE_TIME = 100;       // min leave time to confirm exit
const unsigned long LED_PULSE = 100;            // reward LED duration

char buffer[100];

// State
bool firstPoke = true;
bool givingReward = false;
bool inBurst = false;
bool leavePending = false;
bool timerArmed = false;

int beamState = 0;
int lastBeamState = 0;

unsigned long initialExitTime = 0;
unsigned long leaveConfirmStart = 0;
unsigned long leaveTime = 0;
unsigned long rewardStartTime = 0;
unsigned long lastPulseEndTime = 0;
int pulsesGivenInBurst = 0;

bool ledActive = false;
unsigned long ledStartTime = 0;

/**
 * log_line
 *  - tag: event code (e.g., "B", "REWARD", "ERROR")
 *  - value: integer value
 *
 * Goal:
 *  Send one CSV line over Serial for the log. Format: timestamp_ms,event,value,
 */
void log_line(const char *tag, int value) {
  sprintf(buffer, "%lu,%s,%d,", millis(), tag, value);
  Serial.println(buffer);
}

void write_beam(int state) { log_line("B", state); }
void write_reward() { log_line("REWARD", 1); }
void write_error() { log_line("ERROR", 1); }

/**
 * open_solenoid
 *
 * Goal:
 *  Activate solenoid and record reward start time.
 */
void open_solenoid() {
  digitalWrite(SOLENOID_PORT, HIGH);
  givingReward = true;
  rewardStartTime = millis();
}

/**
 * close_solenoid
 *
 * Goal:
 *  Deactivate solenoid and record pulse end time.
 */
void close_solenoid() {
  digitalWrite(SOLENOID_PORT, LOW);
  givingReward = false;
  lastPulseEndTime = millis();
}

/**
 * pulse_led
 *
 * Goal:
 *  Turn on reward confirmation LED for LED_PULSE ms.
 */
void pulse_led() {
  digitalWrite(LED_PORT, HIGH);
  ledActive = true;
  ledStartTime = millis();
}

/**
 * wait_for_processing_start
 *
 * Goal:
 *  Block until the host sends at least one byte over Serial (handshake).
 */
void wait_for_processing_start() {
  while (!Serial) {
    ;
  }
  while (Serial.available() == 0) {
    ;
  }
  while (Serial.available() > 0) {
    Serial.read();
  }
}

void setup() {
  pinMode(SOLENOID_PORT, OUTPUT);
  pinMode(LED_PORT, OUTPUT);
  pinMode(BEAM_BREAK_PORT, INPUT);

  digitalWrite(SOLENOID_PORT, LOW);
  digitalWrite(LED_PORT, LOW);

  Serial.begin(9600);
  wait_for_processing_start();

  // CSV header (matches Revised_Interval_Timing_Code.ino format)
  Serial.println("timestamp_ms,event,value,");

  // Task starts immediately after header
}

void loop() {
  beamState = !digitalRead(BEAM_BREAK_PORT);

  if (beamState != lastBeamState) {
    write_beam(beamState);
    lastBeamState = beamState;

    if (beamState) {
      unsigned long now = millis();

      if (firstPoke) {
        firstPoke = false;
        write_reward();
        inBurst = true;
        pulsesGivenInBurst = 0;
        open_solenoid();
      } else if (timerArmed) {
        if (now - leaveTime >= INTERVAL_DURATION) {
          write_reward();
          inBurst = true;
          pulsesGivenInBurst = 0;
          open_solenoid();
        } else {
          write_error();
        }
      }

      timerArmed = false;
      leavePending = false;
    } else {
      leavePending = true;
      initialExitTime = millis();
      leaveConfirmStart = millis();
    }
  }

  if (leavePending) {
    if (!beamState && (millis() - leaveConfirmStart >= MIN_LEAVE_TIME)) {
      leavePending = false;
      leaveTime = initialExitTime;
      timerArmed = true;
    }
    if (beamState) {
      leavePending = false;
    }
  }

  if (givingReward) {
    if (millis() - rewardStartTime >= SOLENOID_ON_TIME) {
      close_solenoid();
    }
  }

  if (inBurst && !givingReward) {
    if (pulsesGivenInBurst < SOLENOID_REPEATS) {
      if (millis() - lastPulseEndTime >= SOLENOID_OFF_TIME) {
        open_solenoid();
        pulsesGivenInBurst++;
      }
    } else {
      inBurst = false;
      pulse_led();
    }
  }

  if (ledActive && (millis() - ledStartTime >= LED_PULSE)) {
    digitalWrite(LED_PORT, LOW);
    ledActive = false;
  }
}
