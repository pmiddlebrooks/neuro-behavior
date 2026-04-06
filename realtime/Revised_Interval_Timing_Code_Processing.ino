// Revised_Interval_Timing_Code_Processing.ino
// Same interval timing task as Revised_Interval_Timing_Code.ino, but streams
// logged events over Serial so a Processing sketch can write the CSV to disk.
//
// Handshake: Arduino waits for at least one byte from the host (Processing)
// before starting the task. Processing sends 'S' (or any byte) after opening
// the serial port.
//
// CSV format (one line per event): timestamp_ms,event,value,
// Events: B (beam), REWARD, ERROR, SYNC (external sync line on pin 5)
//   SYNC value 1 = line high (on), 2 = line low (off) — log on each transition

#include <string.h>

// Pins
const int SOLENOID_PORT = 12;
const int BEAM_BREAK_PORT = 2;   // active low
const int LED_PORT = 10;         // reward confirmation LED
const int SYNC_PULSES_PORT = 5;       // external TTL sync (e.g. video frame clock); active HIGH = on

// Timing (ms)
const int SOLENOID_ON_TIME = 20;   // solenoid stays open
const int SOLENOID_OFF_TIME = 300; // wait btwn pulses
const int SOLENOID_REPEATS = 1;    // additional pulses

const unsigned long INTERVAL_DURATION = 4000;   // min wait time after confirmed leave
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

int lastSyncState = -1;  // initialized in setup from digitalRead(SYNC_PULSES_PORT)

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
 *  - tag: event code (e.g., "B", "REWARD", "ERROR", "SYNC")
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
 * write_sync_edge
 *  - syncPinHigh: true if SYNC_PULSES_PORT reads HIGH after transition
 *
 * Goal:
 *  Log external sync line edges for alignment with video (1 = on/high, 2 = off/low).
 */
void write_sync_edge(bool syncPinHigh) {
  log_line("SYNC", syncPinHigh ? 1 : 2);
}

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
  while (Serial.available() == 0) { ; }
  while (Serial.available() > 0) { Serial.read(); }
}

void setup() {
  pinMode(SOLENOID_PORT, OUTPUT);
  pinMode(LED_PORT, OUTPUT);
  pinMode(BEAM_BREAK_PORT, INPUT);
  pinMode(SYNC_PULSES_PORT, INPUT);  // use INPUT_PULLUP if the source is open-collector

  digitalWrite(SOLENOID_PORT, LOW);
  digitalWrite(LED_PORT, LOW);

  Serial.begin(9600);
  wait_for_processing_start();

  // CSV header
  Serial.println("timestamp_ms,event,value,");

  // Baseline sync line so only real transitions are logged (not spurious first edge)
  lastSyncState = digitalRead(SYNC_PULSES_PORT);
}

void loop() {
  int syncState = digitalRead(SYNC_PULSES_PORT);
  if (syncState != lastSyncState) {
    write_sync_edge(syncState == HIGH);
    lastSyncState = syncState;
  }

  beamState = (digitalRead(BEAM_BREAK_PORT) == LOW);
  // beamState = !digitalRead(BEAM_BREAK_PORT);

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
    if (beamState) leavePending = false;
  }

  if (givingReward) {
    if (millis() - rewardStartTime >= SOLENOID_ON_TIME) close_solenoid();
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