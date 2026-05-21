// Revised_Interval_Timing_Code_Processing.ino
// Same interval timing task as Revised_Interval_Timing_Code.ino, but streams
// logged events over Serial so a Processing sketch can write the CSV to disk.
//
// Handshake: Arduino waits for at least one byte from the host (Processing)
// before starting the task. Processing sends 'S' (or any byte) after opening
// the serial port.
//
// CSV format (one line per event): timestamp_ms,event,value,
// Events: B (beam), REWARD, ERROR, SYNC (sync line on pin 5 / breakout board)
//   SYNC value 1 = line high (on), 2 = line low (off) — log on each transition
//   Three 100 ms sync pulses run on the first solenoid open of each reward (not repeats).

#include <string.h>

// Pins
const int SOLENOID_PORT = 12;
const int BEAM_BREAK_PORT = 2;   // active low
const int LED_PORT = 10;         // reward confirmation LED

// Timing (ms)
const int SOLENOID_ON_TIME = 20;   // solenoid stays open
const int SOLENOID_OFF_TIME = 300; // wait btwn pulses
const int SOLENOID_REPEATS = 1;    // additional pulses

const unsigned long INTERVAL_DURATION = 5000;   // min wait time after confirmed leave
const unsigned long MIN_LEAVE_TIME = 100;       // min leave time to confirm exit
const unsigned long LED_PULSE = 100;            // reward LED duration

const int BLINK_DURATION = 100; // ms per sync pulse on/off phase
const int NUM_SYNC_PULSES = 3;    // sync pulses per solenoid trigger
const int SYNC_LED_PORT = 3;
const int BREAKOUT_BOARD_PORT = 5;

char buffer[100];

// State
bool firstPoke = true;
bool givingReward = false;
bool inBurst = false;
bool leavePending = false;
bool timerArmed = false;

bool inSyncBurst = false;
bool inSyncPulse = false;
int syncPulsesCompleted = 0;
unsigned long syncPulseStartTime = 0;
unsigned long syncPulseEndTime = 0;

int beamState = 0;
int lastBeamState = 0;

unsigned long initialExitTime = 0;
unsigned long leaveConfirmStart = 0;
unsigned long leaveTime = 0;
unsigned long rewardStartTime = 0;
unsigned long lastSolenoidPulseEndTime = 0;

int solenoidPulsesInBurst = 0;

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
 *  - syncPinHigh: true for line high (on), false for low (off)
 *
 * Goal:
 *  Log sync line edges for video alignment (1 = on, 2 = off).
 */
void write_sync_edge(bool syncPinHigh) {
  log_line("SYNC", syncPinHigh ? 1 : 2);
}

/**
 * start_sync_burst
 *
 * Goal:
 *  Begin a 3-pulse sync train on SYNC_LED_PORT and BREAKOUT_BOARD_PORT.
 *  Called whenever the solenoid is triggered.
 */
void start_sync_burst() {
  inSyncBurst = true;
  inSyncPulse = true;
  syncPulsesCompleted = 0;
  write_sync_edge(true);
  digitalWrite(SYNC_LED_PORT, HIGH);
  digitalWrite(BREAKOUT_BOARD_PORT, HIGH);
  syncPulseStartTime = millis();
}

/**
 * update_sync_burst
 *  - now: current millis()
 *
 * Goal:
 *  Advance the non-blocking sync pulse state machine (on/off phases).
 */
void update_sync_burst(unsigned long now) {
  if (!inSyncBurst) return;

  if (inSyncPulse) {
    if (now - syncPulseStartTime >= (unsigned long)BLINK_DURATION) {
      write_sync_edge(false);
      digitalWrite(SYNC_LED_PORT, LOW);
      digitalWrite(BREAKOUT_BOARD_PORT, LOW);
      syncPulseEndTime = now;
      syncPulsesCompleted++;
      inSyncPulse = false;

      if (syncPulsesCompleted >= NUM_SYNC_PULSES) {
        inSyncBurst = false;
        syncPulsesCompleted = 0;
      }
    }
  } else {
    if (now - syncPulseEndTime >= (unsigned long)BLINK_DURATION) {
      write_sync_edge(true);
      digitalWrite(SYNC_LED_PORT, HIGH);
      digitalWrite(BREAKOUT_BOARD_PORT, HIGH);
      syncPulseStartTime = now;
      inSyncPulse = true;
    }
  }
}

/**
 * open_solenoid
 *  - triggerSync: if true, start sync burst on this open (first pulse only)
 *
 * Goal:
 *  Activate solenoid and record reward start time.
 */
void open_solenoid(bool triggerSync) {
  if (triggerSync) {
    start_sync_burst();
  }
  digitalWrite(SOLENOID_PORT, HIGH);
  givingReward = true;
  rewardStartTime = millis();
}

/**
 * close_solenoid
 *
 * Goal:
 *  Deactivate solenoid and record pulse end time for burst spacing.
 */
void close_solenoid() {
  digitalWrite(SOLENOID_PORT, LOW);
  givingReward = false;
  lastSolenoidPulseEndTime = millis();
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
  pinMode(SYNC_LED_PORT, OUTPUT);
  pinMode(BREAKOUT_BOARD_PORT, OUTPUT);

  digitalWrite(SOLENOID_PORT, LOW);
  digitalWrite(LED_PORT, LOW);
  digitalWrite(SYNC_LED_PORT, LOW);
  digitalWrite(BREAKOUT_BOARD_PORT, LOW);

  Serial.begin(9600);
  wait_for_processing_start();

  Serial.println("timestamp_ms,event,value,");
}

void loop() {
  unsigned long now = millis();
  update_sync_burst(now);

  beamState = (digitalRead(BEAM_BREAK_PORT) == LOW);

  if (beamState != lastBeamState) {
    write_beam(beamState);
    lastBeamState = beamState;

    if (beamState) {
      if (firstPoke) {
        firstPoke = false;
        write_reward();
        inBurst = true;
        solenoidPulsesInBurst = 0;
        open_solenoid(true);
      } else if (timerArmed) {
        if (now - leaveTime >= INTERVAL_DURATION) {
          write_reward();
          inBurst = true;
          solenoidPulsesInBurst = 0;
          open_solenoid(true);
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
    if (solenoidPulsesInBurst < SOLENOID_REPEATS) {
      if (millis() - lastSolenoidPulseEndTime >= SOLENOID_OFF_TIME) {
        open_solenoid(false);
        solenoidPulsesInBurst++;
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
