// rig2intervalProcessing.ino
// Arduino sketch for interval task that streams logged events over serial
// to a Processing sketch, which writes them to a CSV file.
//
// Handshake protocol:
//  - Arduino powers up and waits for at least one byte from the host
//    (Processing) before starting the task.
//  - Processing opens the serial port and sends any single character
//    (e.g., 'S') to release the Arduino from the wait.
//
// Logged CSV format (sent over Serial):
//  timestamp_ms,event,value,state_label
//  - For non-state events (LED, reward, error, beam), state_label is empty.
//  - For state events, event is 'S', value is 0, and state_label is a tag.

#include <string.h>

// Pins
const int SOLENOID_PORT = 12;
const int BEAM_BREAK_PORT = 2;   // active low beam sensor
const int LED_PORT = 10;         // reward cue LED
const int ERROR_LED_PORT = 9;    // error indicator LED

// Timing (ms)
const int SOLENOID_ON_TIME = 30;           // duration solenoid stays open
const int SOLENOID_OFF_TIME = 300;        // spacing between clicks in a burst
const int SOLENOID_REPEATS = 1;           // additional clicks after first
const unsigned long INTERVAL_DURATION = 4UL * 1000UL;  // ITI duration
const unsigned long REWARD_WINDOW = 1UL * 1000UL;      // LED on window
const unsigned long ERROR_DURATION = 4UL * 1000UL;     // error timeout

// State flags
bool ledOn = false;
bool givingReward = false;
bool inBurst = false;
bool inIti = false;              // true when counting ITI
bool trialReady = false;         // LED on, waiting for beam break
bool errorLedOn = false;         // error indicator LED state
bool inErrorTimeout = false;     // true when in error timeout period

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

/**
 * log_line
 *  - tag: short event code (e.g., "L", "E", "R", "B")
 *  - value: integer value associated with the event
 *
 * Goal:
 *  Format and send a single CSV line for non-state events over Serial.
 *  CSV columns: timestamp_ms,event,value,state_label
 *  State label is left empty for these events.
 */
void log_line(const char *tag, int value) {
  unsigned long now = millis();
  sprintf(buffer, "%lu,%s,%d,", now, tag, value);
  Serial.println(buffer);
}

/**
 * write_state
 *  - label: descriptive string label for the current state
 *
 * Goal:
 *  Log a state transition line to the CSV stream.
 *  Event code is 'S', value is 0, and label is placed in the state_label field.
 */
void write_state(const char *label) {
  unsigned long now = millis();
  sprintf(buffer, "%lu,S,0,%s", now, label);
  Serial.println(buffer);
}

// Convenience wrappers for event logging
void write_led_on() { log_line("L", 1); }
void write_led_off() { log_line("L", 0); }
void write_error_on() { log_line("E", 1); }
void write_error_off() { log_line("E", 0); }
void write_reward_on() { log_line("R", 1); }
void write_reward_off() { log_line("R", 0); }
void write_beam(int state) { log_line("B", state); }

/**
 * turn_on_led
 *
 * Goal:
 *  Turn on the reward cue LED and log the change.
 */
void turn_on_led() {
  digitalWrite(LED_PORT, HIGH);
  ledOn = true;
  write_led_on();
}

/**
 * turn_off_led
 *
 * Goal:
 *  Turn off the reward cue LED and log the change.
 */
void turn_off_led() {
  digitalWrite(LED_PORT, LOW);
  ledOn = false;
  write_led_off();
}

/**
 * turn_on_error
 *
 * Goal:
 *  Turn on the error indicator LED and log the change.
 */
void turn_on_error() {
  digitalWrite(ERROR_LED_PORT, HIGH);
  errorLedOn = true;
  write_error_on();
}

/**
 * turn_off_error
 *
 * Goal:
 *  Turn off the error indicator LED and log the change.
 */
void turn_off_error() {
  digitalWrite(ERROR_LED_PORT, LOW);
  errorLedOn = false;
  write_error_off();
}

/**
 * open_solenoid
 *
 * Goal:
 *  Activate the solenoid to deliver reward and log solenoid-on event.
 */
void open_solenoid() {
  digitalWrite(SOLENOID_PORT, HIGH);
  givingReward = true;
  rewardStartTime = millis();
  write_reward_on();
}

/**
 * close_solenoid
 *
 * Goal:
 *  Deactivate the solenoid and log solenoid-off event.
 */
void close_solenoid() {
  digitalWrite(SOLENOID_PORT, LOW);
  givingReward = false;
  write_reward_off();
  lastPulseEndTime = millis();
}

/**
 * start_error_timeout
 *
 * Goal:
 *  Enter the error timeout period, managing LED states and logging the state.
 */
void start_error_timeout() {
  inErrorTimeout = true;
  errorStartTime = millis();
  turn_off_led(); // Turn off reward light
  if (!errorLedOn) {
    turn_on_error();
  }
  write_state("ERROR_TIMEOUT_START");
}

/**
 * start_iti
 *
 * Goal:
 *  Begin the inter-trial interval (ITI) and log the transition.
 */
void start_iti() {
  inIti = true;
  itiStartTime = millis();
  trialReady = false;
  turn_off_led(); // Ensure LED is off during ITI
  write_state("ITI_START");
}

/**
 * end_iti_and_arm_trial
 *
 * Goal:
 *  End ITI, arm the next trial by turning on the LED, and log the new state.
 */
void end_iti_and_arm_trial() {
  inIti = false;
  trialReady = true;
  turn_on_led();
  ledOnTime = millis();
  write_state("TRIAL_ARMED");
}

/**
 * wait_for_processing_start
 *
 * Goal:
 *  Block until Processing (or another host) opens the serial port and sends
 *  at least one byte, indicating it is ready to log data to disk.
 *
 * Behavior:
 *  - Assumes Serial.begin has already been called.
 *  - Does not send any output before the handshake completes, keeping the
 *    CSV stream clean.
 */
void wait_for_processing_start() {
  // For boards with native USB, wait for serial connection.
  while (!Serial) {
    ; // wait
  }

  // Wait until at least one byte arrives from the host.
  while (Serial.available() == 0) {
    ; // wait
  }

  // Clear any pending bytes from the handshake.
  while (Serial.available() > 0) {
    Serial.read();
  }
}

void setup() {
  pinMode(SOLENOID_PORT, OUTPUT);
  pinMode(LED_PORT, OUTPUT);
  pinMode(BEAM_BREAK_PORT, INPUT);
  pinMode(ERROR_LED_PORT, OUTPUT);

  digitalWrite(SOLENOID_PORT, LOW);
  digitalWrite(LED_PORT, LOW);
  digitalWrite(ERROR_LED_PORT, LOW);

  Serial.begin(9600);

  // Wait for Processing to signal that it is ready to log.
  wait_for_processing_start();

  // Send CSV header as the very first line.
  Serial.println("timestamp_ms,event,value,state_label");

  // Auto-start task and arm the first trial.
  write_state("AUTO_START");
  end_iti_and_arm_trial();
}

void loop() {
  // Read beam (active low -> invert)
  beamState = !digitalRead(BEAM_BREAK_PORT);
  if (beamState != lastBeamState) {
    write_beam(beamState);
    lastBeamState = beamState;

    // Beam broken (mouse enters port)
    if (beamState) {
      // If beam broken during ITI, start error timeout
      if (inIti) {
        inIti = false; // Exit ITI state
        start_error_timeout();
        write_state("BEAM_BROKEN_DURING_ITI");
      }
      // If beam broken during error timeout, restart error timeout
      else if (inErrorTimeout) {
        start_error_timeout(); // Restart the error timeout
        write_state("BEAM_BROKEN_DURING_ERROR_TIMEOUT");
      }
      // If beam broken during trial (LED on), trigger burst
      else if (trialReady) {
        trialReady = false;
        turn_off_led();  // Turn off LED when beam is broken
        inBurst = true;
        pulsesGivenInBurst = 0;
        write_state("BURST_START");
        // Immediately deliver the first pulse upon beam break
        open_solenoid();
      }
    }
    // Beam unbroken (mouse leaves port)
    else {
      // If beam leaves after burst or single pulse, begin ITI
      if (!givingReward && !inBurst && !inIti && !trialReady && !inErrorTimeout) {
        start_iti();
      }
      // If beam leaves during burst, wait for burst to complete before starting ITI
      else if (inBurst && !givingReward) {
        write_state("BEAM_LEAVE_DURING_BURST");
      }
    }
  }

  // Manage ITI
  if (inIti) {
    if (millis() - itiStartTime >= INTERVAL_DURATION) {
      end_iti_and_arm_trial(); // LED on only, no initial click
    }
  }

  // If trial is armed (LED on), wait for beam within REWARD_WINDOW
  if (trialReady) {
    if (millis() - ledOnTime >= REWARD_WINDOW) {
      // window expired, start error timeout
      trialReady = false;
      start_error_timeout();
      write_state("REWARD_WINDOW_EXPIRED");
    }
  }

  // Manage error timeout
  if (inErrorTimeout) {
    if (millis() - errorStartTime >= ERROR_DURATION) {
      // Error timeout complete, turn off error light and start new ITI
      inErrorTimeout = false;
      turn_off_error();
      write_state("ERROR_TIMEOUT_COMPLETE");
      start_iti();
    }
  }

  // Handle ongoing solenoid pulse
  if (givingReward) {
    if (millis() - rewardStartTime >= SOLENOID_ON_TIME) {
      close_solenoid();
    }
  }

  // Handle burst pulses after initial click
  if (inBurst && !givingReward) {
    unsigned long sinceLastPulse = millis() - lastPulseEndTime;
    if (pulsesGivenInBurst < SOLENOID_REPEATS) {
      if (sinceLastPulse >= SOLENOID_OFF_TIME) {
        open_solenoid();
        pulsesGivenInBurst++;
      }
    } else {
      // Burst complete
      inBurst = false;
      write_state("BURST_COMPLETE");
      // If beam is now not broken, begin ITI; otherwise wait until it leaves then ITI will start via beam transition branch
      if (!beamState && !inErrorTimeout) {
        start_iti();
      }
    }
  }
}

