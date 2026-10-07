// ESP32 side of the Pi <-> ESP32 link: publishes VL53L0X readings and
// drives the gripper servo, over MQTT. The protocol, and how the Pi
// syncs this board's clock, are in the Pi's
// laundry_control/hardware/esp32_protocol.py; setup in ../README.md.
//
// Libraries (Arduino Library Manager):
//   PubSubClient (Nick O'Leary), VL53L0X (Pololu),
//   ESP32Servo (Kevin Harrington), ArduinoJson (Benoit Blanchon, v7)
//
// TIMING. Every reading carries esp_timer_get_time() (64-bit us since
// boot; never wraps, unlike micros()) at the MIDDLE of its measurement,
// and every ping is answered at once with the time it arrived. The Pi
// does the rest. So: keep loop() free of anything that blocks - no
// delay(), no blocking sensor reads - or pongs wait and readings are
// noticed late.

#include <WiFi.h>
#include <Wire.h>
#include <esp_random.h>
#include <esp_timer.h>

#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <PubSubClient.h>
#include <VL53L0X.h>

#include "config.h"

// Pins. Wemos/LOLIN D1 mini ESP32 (MH-ET LIVE MiniKit) defaults - TO
// CONFIRM against the rig's board and wiring.
// ESP32-C3 SuperMini: SDA 8, SCL 9, servo e.g. 4.
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define SERVO_PIN 18

static const char *T_RANGE = TOPIC_PREFIX "/tof/range";
static const char *T_PING = TOPIC_PREFIX "/time/ping";
static const char *T_PONG = TOPIC_PREFIX "/time/pong";
static const char *T_GRIPPER_CMD = TOPIC_PREFIX "/gripper/cmd";
static const char *T_GRIPPER_ACK = TOPIC_PREFIX "/gripper/ack";
static const char *T_STATUS = TOPIC_PREFIX "/esp32/status";

WiFiClient net;
PubSubClient mqtt(net);
VL53L0X tof;
Servo servo;

char bootId[9];          // Random per boot: tells the Pi the clock restarted.
uint32_t seq = 0;        // Reading number, for the Pi's lost-reading count.
bool tofOk = false;
int64_t nextTofInitUs = 0;
int64_t nextConnectUs = 0;
int64_t nextPollUs = 0;
int64_t lastReadingUs = 0;

// The move in progress, acknowledged once it has settled.
bool moving = false;
long moveId = 0;
bool moveHold = false;
int64_t moveDoneUs = 0;

static inline int64_t nowUs() { return esp_timer_get_time(); }

void ack(long id, bool ok, const char *msg) {
  char out[128];
  snprintf(out, sizeof out, "{\"id\":%ld,\"ok\":%s,\"msg\":\"%s\"}", id,
           ok ? "true" : "false", msg);
  mqtt.publish(T_GRIPPER_ACK, out);
}

void onPing(const byte *payload, unsigned int len, int64_t rxUs) {
  char id[16];
  len = len < sizeof id - 1 ? len : sizeof id - 1;
  memcpy(id, payload, len);
  id[len] = '\0';

  char out[96];
  snprintf(out, sizeof out, "{\"id\":%ld,\"t_us\":%lld,\"boot\":\"%s\"}",
           atol(id), (long long)rxUs, bootId);
  mqtt.publish(T_PONG, out);
}

void onGripperCmd(const byte *payload, unsigned int len) {
  JsonDocument doc;

  if (deserializeJson(doc, payload, len)) {
    ack(-1, false, "malformed command");
    return;
  }

  long id = doc["id"] | -1L;
  float angle = doc["angle"] | -1.0f;
  float settleSec = doc["settle"] | 1.5f;

  if (angle < SERVO_MIN_DEG || angle > SERVO_MAX_DEG) {
    ack(id, false, "angle out of range");
    return;
  }

  if (!servo.attached()) {
    servo.setPeriodHertz(50);
    servo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  }

  float fraction = (angle - SERVO_MIN_DEG) / (SERVO_MAX_DEG - SERVO_MIN_DEG);
  servo.writeMicroseconds(SERVO_MIN_US + lroundf(fraction * (SERVO_MAX_US - SERVO_MIN_US)));

  // A newer command replaces one still settling: the Pi has given up on
  // that one by then.
  moving = true;
  moveId = id;
  moveHold = doc["hold"] | false;
  moveDoneUs = nowUs() + (int64_t)(settleSec * 1e6f);
}

void onMessage(char *topic, byte *payload, unsigned int len) {
  int64_t rxUs = nowUs();  // First: the ping's arrival time.

  if (strcmp(topic, T_PING) == 0) {
    onPing(payload, len, rxUs);
  } else if (strcmp(topic, T_GRIPPER_CMD) == 0) {
    onGripperCmd(payload, len);
  }
}

void finishMove() {
  if (!moving || nowUs() < moveDoneUs) return;

  if (!moveHold) servo.detach();  // Stop the pulses, as the Pi's servo.py does.

  moving = false;
  ack(moveId, true, moveHold ? "done, holding" : "done");
}

void initTof() {
  tof.setTimeout(100);

  if (!tof.init()) {
    Serial.println("VL53L0X not found; retrying in 1 s");
    nextTofInitUs = nowUs() + 1000000;
    return;
  }

  tof.setMeasurementTimingBudget(TOF_TIMING_BUDGET_US);
  tof.startContinuous(0);  // Back to back: one reading per budget.
  tofOk = true;
  lastReadingUs = nowUs();
  Serial.println("VL53L0X ready");
}

void pollTof() {
  if (!tofOk) {
    if (nowUs() >= nextTofInitUs) initTof();
    return;
  }

  // Poll the data-ready flag instead of blocking in the read, every
  // 0.5 ms: often enough to notice a reading promptly.
  if (nowUs() < nextPollUs) return;
  nextPollUs = nowUs() + 500;

  if ((tof.readReg(VL53L0X::RESULT_INTERRUPT_STATUS) & 0x07) == 0) {
    // Nothing for 10 budgets: a loose wire or a hung sensor. Re-init.
    if (nowUs() - lastReadingUs > 10 * TOF_TIMING_BUDGET_US) tofOk = false;
    return;
  }

  int64_t readyUs = nowUs();
  lastReadingUs = readyUs;
  uint16_t mm = tof.readRangeContinuousMillimeters();  // Ready: no wait.

  if (tof.timeoutOccurred()) {
    tofOk = false;
    return;
  }

  // Out-of-range readings (~8190 mm) go out too; the Pi drops them
  // against its max range, as it does for its own sensor.
  char out[128];
  snprintf(out, sizeof out,
           "{\"seq\":%lu,\"t_us\":%lld,\"mm\":%u,\"boot\":\"%s\"}",
           (unsigned long)++seq,
           (long long)(readyUs - TOF_TIMING_BUDGET_US / 2), mm, bootId);
  mqtt.publish(T_RANGE, out);
}

void keepConnected() {
  if (WiFi.status() != WL_CONNECTED || mqtt.connected()) return;
  if (nowUs() < nextConnectUs) return;
  nextConnectUs = nowUs() + 1000000;

  // Last will: the broker announces "offline" if this board drops off.
  if (!mqtt.connect(MQTT_CLIENT, nullptr, nullptr, T_STATUS, 1, true,
                    "offline")) {
    Serial.printf("MQTT connect failed (%d)\n", mqtt.state());
    return;
  }

  net.setNoDelay(true);  // No Nagle: send each small packet at once.
  mqtt.subscribe(T_PING, 0);
  mqtt.subscribe(T_GRIPPER_CMD, 1);
  mqtt.publish(T_STATUS, "online", true);
  Serial.println("MQTT connected");
}

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_STA);
  // Power save holds packets for up to a beacon interval (~100 ms):
  // it would wreck the clock sync and the reading latency.
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setAutoReconnect(true);

  snprintf(bootId, sizeof bootId, "%08lx", (unsigned long)esp_random());

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
  mqtt.setKeepAlive(5);  // "offline" within ~8 s of a power cut.
  mqtt.setBufferSize(256);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);
  initTof();
}

void loop() {
  keepConnected();
  mqtt.loop();  // Delivers pings and commands (onMessage).

  pollTof();  // Publishing just fails while disconnected.

  finishMove();
}
