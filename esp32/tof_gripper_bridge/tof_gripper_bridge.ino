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

// Pins: ESP32-C3 SuperMini.
#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9
#define SERVO_PIN 20

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
int64_t nextStatusUs = 0;

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

// ---- Connection diagnostics (serial monitor, 115200) -----------------

const char *wifiStatusName(wl_status_t status) {
  switch (status) {
    case WL_IDLE_STATUS: return "IDLE (starting)";
    case WL_NO_SSID_AVAIL: return "NO_SSID_AVAIL (network not seen)";
    case WL_SCAN_COMPLETED: return "SCAN_COMPLETED";
    case WL_CONNECTED: return "CONNECTED";
    case WL_CONNECT_FAILED: return "CONNECT_FAILED (wrong password?)";
    case WL_CONNECTION_LOST: return "CONNECTION_LOST";
    case WL_DISCONNECTED: return "DISCONNECTED";
    default: return "UNKNOWN";
  }
}

// ESP-IDF disconnect reasons that matter here; others print as a number.
const char *disconnectReason(uint8_t reason) {
  switch (reason) {
    case 2: return "AUTH_EXPIRE";
    case 15: return "4WAY_HANDSHAKE_TIMEOUT (wrong password)";
    case 201: return "NO_AP_FOUND (SSID not seen: wrong name, 5 GHz only, "
                     "or out of range)";
    case 202: return "AUTH_FAIL (wrong password or security mode)";
    case 203: return "ASSOC_FAIL";
    case 204: return "HANDSHAKE_TIMEOUT (wrong password)";
    case 205: return "CONNECTION_FAIL";
    default: return "";
  }
}

void onWifiEvent(arduino_event_id_t event, arduino_event_info_t info) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_STA_CONNECTED:
      Serial.printf("WiFi: joined '%s' (channel %u)\n", WIFI_SSID,
                    info.wifi_sta_connected.channel);
      break;
    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
      Serial.printf("WiFi: got IP %s, RSSI %d dBm\n",
                    WiFi.localIP().toString().c_str(), WiFi.RSSI());
      break;
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED: {
      uint8_t reason = info.wifi_sta_disconnected.reason;
      Serial.printf("WiFi: disconnected, reason %u %s\n", reason,
                    disconnectReason(reason));
      break;
    }
    default:
      break;
  }
}

// What the board can actually see: is the SSID there, how strong, and
// on which channel (the C3 only does 2.4 GHz, channels 1-13).
void scanForSsid() {
  Serial.println("WiFi: scanning...");
  int n = WiFi.scanNetworks();
  if (n < 0) {
    Serial.printf("WiFi: scan failed (%d)\n", n);
    return;
  }

  bool found = false;
  for (int i = 0; i < n; i++) {
    bool match = WiFi.SSID(i) == WIFI_SSID;
    found = found || match;
    Serial.printf("  %s '%s' ch %d, %d dBm%s\n", match ? ">>" : "  ",
                  WiFi.SSID(i).c_str(), WiFi.channel(i), WiFi.RSSI(i),
                  WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? ", open" : "");
  }
  Serial.printf("WiFi: %d networks, '%s' %s\n", n, WIFI_SSID,
                found ? "SEEN" : "NOT SEEN (2.4 GHz only; check the name)");
  WiFi.scanDelete();
}

// One line every 5 s: where the link is, whatever the other output says.
void printStatus() {
  if (nowUs() < nextStatusUs) return;
  nextStatusUs = nowUs() + 5000000;

  wl_status_t wifi = WiFi.status();
  Serial.printf("status: wifi=%s", wifiStatusName(wifi));
  if (wifi == WL_CONNECTED) {
    Serial.printf(" ip=%s rssi=%d", WiFi.localIP().toString().c_str(),
                  WiFi.RSSI());
  }
  Serial.printf(" mqtt=%s tof=%s readings=%lu\n",
                mqtt.connected() ? "connected" : "DOWN",
                tofOk ? "ok" : "DOWN", (unsigned long)seq);
}

void keepConnected() {
  if (WiFi.status() != WL_CONNECTED || mqtt.connected()) return;
  if (nowUs() < nextConnectUs) return;
  nextConnectUs = nowUs() + 1000000;

  // Last will: the broker announces "offline" if this board drops off.
  if (!mqtt.connect(MQTT_CLIENT, nullptr, nullptr, T_STATUS, 1, true,
                    "offline")) {
    IPAddress broker;
    bool resolved = WiFi.hostByName(MQTT_HOST, broker) == 1;
    Serial.printf("MQTT connect to %s:%d failed (state %d); host %s %s\n",
                  MQTT_HOST, MQTT_PORT, mqtt.state(),
                  resolved ? "resolves to" : "does NOT resolve",
                  resolved ? broker.toString().c_str() : "(use the Pi's IP)");
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
  // The C3's USB serial only appears once the monitor attaches.
  uint32_t waitStart = millis();
  while (!Serial && millis() - waitStart < 3000) delay(10);

  Serial.println();
  Serial.printf("laundrobros ESP32 bridge: WiFi '%s' -> MQTT %s:%d\n",
                WIFI_SSID, MQTT_HOST, MQTT_PORT);

  WiFi.onEvent(onWifiEvent);
  WiFi.mode(WIFI_STA);
  Serial.printf("WiFi: MAC %s\n", WiFi.macAddress().c_str());
  scanForSsid();
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
  printStatus();

  finishMove();
}
