#include "DHT.h"

#define DHTPIN 4       // Pin connected to DHT22
#define DHTTYPE DHT22  // Define sensor type

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();
  Serial.println("Time,Humidity,Temperature");  // CSV header
}

void loop() {
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  if (!isnan(humidity) && !isnan(temperature)) {
    unsigned long time = millis();  // Elapsed time in milliseconds
    Serial.print(time);
    Serial.print(",");
    Serial.print(humidity);
    Serial.print(",");
    Serial.println(temperature);
  }

  delay(2000);                // Sampling rate
}