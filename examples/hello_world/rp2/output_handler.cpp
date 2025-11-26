/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "output_handler.h"

#include "pico/stdlib.h"
#include "constants.h"
#include "tensorflow/lite/micro/micro_log.h"  // For MicroPrintf

void HandleOutput(float x_value, float y_value) {
  // Initialize stdio once (USB/UART)
  static bool is_initialized = false;
  if (!is_initialized) {
    stdio_init_all();
    is_initialized = true;
  }

  // Same "brightness" computation as πριν, απλά δεν το στέλνουμε σε LED
  int led_brightness = static_cast<int>(127.5f * (y_value + 1));

  // Print value so το βλέπεις στο serial monitor
  MicroPrintf("%d\n", led_brightness);

  // Μικρό delay για να είναι πιο “ορατή” η εξέλιξη της εξόδου
  sleep_ms(10);
}
