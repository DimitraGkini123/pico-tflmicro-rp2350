#include "constants.h"
#include "hello_world_int8_model_data.h"
#include "main_functions.h"
#include "output_handler.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "pico/stdlib.h"

// =============================================================
//                     DWT COUNTERS
// =============================================================
#define DEMCR                (*(volatile uint32_t *)0xE000EDFC)
#define DEMCR_TRCENA         (1u << 24)

#define DWT_BASE             (0xE0001000u)
#define DWT_CTRL             (*(volatile uint32_t *)(DWT_BASE + 0x000))
#define DWT_CYCCNT           (*(volatile uint32_t *)(DWT_BASE + 0x004))
#define DWT_CPICNT           (*(volatile uint32_t *)(DWT_BASE + 0x008))
#define DWT_EXCCNT           (*(volatile uint32_t *)(DWT_BASE + 0x00C))
#define DWT_SLEEPCNT         (*(volatile uint32_t *)(DWT_BASE + 0x010))
#define DWT_LSUCNT           (*(volatile uint32_t *)(DWT_BASE + 0x014))
#define DWT_FOLDCNT          (*(volatile uint32_t *)(DWT_BASE + 0x018))
#define DWT_LAR              (*(volatile uint32_t *)(DWT_BASE + 0xFB0))

#define DWT_CTRL_CYCCNTENA   (1u << 0)
#define DWT_CTRL_CPIEVTENA   (1u << 17)
#define DWT_CTRL_EXCEVTENA   (1u << 18)
#define DWT_CTRL_SLEEPEVTENA (1u << 19)
#define DWT_CTRL_LSUEVTENA   (1u << 20)
#define DWT_CTRL_FOLDEVTENA  (1u << 21)

static inline void dwt_enable_all() {
    DEMCR |= DEMCR_TRCENA;
    DWT_LAR = 0xC5ACCE55;

    DWT_CYCCNT   = 0;
    DWT_CPICNT   = 0;
    DWT_EXCCNT   = 0;
    DWT_SLEEPCNT = 0;
    DWT_LSUCNT   = 0;
    DWT_FOLDCNT  = 0;

    DWT_CTRL |= DWT_CTRL_CYCCNTENA |
                DWT_CTRL_CPIEVTENA |
                DWT_CTRL_EXCEVTENA |
                DWT_CTRL_SLEEPEVTENA |
                DWT_CTRL_LSUEVTENA |
                DWT_CTRL_FOLDEVTENA;
}

// =============================================================
//                        TFLM GLOBALS
// =============================================================
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// =============================================================
//                          SETUP()
// =============================================================
void setup() {
  stdio_init_all();

  tflite::InitializeTarget();

  model = tflite::GetModel(g_hello_world_int8_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Bad model version");
    return;
  }

  static tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  inference_count = 0;
}

// =============================================================
//                           LOOP()
// =============================================================
void loop() {
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // -----------------------------
  // QUANTIZE INPUT → INT8
  // -----------------------------
  int8_t x_q = static_cast<int8_t>(x / input->params.scale + input->params.zero_point);
  input->data.int8[0] = x_q;

  dwt_enable_all();
  uint64_t t0 = time_us_64();

  TfLiteStatus status = interpreter->Invoke();

  uint64_t t1 = time_us_64();

  if (status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // -----------------------------
  // DEQUANTIZE OUTPUT
  // -----------------------------
  int8_t y_q = output->data.int8[0];
  float y = (y_q - output->params.zero_point) * output->params.scale;

  uint32_t cycles = DWT_CYCCNT;
  uint32_t cpi    = DWT_CPICNT;
  uint32_t lsu    = DWT_LSUCNT;
  uint32_t fold   = DWT_FOLDCNT;

  printf("x=%.3f  y=%.3f  cycles=%u  lsu=%u  cpi=%u  fold=%u  time_us=%llu\n",
         x, y, cycles, lsu, cpi, fold, (t1 - t0));

  HandleOutput(x, y);

  inference_count++;
  if (inference_count >= kInferencesPerCycle)
      inference_count = 0;

  sleep_ms(30);
}
