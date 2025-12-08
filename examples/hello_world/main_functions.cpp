#include "constants.h"
#include "hello_world_int8_model_data.h"
#include "mlp_int8_model_data.h"
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

constexpr int kTensorArenaSize = 100 * 1024;  // 100 KB
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// =============================================================
//                          SETUP()
// =============================================================
void setup() {
  stdio_init_all();

  tflite::InitializeTarget();

  model = tflite::GetModel(g_mlp_int8_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Bad model version");
    return;
  }

  static tflite::MicroMutableOpResolver<3> resolver;
resolver.AddFullyConnected();
resolver.AddRelu();
resolver.AddSoftmax();


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
  
  
  // =====================================
  // 1. ΕΤΟΙΜΑΖΩ ΤΑ 59 FEATURES (float)
  // =====================================
 float features[59] = {
      0.12, -0.33, 1.22, -0.88, 0.44,   // 5
      -1.11, 0.95, -0.42, 0.77, 0.05,   // 10
      -0.66, 1.44, -0.12, 0.33, -1.22,  // 15
      0.21, 0.54, -0.74, 0.99, -0.25,   // 20
      1.31, -0.17, 0.08, -0.51, 0.63,   // 25
      0.27, -1.44, 0.11, 1.05, -0.96,   // 30
      0.32, -0.55, 0.66, -0.83, 1.17,   // 35
      0.41, 0.02, -0.14, 0.53, -1.09,   // 40
      1.26, -0.72, 0.19, 0.07, -0.31,   // 45
      0.58, -0.48, 0.24, 1.11, -0.67,   // 50
      0.10, 0.39, -0.52, 0.74, -1.33,   // 55
      0.89, -0.28, 0.61, 0.03           // 59
  };

  // =====================================
  // 2. QUANTIZE → INT8
  // =====================================
  for (int i = 0; i < 59; i++) {
      float x = features[i];
      int32_t q = (int32_t)(x / input->params.scale + input->params.zero_point);

      if (q < -128) q = -128;
      if (q > 127) q = 127;

      input->data.int8[i] = (int8_t)q;
  }

  // =====================================
  // 3. DWT + Invoke
  // =====================================
  dwt_enable_all();
  uint64_t t0 = time_us_64();

  TfLiteStatus status = interpreter->Invoke();

  uint64_t t1 = time_us_64();

  if (status != kTfLiteOk) {
    printf("Invoke failed!\n");
    return;
  }

  // =====================================
  // 4. READ OUTPUT (2 CLASS logits)
  // =====================================
  int8_t y0_q = output->data.int8[0];
  int8_t y1_q = output->data.int8[1];

  float y0 = (y0_q - output->params.zero_point) * output->params.scale;
  float y1 = (y1_q - output->params.zero_point) * output->params.scale;

  int predicted_class = (y0 > y1 ? 0 : 1);

  // =====================================
  // 5. READ DWT COUNTERS
  // =====================================
  uint32_t cycles = DWT_CYCCNT;
  uint32_t cpi    = DWT_CPICNT;
  uint32_t lsu    = DWT_LSUCNT;
  uint32_t fold   = DWT_FOLDCNT;

  // =====================================
  // 6. PRINT
  // =====================================
  printf("Pred=%d  y0=%.3f  y1=%.3f  cycles=%u  lsu=%u  cpi=%u  fold=%u  time_us=%llu\n",
         predicted_class, y0, y1, cycles, lsu, cpi, fold, (t1 - t0));

  sleep_ms(300);  // λίγο delay για καθαρό output
}
