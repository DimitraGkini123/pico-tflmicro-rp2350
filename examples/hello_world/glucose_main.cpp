#include "pico/stdlib.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "mlp_glucose_int8_data.h"   // <<=== ΤΟ ΜΟΝΤΕΛΟ ΣΟΥ

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
    DWT_CYCCNT = DWT_CPICNT = DWT_EXCCNT = DWT_SLEEPCNT = DWT_LSUCNT = DWT_FOLDCNT = 0;

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

constexpr int kTensorArenaSize = 80 * 1024;  // αρκετό για μικρό MLP
uint8_t tensor_arena[kTensorArenaSize];
}

// =============================================================
//                          SETUP()
// =============================================================
void setup() {
    stdio_init_all();
    tflite::InitializeTarget();

    // Load model
    model = tflite::GetModel(g_mlp_glucose_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model version mismatch!");
        return;
    }

    // Only FC and ReLU needed for MLP
    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();

    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors failed!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    MicroPrintf("Setup complete.");
}

// =============================================================
//                           LOOP()
// =============================================================
void loop() {

    // =====================================
    // 1. 25 FEATURES (float)
    // =====================================
    float features[25] = {
        0.12, -0.33, 1.22, -0.88, 0.44,
        -1.11, 0.95, -0.42, 0.77, 0.05,
        -0.66, 1.44, -0.12, 0.33, -1.22,
        0.21, 0.54, -0.74, 0.99, -0.25,
        1.31, -0.17, 0.08, -0.51, 0.63
    };

    // =====================================
    // 2. QUANTIZE TO INT8
    // =====================================
    for (int i = 0; i < 25; i++) {
        float x = features[i];
        int32_t q = (int32_t)(x / input->params.scale + input->params.zero_point);

        if (q < -128) q = -128;
        if (q > 127) q = 127;

        input->data.int8[i] = (int8_t)q;
    }

    // =====================================
    // 3. DWT START + Invoke
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
    // 4. READ REGRESSION OUTPUT (1 value)
    // =====================================
    int8_t q_out = output->data.int8[0];

    float glucose = (q_out - output->params.zero_point) * output->params.scale;

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
    printf("Glucose = %.2f mg/dL   cycles=%u  cpi=%u  lsu=%u  fold=%u  time=%llu us\n",
           glucose, cycles, cpi, lsu, fold, (t1 - t0));

    sleep_ms(500);
}
