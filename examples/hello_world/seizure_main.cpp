#include "constants.h"
#include "seizure_mlp_int8_model_data.h" 
#include "main_functions.h"
#include "output_handler.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "pico/stdlib.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h> // Για τη συνάρτηση isnan
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

constexpr float kFeatureMeans[16] = {
    0.01648546f, 0.01030279f, 0.00719316f, 0.00549122f, 0.02428695f, 0.00549539f,
    0.00641472f, 0.01100811f, 0.00771037f, -0.01160149f, 0.00851172f, -0.01132491f,
    0.00761950f, 0.00646390f, 0.00804614f, 0.00533492f
};
constexpr float kFeatureStdDevs[16] = {
    1.01581902f, 1.01340002f, 1.01925573f, 1.00305759f, 0.98655081f, 1.00340754f,
    1.00515812f, 1.01253829f, 1.00576271f, 0.99906173f, 0.99703052f, 0.98423961f,
    1.00515800f, 1.00392036f, 1.00701311f, 1.00012826f
};

// =============================================================
//                        TFLM GLOBALS
// =============================================================
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 256 * 1024; 
uint8_t tensor_arena[kTensorArenaSize];
}

// =============================================================
//                          SETUP()
// =============================================================
void setup() {
    stdio_init_all();
    tflite::InitializeTarget();

    model = tflite::GetModel(g_seizure_mlp_int8_model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Bad model version!");
        return;
    }

   static tflite::MicroMutableOpResolver<3> resolver; // Αλλάζουμε το μέγεθος σε 3
        resolver.AddFullyConnected();
        resolver.AddRelu();
        resolver.AddLogistic(); // Το TFLM χρησιμοποιεί Logistic αντί για Sigmoid

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    interpreter = &static_interpreter;

printf("Before AllocateTensors\n");

if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors failed\n");
    return;
}

printf("After AllocateTensors\n");
    input = interpreter->input(0);
    output = interpreter->output(0);

printf("Input Quantization: Scale = %f, Zero Point = %d\n", 
           input->params.scale, input->params.zero_point);
}

// =============================================================
//              *** 16 FEATURE SEIZURE CLASSIFIER ***
// =============================================================
void loop() {

    // Fake input (16-D). Μπορείς να βάλεις ό,τι θέλεις.
// TEST A: Καθαρό Normal (Όλες οι τιμές ίσες με τη μέση τιμή)
// TEST B: Σπασμός (Υψηλή διακύμανση ACC & EMG)
float raw_features[16] = {
    // 1-2. ACC meanX, stdX
    4.0797f, 4.0640f, 
    // 3-4. ACC meanY, stdY
    4.0858f, 4.0277f, 
    // 5-6. ACC meanZ, stdZ
    3.9691f, 4.0171f, 
    // 7-8. SMA, peak ACC
    4.0275f, 4.0610f, 
    // 9. RMS EMG
    4.0298f, 
    // 10-12. ZC, WA, SSC
    3.9846f, 3.9965f, 3.9256f, 
    // 13-16. EMG mean abs, WL, variance, peak amplitude
    4.0302f, 4.0221f, 4.0361f, 4.0055f 
};


    // ===============================
    // QUANTIZE INPUT
    // ===============================
    float standardized_features[16];
    for (int i = 0; i < 16; i++) {
        // Standardized = (Raw - Mean) / StdDev
        float raw_x = raw_features[i];
        float mean = kFeatureMeans[i];
        float stddev = kFeatureStdDevs[i];
        
        // Prevent division by zero if stddev is 0 (occurs if a feature is constant)
        if (stddev < 1e-6f) {
             standardized_features[i] = raw_x - mean; // Apply only mean subtraction
        } else {
             standardized_features[i] = (raw_x - mean) / stddev;
        }

        // Safety check for NaN/Inf which can crash the interpreter
        if (isnan(standardized_features[i]) || isinf(standardized_features[i])) {
            standardized_features[i] = 0.0f;
        }
    }
// ===============================
    // QUANTIZE INPUT (ASYMPHALES)
    // ===============================
    for (int i = 0; i < 16; i++) {
        float x = standardized_features[i];
        
        // Quantization formula: q = round(x / scale) + zero_point
        int32_t q = (int32_t)(x / input->params.scale + input->params.zero_point);

        // 💡 ΑΣΦΑΛΗΣ ΜΕΤΑΤΡΟΠΗ/CLAMPING ΣΕ int8_t
        int8_t q_final;
        
        // 1. Clamping (Βεβαιωνόμαστε ότι η τιμή είναι εντός [-128, 127])
        if (q < -128) {
            q_final = -128;
        } else if (q > 127) {
            q_final = 127;
        } else {
            q_final = (int8_t)q; // Απλό cast εφόσον είναι εντός ορίων
        }

        // Εκτύπωση για αποσφαλμάτωση (μόνο για το πρώτο feature)
        if (i == 0) {
            printf("Feature 1 Calc: q_raw=%d -> q_final=%d\n", q, q_final);
        }

        // Εκχώρηση της τελικής clamped τιμής
        input->data.int8[i] = q_final;
        if (i == 0) {
            printf("Pointer Check: Final Q_Final=%d, Read Back Immediately=%d\n", 
                   q_final, input->data.int8[i]);
        }
    }


    // =======================================
    // 3. RUN INFERENCE + COUNTERS
    // =======================================
    dwt_enable_all();
    uint64_t t0 = time_us_64();
    //TfLiteStatus s = interpreter->Invoke();
    uint64_t t1 = time_us_64();

    //if (s != kTfLiteOk) {
     //   printf("Invoke failed! Check TFLiteMicro error logs.\n");
      //  return;
    //}

    // =======================================
    // 4. DE-QUANTIZE OUTPUT
    // =======================================
int8_t q = output->data.int8[0];  // ONLY ONE OUTPUT

float prob = (q - output->params.zero_point) * output->params.scale;

// Clamp to [0,1] just for safety
if (prob < 0) prob = 0;
if (prob > 1) prob = 1;

int pred = (prob > 0.5f ? 1 : 0);  // 1 = SEIZURE


   printf("\n=== SEIZURE MLP ===\n");
    // Εκτυπώνουμε το [0] όπως πριν, αλλά προσέχουμε:
    printf("Raw Feature 1: %.2f -> Standardized: %.2f -> Quantized: %d\n", 
           raw_features[0], standardized_features[0], (int)input->data.int8[0]); 
    // ΝΕΑ ΓΡΑΜΜΗ: Δείτε τι υπάρχει στη δεύτερη θέση
    printf("Quantized Feature 2 Check: %d\n", (int)input->data.int8[1]); 
    printf("Prediction: %s\n", pred ? "SEIZURE" : "NORMAL");

    printf("cycles=%u  cpi=%u  lsu=%u  fold=%u  time=%llu us\n",
           DWT_CYCCNT, DWT_CPICNT, DWT_LSUCNT, DWT_FOLDCNT, (t1 - t0));

    sleep_ms(300);
}
