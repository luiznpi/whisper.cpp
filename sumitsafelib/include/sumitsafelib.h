#ifndef WHISPER_SERVICE_H
#define WHISPER_SERVICE_H

#ifdef _WIN32
#ifdef SUMITSAFE_EXPORTS
#define WHISPER_API __declspec(dllexport)
#else
#define WHISPER_API __declspec(dllimport)
#endif
#else
#define WHISPER_API __attribute__((visibility("default")))
#endif

#include <stdint.h>
#include <stdbool.h>
#include <chrono>

#ifdef __cplusplus
extern "C"
{
#endif

    // Opaque handle to the WhisperService instance
    typedef void *WhisperServiceHandle;

    // Configuration structure for WhisperService (C-compatible)
    typedef struct WhisperParams
    {
        int32_t n_threads;
        int32_t step_ms;   // Step size in ms
        int32_t length_ms; // Max length in ms
        int32_t keep_ms;   // Context to retain in ms
        int32_t max_tokens;
        int32_t audio_ctx;

        float vad_thold;  // VAD threshold
        float freq_thold; // Frequency threshold for VAD

        bool translate;     // Translate to English
        bool no_fallback;   // Disable temperature fallback
        bool print_special; // Include special tokens in output
        bool no_context;    // Clear context between steps
        bool no_timestamps; // Omit timestamps
        bool use_gpu;       // Use GPU
        bool flash_attn;    // Enable Flash Attention
        bool verbose;       // enable debug prints

        const char *language; // Language code (e.g., "en")
        const char *model;    // Path to the model file
    } WhisperParams;

    // Callback function type for returning transcription results
    typedef void (*TranscriptionCallbackC)(const char *transcription);

    // Function to create a new WhisperService instance
    WHISPER_API WhisperServiceHandle whisper_service_create(const WhisperParams *params);

    // Function to destroy the WhisperService instance
    WHISPER_API void whisper_service_destroy(WhisperServiceHandle handle);

    // Function to initialize the WhisperService
    // Returns 1 on success, 0 on failure
    WHISPER_API int whisper_service_initialize(WhisperServiceHandle handle);

    // Function to process an audio chunk
    // audio_data: Pointer to float array containing audio samples
    // length: Number of samples in the audio_data array
    WHISPER_API void whisper_service_process_audio_chunk(WhisperServiceHandle handle, const float *audio_data, int length);

    // length: Number of samples in the audio_data array
    WHISPER_API void whisper_service_process_audio_stream(WhisperServiceHandle handle, const float *audio_data, int length, bool flushCmd, int minSilentSpeakingMs, int maxSilenceMs);

    // Function to process an audio chunk
    // audio_data: Pointer to float array containing audio samples
    // Function to stop the transcription service
    WHISPER_API void whisper_service_stop(WhisperServiceHandle handle);

    // Function to set the transcription callback
    // The callback should be a function that takes a const char* (transcription result)
    WHISPER_API void whisper_service_set_callback(WhisperServiceHandle handle, TranscriptionCallbackC callback);

#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
// Include C++ headers first
#include <vector>
#include <string>
#include <functional>

#endif

#ifdef __cplusplus

#include <string>
#include <vector>
#include <functional>

// C++ class declaration remains private and is not exposed via FFI
void high_pass_filter(std::vector<float> &data, float cutoff, float sample_rate);
bool vad_deepseek(const std::vector<float> &pcmf32, int sample_rate, int last_ms,
                  float vad_thold, float freq_thold, bool verbose);

// Class declaration (export only if building the library)
#ifdef SUMITSAFE_EXPORTS
#define WHISPER_CPP_API WHISPER_API
#else
#define WHISPER_CPP_API
#endif
class WHISPER_CPP_API WhisperService
{
public:
    // Constructor and Destructor
    WhisperService(const WhisperParams &params);
    ~WhisperService();

    // Initialize the Whisper context
    bool initialize();

    // Process an audio chunk
    void processAudioChunk(const std::vector<float> &audio_data);

    // Process an audio stream
    void processAudioStream(const std::vector<float> &audio_data, const bool flushCmd, const int minSilenceSpeakingMs, const int maxSilenceMs);

    // Stop the transcription service
    void stop();

    // Set the transcription callback
    void setCallback(std::function<void(const std::string &)> callback);

private:
    struct whisper_context *ctx;
    WhisperParams params;
    std::vector<float> pcmf32_old;                 // Buffer for retaining previous audio
    std::vector<float> pcmf32_audio_mem;           // Buffer for retaining previous audio to detect levels
    std::vector<float> pcmf32_audio_to_transcribe; // Buffer to accumulate audio to be transcribed

    bool is_speaking = false;
    std::function<void(const std::string &)> callback; // User-defined callback for results
    std::chrono::time_point<std::chrono::high_resolution_clock> last_voice_time;
};

#endif // __cplusplus

#endif // WHISPER_SERVICE_H