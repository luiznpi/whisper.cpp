#ifndef WHISPER_SERVICE_H
#define WHISPER_SERVICE_H

#include <stdint.h>  // For fixed-width integer types
#include <stdbool.h> // For boolean types

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

        const char *language; // Language code (e.g., "en")
        const char *model;    // Path to the model file
    } WhisperParams;

    // Callback function type for returning transcription results
    typedef void (*TranscriptionCallbackC)(const char *transcription);

    // Function to create a new WhisperService instance
    WhisperServiceHandle whisper_service_create(const WhisperParams *params);

    // Function to destroy the WhisperService instance
    void whisper_service_destroy(WhisperServiceHandle handle);

    // Function to initialize the WhisperService
    // Returns 1 on success, 0 on failure
    int whisper_service_initialize(WhisperServiceHandle handle);

    // Function to process an audio chunk
    // audio_data: Pointer to float array containing audio samples
    // length: Number of samples in the audio_data array
    void whisper_service_process_audio_chunk(WhisperServiceHandle handle, const float *audio_data, int length);

    // Function to stop the transcription service
    void whisper_service_stop(WhisperServiceHandle handle);

    // Function to set the transcription callback
    // The callback should be a function that takes a const char* (transcription result)
    void whisper_service_set_callback(WhisperServiceHandle handle, TranscriptionCallbackC callback);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <string>
#include <vector>
#include <functional>

// C++ class declaration remains private and is not exposed via FFI
class WhisperService
{
public:
    // Constructor and Destructor
    WhisperService(const WhisperParams &params);
    ~WhisperService();

    // Initialize the Whisper context
    bool initialize();

    // Process an audio chunk
    void processAudioChunk(const std::vector<float> &audio_data);

    // Stop the transcription service
    void stop();

    // Set the transcription callback
    void setCallback(std::function<void(const std::string &)> callback);

private:
    struct whisper_context *ctx;
    WhisperParams params;
    std::vector<float> pcmf32_old;                     // Buffer for retaining previous audio
    std::function<void(const std::string &)> callback; // User-defined callback for results
};

#endif // __cplusplus

#endif // WHISPER_SERVICE_H