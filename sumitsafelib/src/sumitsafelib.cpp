#include "sumitsafelib.h"
#include "whisper.h"
#include <stdexcept>
#include <iostream>
#include <cstring>

#ifdef __cplusplus
#define M_PI 3.14159265358979323846 // pi

void high_pass_filter(std::vector<float> &data, float cutoff, float sample_rate)
{
    if (cutoff <= 0 || cutoff >= sample_rate / 2)
    {
        // Invalid cutoff frequency
        return;
    }

    const float rc = 1.0f / (2.0f * M_PI * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = rc / (rc + dt);

    float prev_input = data[0];
    float prev_output = 0.0f;

    // First sample remains unchanged (or could be set to 0)
    for (size_t i = 1; i < data.size(); i++)
    {
        float output = alpha * (prev_output + data[i] - prev_input);
        prev_input = data[i];
        prev_output = output;
        data[i] = output;
    }
}
static float noise_floor = 0.0f;
const float alpha = 0.01f; // Smoothing factor

// Const-correct version that doesn't modify input
bool vad_deepseek(const std::vector<float> &pcmf32, int sample_rate, int last_ms,
                  float vad_thold, float freq_thold, bool verbose)
{
    const int n_samples = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    // Handle edge cases
    if (n_samples_last >= n_samples || n_samples == 0)
    {
        if (verbose)
            fprintf(stderr, "%d n_samples: %d, n_samples_last: \n", n_samples, n_samples_last);
        return false; // Not enough samples for reliable detection
    }

    // Create a filtered copy for processing
    std::vector<float> filtered(pcmf32);
    if (freq_thold > 0.0f)
    {
        high_pass_filter(filtered, freq_thold, sample_rate);
    }

    // Calculate energies for previous part and last window
    float energy_all = 0.0f;
    float energy_last = 0.0f;

    int n_samples_all = n_samples - n_samples_last;
    // Replace energy accumulation with squared values
    for (int i = 0; i < n_samples; i++)
    {
        float sample = filtered[i];
        if (i < n_samples_all)
        {
            energy_all += sample * sample;
        }
        else
        {
            energy_last += sample * sample;
        }
    }
    energy_all = (n_samples_all > 0) ? sqrtf(energy_all / n_samples_all) : 0.0f;
    energy_last = sqrtf(energy_last / n_samples_last);

    // Detect speech using relative threshold and absolute minimum
    bool is_silent = energy_last < fmaxf(energy_all, noise_floor) / vad_thold;
    if (is_silent)
    {
        // Update noise floor during silence
        noise_floor = alpha * energy_all + (1 - alpha) * noise_floor;
        noise_floor = fmaxf(noise_floor, 0.1);
        fprintf(stderr, " noise floor: %f\n", noise_floor);
    }
    if (verbose)
    {
        fprintf(stderr, " n_samples: %d, n_samples_last: %d VAD: energy_last=%.3f energy_all=%.3f → %s\n", n_samples, n_samples_last,
                energy_last, energy_all,
                is_silent ? "SILENCE" : "SPEECH");
    }

    return is_silent;
}

// C++ class implementation
WhisperService::WhisperService(const WhisperParams &params)
    : params(params), ctx(nullptr), callback(nullptr)
{
    last_voice_time = std::chrono::high_resolution_clock::now();
}

WhisperService::~WhisperService()
{
    stop();
}

bool WhisperService::initialize()
{
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    ctx = whisper_init_from_file_with_params(params.model, cparams);
    if (!ctx)
    {
        std::cerr << "Failed to initialize Whisper context!" << std::endl;
        return false;
    }

    return true;
}

void WhisperService::processAudioChunk(const std::vector<float> &audio_data)
{
    if (!ctx)
    {
        throw std::runtime_error("Whisper context not initialized!");
    }

    // Merge old audio and new audio data
    std::vector<float> pcmf32;
    const int n_samples_keep = (params.keep_ms * WHISPER_SAMPLE_RATE) / 1000;
    const int n_samples_take = std::min(static_cast<int>(pcmf32_old.size()), n_samples_keep);

    if (n_samples_take > 0)
    {
        pcmf32.reserve(n_samples_take + audio_data.size());
        pcmf32.insert(pcmf32.end(), pcmf32_old.end() - n_samples_take, pcmf32_old.end());
    }
    pcmf32.insert(pcmf32.end(), audio_data.begin(), audio_data.end());

    // Retain part of the current audio for the next iteration
    if (audio_data.size() > n_samples_keep)
    {
        pcmf32_old.assign(audio_data.end() - n_samples_keep, audio_data.end());
    }
    else
    {
        pcmf32_old = pcmf32;
    }

    // Prepare transcription parameters
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = params.language;
    wparams.n_threads = params.n_threads;
    wparams.translate = params.translate;
    wparams.print_special = params.print_special;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.single_segment = true;

    // Run Whisper transcription
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0)
    {
        throw std::runtime_error("Whisper transcription failed!");
    }

    // Collect results
    std::string result;
    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i)
    {
        const char *segment_text = whisper_full_get_segment_text(ctx, i);
        if (segment_text != nullptr)
        {
            result += segment_text;
        }
    }

    // Call the user-defined callback
    if (callback)
    {
        callback(result);
    }
}

void WhisperService::processAudioStream(const std::vector<float> &audio_data, bool flushCmd, const int minSilenceSpeakingMs, const int maxSilenceMs)
{
    if (!ctx)
    {
        throw std::runtime_error("Whisper context not initialized!");
    }

    const int MAX_TALKING_MS = 600000; // 10 minutes without transcription, force transcription
    const size_t max_buffer_samples = (MAX_TALKING_MS * WHISPER_SAMPLE_RATE) / 1000;

    const auto maxSilenceChrono = std::chrono::milliseconds(maxSilenceMs);

    // Audio buffer management
    const int n_samples_keep = (params.keep_ms * WHISPER_SAMPLE_RATE) / 1000;

    // Merge context from previous iteration
    std::vector<float> pcmf32_context;
    if (!pcmf32_old.empty())
    {
        const int n_samples_take = std::min(static_cast<int>(pcmf32_old.size()), n_samples_keep);
        pcmf32_context.insert(pcmf32_context.end(),
                              pcmf32_old.end() - n_samples_take,
                              pcmf32_old.end());
    }

    // Add new audio data to voice buffer
    pcmf32_voice.insert(pcmf32_voice.end(), audio_data.begin(), audio_data.end());

    // Analyze the accumulated voice buffer + new audio
    const int vad_window_ms = minSilenceSpeakingMs; // Optimal for voice detection
    const auto &analysis_buffer = pcmf32_voice;     // Use accumulated buffer

    bool silence_detected = vad_deepseek(
        analysis_buffer,
        WHISPER_SAMPLE_RATE,
        vad_window_ms,
        params.vad_thold,  // Recommended: 1.5-2.0
        params.freq_thold, // Recommended: 50.0-100.0
        params.verbose);   // debug

    // State tracking with hysteresis
    const auto now = std::chrono::steady_clock::now();
    const auto silence_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_voice_time);

    if (!silence_detected) // if it was speaking process when the sentence finish, if too long silence process anyway to get back a "silent" data processing warning
    {
        is_speaking = true;
    }
    else if (is_speaking)
    {
        last_voice_time = now;
        flushCmd = true; // true;
    }
    else // eliminate silence from transcription
        if (silence_duration > maxSilenceChrono)
        {
            if (params.verbose)
                fprintf(stderr, "Cleaned up silent buffer\n");

            last_voice_time = now;
            // Clean up buffers
            pcmf32_old = pcmf32_voice;
            pcmf32_voice.clear();

            const size_t min_silence_samples = (minSilenceSpeakingMs * WHISPER_SAMPLE_RATE) / 1000;
            // Keep only the context portion
            if (pcmf32_old.size() > min_silence_samples)
            {
                pcmf32_old.erase(pcmf32_old.begin(), pcmf32_old.end() - min_silence_samples);
            }
        }

    bool forceTranscription = pcmf32_voice.size() > max_buffer_samples;
    // Process audio when flush is requested or buffer is full
    if (flushCmd || forceTranscription)
    {
        // Combine context and voice data
        std::vector<float> pcmf32(pcmf32_context);
        pcmf32.insert(pcmf32.end(), pcmf32_voice.begin(), pcmf32_voice.end());

        // Prepare Whisper parameters
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = params.language;
        wparams.n_threads = params.n_threads;
        wparams.translate = params.translate;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.single_segment = true;
        wparams.no_context = true;
        // wparams.suppress_non_speech_tokens = true;

        // Run transcription
        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0)
        {
            throw std::runtime_error("Whisper transcription failed!");
        }

        // Collect results
        std::string result;
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i)
        {
            const char *text = whisper_full_get_segment_text(ctx, i);
            if (text)
                result += text;
        }

        // Clean up buffers
        pcmf32_old = pcmf32_voice;
        pcmf32_voice.clear();
        is_speaking = false;

        int n_samples_to_actually_keep = n_samples_keep;
        if (forceTranscription)
            n_samples_to_actually_keep = fmaxf(WHISPER_SAMPLE_RATE * 2, n_samples_keep); // keep at least 2 seconds overlap
        // Keep only the context portion
        if (pcmf32_old.size() > n_samples_keep)
        {
            pcmf32_old.erase(pcmf32_old.begin(), pcmf32_old.end() - n_samples_keep);
        }

        // Call callback with final phrase
        if (callback && !result.empty())
        {
            callback(result);
        }
    }
    else
    {
        // Retain context for next iteration
        pcmf32_old = pcmf32_voice;
        if (pcmf32_old.size() > n_samples_keep)
        {
            pcmf32_old.erase(pcmf32_old.begin(), pcmf32_old.end() - n_samples_keep);
        }
    }
}

void WhisperService::stop()
{
    if (ctx)
    {
        whisper_free(ctx);
        ctx = nullptr;
    }
}

void WhisperService::setCallback(std::function<void(const std::string &)> cb)
{
    this->callback = cb;
}

#endif // __cplusplus

// C-compatible functions implementation
extern "C"
{

    // Function to create a new WhisperService instance
    WhisperServiceHandle whisper_service_create(const WhisperParams *params)
    {
        if (!params)
            return nullptr;

        // Convert C WhisperParams to C++ WhisperParams
        WhisperParams cpp_params;
        cpp_params.n_threads = params->n_threads;
        cpp_params.step_ms = params->step_ms;
        cpp_params.length_ms = params->length_ms;
        cpp_params.keep_ms = params->keep_ms;
        cpp_params.max_tokens = params->max_tokens;
        cpp_params.audio_ctx = params->audio_ctx;

        cpp_params.vad_thold = params->vad_thold;
        cpp_params.freq_thold = params->freq_thold;

        cpp_params.translate = params->translate;
        cpp_params.no_fallback = params->no_fallback;
        cpp_params.print_special = params->print_special;
        cpp_params.no_context = params->no_context;
        cpp_params.no_timestamps = params->no_timestamps;
        cpp_params.use_gpu = params->use_gpu;
        cpp_params.flash_attn = params->flash_attn;

        // Convert C strings to C++ std::string
        cpp_params.language = params->language ? params->language : "";
        cpp_params.model = params->model ? params->model : "";

        // Create a new WhisperService instance
        WhisperService *service = nullptr;
        try
        {
            service = new WhisperService(cpp_params);
        }
        catch (const std::bad_alloc &)
        {
            std::cerr << "Failed to allocate WhisperService instance!" << std::endl;
            return nullptr;
        }

        return reinterpret_cast<WhisperServiceHandle>(service);
    }

    // Function to destroy the WhisperService instance
    void whisper_service_destroy(WhisperServiceHandle handle)
    {
        if (!handle)
            return;

        WhisperService *service = reinterpret_cast<WhisperService *>(handle);
        delete service;
    }

    // Function to initialize the WhisperService
    int whisper_service_initialize(WhisperServiceHandle handle)
    {
        if (!handle)
            return 0;

        WhisperService *service = reinterpret_cast<WhisperService *>(handle);
        try
        {
            return service->initialize() ? 1 : 0;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception during initialization: " << e.what() << std::endl;
            return 0;
        }
    }

    // Function to process an audio chunk
    void whisper_service_process_audio_chunk(WhisperServiceHandle handle, const float *audio_data, int length)
    {
        if (!handle || !audio_data || length <= 0)
            return;

        WhisperService *service = reinterpret_cast<WhisperService *>(handle);
        std::vector<float> audio_chunk(audio_data, audio_data + length);

        try
        {
            service->processAudioChunk(audio_chunk);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error processing audio chunk: " << e.what() << std::endl;
            // Handle error as needed (e.g., set an error state)
        }
    }

    // Function to process an audio chunk
    void whisper_service_process_audio_stream(WhisperServiceHandle handle, const float *audio_data, int length, bool flushCmd, int minSilentSpeakingMs, int maxSilenceMs)
    {
        if (!handle || !audio_data || length <= 0)
            return;

        WhisperService *service = reinterpret_cast<WhisperService *>(handle);
        std::vector<float> audio_chunk(audio_data, audio_data + length);

        try
        {
            service->processAudioStream(audio_chunk, flushCmd, minSilentSpeakingMs, maxSilenceMs);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error processing audio chunk: " << e.what() << std::endl;
            // Handle error as needed (e.g., set an error state)
        }
    }

    // Function to stop the transcription service
    void whisper_service_stop(WhisperServiceHandle handle)
    {
        if (!handle)
            return;

        WhisperService *service = reinterpret_cast<WhisperService *>(handle);
        service->stop();
    }

    // Function to set the transcription callback
    void whisper_service_set_callback(WhisperServiceHandle handle, TranscriptionCallbackC callback)
    {
        if (!handle || !callback)
            return;

        WhisperService *service = reinterpret_cast<WhisperService *>(handle);

        // Wrap the C callback in a C++ lambda
        service->setCallback([callback](const std::string &transcription)
                             {
        // Convert std::string to C string
        callback(transcription.c_str()); });
    }

} // extern "C"