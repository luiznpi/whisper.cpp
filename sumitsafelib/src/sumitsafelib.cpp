#include "sumitsafelib.h"
#include "whisper.h"

#include <stdexcept>
#include <iostream>
#include <cstring>

#ifdef __cplusplus

// C++ class implementation
WhisperService::WhisperService(const WhisperParams &params)
    : params(params), ctx(nullptr), callback(nullptr) {}

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