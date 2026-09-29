#include "ambient_music.h"
#include <algorithm>
#include <array>

AmbientMusic::~AmbientMusic() {
  // SDL waits for an in-flight callback before freeing the stream and device.
  if (stream_) SDL_DestroyAudioStream(stream_);
}

bool AmbientMusic::start() {
  if (available()) return true;
  if (stream_) { SDL_DestroyAudioStream(stream_); stream_ = nullptr; }
  failed_ = false;
  if (!(SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO) && !SDL_InitSubSystem(SDL_INIT_AUDIO)) return false;
  const SDL_AudioSpec spec{SDL_AUDIO_F32, 2, AmbientSynth::sample_rate};
  stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, feed, this);
  if (!stream_) return false;
  if (SDL_ResumeAudioStreamDevice(stream_)) return true;
  SDL_DestroyAudioStream(stream_);
  stream_ = nullptr;
  return false;
}

void AmbientMusic::set_controls(bool enabled, bool muted, int volume) {
  gain_.store(enabled && !muted ? std::clamp(volume, 0, 100) / 100.0f : 0.0f,
              std::memory_order_relaxed);
}

void SDLCALL AmbientMusic::feed(void* user, SDL_AudioStream* stream, int additional, int) {
  auto& music = *static_cast<AmbientMusic*>(user);
  std::array<float, 1024 * 2> samples{};
  int remaining = additional > 0 ? (additional + 7) / 8 : 0;
  while (remaining > 0) {
    const int count = std::min(remaining, 1024);
    music.synth_.render(samples.data(), count, music.gain_.load(std::memory_order_relaxed),
                       music.depth_.load(std::memory_order_relaxed));
    if (!SDL_PutAudioStreamData(stream, samples.data(), count * 2 * sizeof(float))) {
      music.failed_ = true;
      return;
    }
    music.rendered_.fetch_add(count, std::memory_order_relaxed);
    remaining -= count;
  }
}
