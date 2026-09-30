#pragma once
#include "ambient_synth.h"
#include <SDL3/SDL.h>
#include <atomic>

class AmbientMusic {
 public:
  AmbientMusic() = default;
  ~AmbientMusic();
  AmbientMusic(const AmbientMusic&) = delete;
  AmbientMusic& operator=(const AmbientMusic&) = delete;
  bool start();
  void set_controls(bool enabled, bool muted, int volume);
  void set_depth(float depth) { depth_.store(depth, std::memory_order_relaxed); }
  bool available() const { return stream_ && !failed_.load(); }
  unsigned long long rendered_frames() const { return rendered_.load(); }
  float envelope() const { return envelope_.load(std::memory_order_relaxed); }
 private:
  static void SDLCALL feed(void* user, SDL_AudioStream* stream, int additional, int total);
  SDL_AudioStream* stream_ = nullptr;
  AmbientSynth synth_;
  std::atomic<float> gain_{0}, depth_{0};
  std::atomic<float> envelope_{0};
  std::atomic<bool> failed_{false};
  std::atomic<unsigned long long> rendered_{0};
};
