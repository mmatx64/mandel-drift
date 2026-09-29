#pragma once
#include <array>
#include <cstdint>
#include <vector>

// Device-independent, deterministic stereo synthesis. Owned by the audio thread.
class AmbientSynth {
 public:
  static constexpr int sample_rate = 48000;
  AmbientSynth();
  void render(float* stereo, int frames, float target_gain, float target_depth);
 private:
  float wave(double& phase, double frequency);
  std::array<float, 2049> table_{};
  std::array<std::array<double, 8>, 2> phases_{};
  std::array<std::array<double, 4>, 2> frequencies_{};
  std::vector<float> echo_left_, echo_right_;
  std::uint64_t frame_ = 0;
  int chord_ = 0;
  int bank_ = 0;
  int echo_index_ = 0;
  double bass_phase_ = 0;
  double bell_phase_ = 0;
  double bell_frequency_ = 440;
  float bell_envelope_ = 0;
  float bell_attack_ = 0;
  float gain_ = 0;
  float depth_ = 0;
  float low_left_ = 0, low_right_ = 0;
};
