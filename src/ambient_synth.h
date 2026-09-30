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
  float sample(double phase) const;
  float pad(double& phase, double frequency, float brightness);
  float diffuse(float value, int channel);
  std::array<float, 2049> table_{};
  std::array<std::array<double, 8>, 2> phases_{};
  std::array<std::array<double, 4>, 2> frequencies_{};
  std::vector<float> echo_left_, echo_right_;
  std::array<std::vector<float>, 4> diffusion_;
  std::array<int, 4> diffusion_index_{};
  std::uint64_t frame_ = 0;
  int chord_ = 0;
  int bank_ = 0;
  int echo_index_ = 0;
  std::array<double, 2> bass_phases_{};
  struct Note {
    double phase = 0;
    double frequency = 440;
    float envelope = 0;
    float attack = 0;
    float pan = 0.5f;
  };
  std::array<Note, 4> notes_{};
  std::array<double, 2> binaural_phases_{};
  float gain_ = 0;
  float depth_ = 0;
  float low_left_ = 0, low_right_ = 0;
  float pad_low_left_ = 0, pad_low_right_ = 0;
};
