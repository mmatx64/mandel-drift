#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <vector>

// Device-independent, deterministic stereo synthesis. Owned by the audio thread.
class AmbientSynth {
 public:
  static constexpr int sample_rate = 48000;
  explicit AmbientSynth(std::uint32_t seed = 3080);
  void render(float* stereo, int frames, float target_gain, float target_depth,
              float target_julia = 0);
  float musical_level() const;
 private:
  struct Note {
    double phase = 0;
    double frequency = 440;
    int age = -1;
    float pan = 0.5f;
    float expression = 1;
  };
  struct ReverbLine {
    std::vector<float> samples;
    int index = 0;
    float low = 0;
  };
  float wave(double& phase, double frequency);
  float sample(double phase) const;
  float pad(double& phase, double frequency, float brightness);
  float diffuse(float value, int channel);
  std::array<float, 2> reverberate(float left, float right);
  void update_controls(float target_depth, float target_julia);
  std::mt19937 random_;
  std::array<int, 5> voicing_{};
  double drift_phase_ = 0;
  int lead_octave_ = 0, figure_offset_ = 0, air_offset_ = 0;
  std::uint64_t next_lead_frame_ = 0, lead_step_ = 0;
  int lead_spacing_ = sample_rate * 4;
  float musical_energy_ = 0;
  std::array<float, 2049> table_{};
  std::array<std::array<double, 10>, 2> phases_{};
  std::array<std::array<double, 10>, 2> frequencies_{};
  std::array<float, 5> pad_levels_{};
  std::array<ReverbLine, 8> reverb_;
  std::vector<float> echo_left_, echo_right_;
  std::array<std::vector<float>, 4> diffusion_;
  std::array<int, 4> diffusion_index_{};
  std::uint64_t frame_ = 0;
  int chord_ = 0;
  int bank_ = 0;
  int echo_index_ = 0;
  std::array<double, 2> bass_phases_{};
  std::array<double, 2> bass_frequencies_{};
  std::array<Note, 8> lead_notes_{};
  std::array<Note, 8> figure_notes_{};
  std::array<Note, 4> air_notes_{};
  std::array<double, 2> binaural_phases_{};
  double depth_ = 0, julia_ = 0;
  float gain_ = 0;
  float brightness_ = 0, filter_ = 0;
  float lead_level_ = 0, figure_level_ = 0, air_level_ = 0;
  float bass_level_ = 0, lead_vibrato_ = 0, air_drift_ = 0;
  float echo_low_left_ = 0, echo_low_right_ = 0;
  float pad_low_left_ = 0, pad_low_right_ = 0;
  float low_left_ = 0, low_right_ = 0;
};
