#include "ambient_synth.h"
#include <algorithm>
#include <cmath>

namespace {
constexpr double pi = 3.14159265358979323846;
constexpr int chords[6][4] = {
    {50, 57, 60, 64}, {46, 53, 57, 62}, {48, 53, 57, 64},
    {48, 55, 62, 64}, {50, 57, 60, 65}, {46, 53, 60, 65}};
constexpr int roots[6] = {38, 34, 29, 36, 38, 34};
constexpr int chord_frames = AmbientSynth::sample_rate * 18;
double frequency(int midi) { return 440.0 * std::exp2((midi - 69) / 12.0); }
}

AmbientSynth::AmbientSynth() : echo_left_(48000, 0), echo_right_(48000, 0) {
  for (int i = 0; i <= 2048; ++i) table_[i] = static_cast<float>(std::sin(2 * pi * i / 2048));
  for (int bank = 0; bank < 2; ++bank)
    for (int voice = 0; voice < 4; ++voice) frequencies_[bank][voice] = frequency(chords[bank][voice]);
}

float AmbientSynth::wave(double& phase, double hz) {
  phase += hz / sample_rate;
  phase -= std::floor(phase);
  const double index = phase * 2048;
  const int whole = static_cast<int>(index);
  return table_[whole] + static_cast<float>(index - whole) * (table_[whole + 1] - table_[whole]);
}

void AmbientSynth::render(float* stereo, int frames, float target_gain, float target_depth) {
  target_gain = std::clamp(target_gain, 0.0f, 1.0f);
  target_depth = std::clamp(target_depth, 0.0f, 1.0f);
  for (int i = 0; i < frames; ++i) {
    gain_ += (target_gain - gain_) * 0.00045f; // ~150 ms fade; no abrupt gain jumps.
    if (target_gain == 0 && gain_ < 0.000001f) {
      gain_ = 0;
      std::fill(stereo + i * 2, stereo + frames * 2, 0.0f);
      return;
    }
    depth_ += (target_depth - depth_) * 0.000007f;
    const int chord_frame = static_cast<int>(frame_ % chord_frames);
    if (chord_frame == 0 && frame_ != 0) {
      chord_ = (chord_ + 1) % 6;
      bank_ = 1 - bank_;
      for (int voice = 0; voice < 4; ++voice)
        frequencies_[1 - bank_][voice] = frequency(chords[(chord_ + 1) % 6][voice]);
    }
    const float cross = std::clamp((chord_frame / static_cast<float>(sample_rate) - 12.0f) / 6.0f, 0.0f, 1.0f);
    const float blend = cross * cross * (3 - 2 * cross);
    float left = 0, right = 0;
    for (int bank = 0; bank < 2; ++bank) {
      const float weight = bank == bank_ ? 1 - blend : blend;
      for (int voice = 0; voice < 4; ++voice) {
        const double hz = frequencies_[bank][voice];
        const float a = wave(phases_[bank][voice * 2], hz * 0.9985);
        const float b = wave(phases_[bank][voice * 2 + 1], hz * 1.0015);
        // A quiet third harmonic, slowly brightened by dive depth.
        const float warm_a = a + (0.04f + depth_ * 0.07f) * (3 * a - 4 * a * a * a);
        const float warm_b = b + (0.04f + depth_ * 0.07f) * (3 * b - 4 * b * b * b);
        const float pan = 0.25f + voice * 0.16f;
        left += weight * (warm_a * (1 - pan) + warm_b * 0.15f) * 0.10f;
        right += weight * (warm_b * pan + warm_a * 0.15f) * 0.10f;
      }
    }
    // Bass follows the chord with a six-second pitch glide during the crossfade.
    const double root = frequency(roots[chord_]) * (1 - blend) + frequency(roots[(chord_ + 1) % 6]) * blend;
    const float pulse = 0.8f + 0.2f * static_cast<float>(std::sin(frame_ * (2 * pi / (sample_rate * 6.0))));
    const float bass = wave(bass_phase_, root) * 0.12f * pulse;
    // Sparse notes remain in the active harmony; long stereo echoes fill the gaps.
    constexpr int note_frames = sample_rate * 3;
    if (frame_ % note_frames == 0) {
      const int step = static_cast<int>((frame_ / note_frames) % 8);
      if (step != 3 && step != 7) {
        const int voice = (step * 3 + chord_) % 4;
        bell_frequency_ = frequency(chords[chord_][voice] + 12);
        bell_envelope_ = 1;
        bell_attack_ = 0;
      }
    }
    bell_envelope_ *= 0.999975f;
    bell_attack_ += (1 - bell_attack_) * 0.0005f;
    const float bell = wave(bell_phase_, bell_frequency_) * bell_envelope_ * bell_attack_ * (0.035f + 0.035f * depth_);
    left += bass + bell * 0.75f;
    right += bass + bell * 0.50f;
    const float echo_l = echo_left_[(echo_index_ + 48000 - 31200) % 48000];
    const float echo_r = echo_right_[(echo_index_ + 48000 - 40800) % 48000];
    low_left_ += (echo_r - low_left_) * 0.12f;
    low_right_ += (echo_l - low_right_) * 0.12f;
    echo_left_[echo_index_] = left + low_left_ * 0.42f;
    echo_right_[echo_index_] = right + low_right_ * 0.42f;
    echo_index_ = (echo_index_ + 1) % 48000;
    stereo[i * 2] = std::tanh((left + echo_l * 0.32f) * 1.25f) * gain_ * 0.8f;
    stereo[i * 2 + 1] = std::tanh((right + echo_r * 0.32f) * 1.25f) * gain_ * 0.8f;
    ++frame_;
  }
}
