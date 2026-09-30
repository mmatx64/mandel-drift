#include "ambient_synth.h"
#include <algorithm>
#include <cmath>

namespace {
constexpr double pi = 3.14159265358979323846;
constexpr int chords[6][4] = {
    {48, 55, 62, 63}, {48, 55, 58, 63}, {48, 55, 60, 63},
    {51, 58, 62, 65}, {53, 60, 63, 67}, {48, 55, 58, 62}};
constexpr int roots[6] = {48, 44, 44, 51, 53, 48};
constexpr int chord_frames = AmbientSynth::sample_rate * 24;
// Two-second steps with breathing room; original phrase, no sample playback.
constexpr int phrase[12] = {0, 2, 1, -1, 3, 2, 0, -1, 1, 2, 3, -1};
constexpr int note_frames = AmbientSynth::sample_rate * 2;
double frequency(int midi) { return 440.0 * std::exp2((midi - 69) / 12.0); }
}

AmbientSynth::AmbientSynth() : echo_left_(48000, 0), echo_right_(48000, 0) {
  constexpr int lengths[] = {1499, 2111, 1601, 1987};
  for (int i = 0; i < 4; ++i) diffusion_[i].resize(lengths[i], 0);
  for (int i = 0; i <= 2048; ++i) table_[i] = static_cast<float>(std::sin(2 * pi * i / 2048));
  for (int bank = 0; bank < 2; ++bank)
    for (int voice = 0; voice < 4; ++voice) frequencies_[bank][voice] = frequency(chords[bank][voice]);
}

float AmbientSynth::wave(double& phase, double hz) {
  phase += hz / sample_rate;
  phase -= std::floor(phase);
  return sample(phase);
}

float AmbientSynth::sample(double phase) const {
  phase -= std::floor(phase);
  const double index = phase * 2048;
  const int whole = static_cast<int>(index);
  return table_[whole] + static_cast<float>(index - whole) * (table_[whole + 1] - table_[whole]);
}

float AmbientSynth::pad(double& phase, double hz, float brightness) {
  const float fundamental = wave(phase, hz);
  // Round body and a soft upper sheen; keep high harmonics below the motif.
  return fundamental + brightness * (0.30f * sample(phase * 2) +
      0.12f * sample(phase * 3) + 0.045f * sample(phase * 4) +
      0.018f * sample(phase * 6));
}

float AmbientSynth::diffuse(float value, int channel) {
  // Two stable all-pass stages smear the long echoes without emphasizing a
  // frequency band. Different prime lengths decorrelate the stereo tails.
  for (int stage = channel * 2; stage < channel * 2 + 2; ++stage) {
    auto& buffer = diffusion_[stage];
    int& index = diffusion_index_[stage];
    const float delayed = buffer[index];
    const float output = delayed - value * 0.6f;
    buffer[index] = value + output * 0.6f;
    index = (index + 1) % static_cast<int>(buffer.size());
    value = output;
  }
  return value;
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
    const double seconds = frame_ / static_cast<double>(sample_rate);
    const float cross = std::clamp((chord_frame / static_cast<float>(sample_rate) - 12.0f) / 12.0f, 0.0f, 1.0f);
    const float blend = cross * cross * (3 - 2 * cross);
    float left = 0, right = 0;
    for (int bank = 0; bank < 2; ++bank) {
      const float weight = bank == bank_ ? 1 - blend : blend;
      for (int voice = 0; voice < 4; ++voice) {
        const double hz = frequencies_[bank][voice];
        const double drift = 0.00035 * sample(seconds / 23.0 + voice * 0.21);
        const float a = pad(phases_[bank][voice * 2], hz * (0.9988 + drift), 0.85f + depth_ * 0.15f);
        const float b = pad(phases_[bank][voice * 2 + 1], hz * (1.0012 - drift), 0.85f + depth_ * 0.15f);
        const float swell = 0.85f + 0.15f * sample(seconds / (9.0 + voice * 1.9) + voice / (2 * pi));
        const float level = voice < 2 ? 0.055f : 0.032f;
        left += weight * swell * (a + b * 0.18f) * level;
        right += weight * swell * (b + a * 0.18f) * level;
      }
    }
    // Crossfade two tuned roots instead of sliding through unrelated pitches.
    const float bass = (wave(bass_phases_[bank_], frequency(roots[chord_])) * (1 - blend) +
        wave(bass_phases_[1 - bank_], frequency(roots[(chord_ + 1) % 6])) * blend) * 0.014f;
    // Independent voices let each note decay beneath the next, with no retrigger clicks.
    if (frame_ % note_frames == 0) {
      const auto step = frame_ / note_frames;
      const int voice = phrase[step % 12];
      if (voice >= 0) {
        auto& note = notes_[step % notes_.size()];
        // Follow the louder pad bank halfway through each harmonic transition.
        const int harmony = blend < 0.5f ? chord_ : (chord_ + 1) % 6;
        note.frequency = frequency(chords[harmony][voice] + 12);
        note.envelope = 1;
        note.attack = 0;
        note.pan = step % 2 == 0 ? 0.32f : 0.68f;
      }
    }
    for (auto& note : notes_) {
      note.envelope *= 0.999981f;
      note.attack += (1 - note.attack) * 0.00022f;
      const float tone = wave(note.phase, note.frequency) + 0.10f * sample(note.phase * 2);
      const float bell = tone * note.envelope * note.attack * (0.036f + 0.004f * depth_);
      left += bell * (1 - note.pan);
      right += bell * note.pan;
    }
    left += bass;
    right += bass;
    const float echo_l = echo_left_[(echo_index_ + 48000 - 31200) % 48000];
    const float echo_r = echo_right_[(echo_index_ + 48000 - 40800) % 48000];
    low_left_ += (echo_r - low_left_) * 0.045f;
    low_right_ += (echo_l - low_right_) * 0.045f;
    echo_left_[echo_index_] = left + low_left_ * 0.68f;
    echo_right_[echo_index_] = right + low_right_ * 0.68f;
    echo_index_ = (echo_index_ + 1) % 48000;
    pad_low_left_ += (left + diffuse(echo_l, 0) * 0.55f - pad_low_left_) * 0.12f;
    pad_low_right_ += (right + diffuse(echo_r, 1) * 0.55f - pad_low_right_) * 0.12f;
    const float fade = static_cast<float>(std::min(frame_ / (sample_rate * 4.0), 1.0));
    const float attack = fade * fade * (3 - 2 * fade);
    // Quiet 200/210 Hz carriers: a 10 Hz difference on headphones. Keep these
    // channel-specific and outside crossfeed/reverb and music saturation.
    const float binaural_left = wave(binaural_phases_[0], 200) * 0.009f;
    const float binaural_right = wave(binaural_phases_[1], 210) * 0.009f;
    stereo[i * 2] = (std::tanh(pad_low_left_ * 1.25f) * 0.8f + binaural_left) * gain_ * attack;
    stereo[i * 2 + 1] = (std::tanh(pad_low_right_ * 1.25f) * 0.8f + binaural_right) * gain_ * attack;
    ++frame_;
  }
}
