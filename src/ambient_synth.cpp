#include "ambient_synth.h"
#include <algorithm>
#include <cmath>

namespace {
constexpr double pi = 3.14159265358979323846;
constexpr int chords[6][5] = {
    {48, 55, 60, 62, 63}, {44, 55, 60, 63, 70}, {51, 58, 60, 65, 67},
    {46, 53, 60, 62, 65}, {41, 53, 60, 63, 67}, {48, 55, 58, 62, 67}};
constexpr int roots[6] = {36, 32, 39, 34, 29, 36};
constexpr int chord_frames = AmbientSynth::sample_rate * 32;
// The same original C/G/D/Eb motif survives every change in orchestration.
constexpr int theme[8] = {72, 79, 74, 75, -1, 74, 72, -1};
constexpr int figure[8] = {60, 67, 74, 67, 63, 67, 74, -1};
constexpr int air[6] = {79, 82, 86, 82, 79, 74};
constexpr int control_frames = 64;
double frequency(int midi) { return 440.0 * std::exp2((midi - 69) / 12.0); }
float smooth(double value) {
  const float t = static_cast<float>(std::clamp(value, 0.0, 1.0));
  return t * t * (3 - 2 * t);
}
float envelope(int age, double attack, double hold, double release) {
  const double seconds = age / static_cast<double>(AmbientSynth::sample_rate);
  return smooth(seconds / attack) * (1 - smooth((seconds - hold) / release));
}
}

AmbientSynth::AmbientSynth(std::uint32_t seed)
    : random_(seed), echo_left_(48000, 0), echo_right_(48000, 0) {
  chord_ = std::uniform_int_distribution<int>(0, 5)(random_);
  drift_phase_ = std::uniform_real_distribution<double>(0, 1)(random_);
  lead_octave_ = std::bernoulli_distribution(.35)(random_) ? -12 : 0;
  figure_offset_ = std::uniform_int_distribution<int>(0, 7)(random_);
  air_offset_ = std::uniform_int_distribution<int>(0, 5)(random_);
  for (int voice = 3; voice < 5; ++voice)
    voicing_[voice] = std::bernoulli_distribution(.4)(random_) ? 12 : 0;
  for (auto& bank : phases_)
    for (auto& phase : bank) phase = std::uniform_real_distribution<double>(0, 1)(random_);
  constexpr int diffusion_lengths[] = {1499, 2111, 1601, 1987};
  constexpr int reverb_lengths[] = {4217, 4787, 5521, 6311, 7219, 8089, 9041, 10007};
  for (int i = 0; i < 4; ++i) diffusion_[i].resize(diffusion_lengths[i], 0);
  for (int i = 0; i < 8; ++i) reverb_[i].samples.resize(reverb_lengths[i], 0);
  for (int i = 0; i <= 2048; ++i)
    table_[i] = static_cast<float>(std::sin(2 * pi * i / 2048));
  update_controls(0, 0);
}

float AmbientSynth::musical_level() const { return std::sqrt(musical_energy_); }

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
  return fundamental + brightness * (0.42f * sample(phase * 2) +
      0.20f * sample(phase * 3) + 0.10f * sample(phase * 4) +
      0.04f * sample(phase * 5) + 0.02f * sample(phase * 7));
}

float AmbientSynth::diffuse(float value, int channel) {
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

std::array<float, 2> AmbientSynth::reverberate(float left, float right) {
  std::array<float, 8> delayed{};
  float sum = 0;
  for (int i = 0; i < 8; ++i) {
    auto& line = reverb_[i];
    line.low += (line.samples[line.index] - line.low) * 0.12f;
    delayed[i] = line.low;
    sum += delayed[i];
  }
  // Orthogonal Householder feedback mixes eight delays into a stable, long tail.
  for (int i = 0; i < 8; ++i) {
    auto& line = reverb_[i];
    const float input = (i % 2 == 0 ? left : right) * (i < 4 ? 0.22f : -0.22f);
    line.samples[line.index] = input + 0.91f * (delayed[i] - sum * 0.25f);
    line.index = (line.index + 1) % static_cast<int>(line.samples.size());
  }
  return {{(delayed[0] + delayed[2] - delayed[4] - delayed[6]) * 0.5f,
           (delayed[1] + delayed[3] - delayed[5] - delayed[7]) * 0.5f}};
}

void AmbientSynth::update_controls(float target_depth, float target_julia) {
  // Control-rate smoothing reaches 95% in about 20 seconds, including on return.
  constexpr double depth_step = 0.000205107167786; // 6.5-second time constant.
  constexpr double julia_step = 0.000166652778549; // 8-second time constant.
  depth_ += (target_depth - depth_) * depth_step;
  julia_ += (target_julia - julia_) * julia_step;
  const double seconds = frame_ / static_cast<double>(sample_rate);
  const float movement = sample(seconds / 37.0 + drift_phase_);
  brightness_ = 0.30f + static_cast<float>(depth_) * 0.45f + movement * 0.055f;
  const double cutoff = 720 + depth_ * 1150 + movement * 110;
  filter_ = static_cast<float>(1 - std::exp(-2 * pi * cutoff / sample_rate));
  lead_level_ = 0.042f + static_cast<float>(depth_) * 0.031f;
  figure_level_ = 0.020f * smooth((depth_ - 0.20) / 0.65);
  air_level_ = 0.014f * smooth((depth_ - 0.45) / 0.50) + 0.016f * static_cast<float>(julia_);
  bass_level_ = 0.020f + static_cast<float>(depth_) * 0.008f;
  lead_vibrato_ = 0.0012f * sample(seconds * 4.6);
  air_drift_ = 0.0009f * sample(seconds / 11.0);
  for (int voice = 0; voice < 5; ++voice) {
    const float breath = 0.86f + 0.14f * sample(seconds / (15.0 + voice * 3.1) + voice * 0.17 + drift_phase_);
    pad_levels_[voice] = breath * (voice < 2 ? 0.032f : 0.021f) *
        (voice < 3 ? 1.0f : 0.65f + static_cast<float>(depth_) * 0.35f);
    const double drift = 0.0006 * sample(seconds / (27.0 + voice * 2.0) + voice * 0.21);
    for (int bank = 0; bank < 2; ++bank) {
      const int harmony = bank == bank_ ? chord_ : (chord_ + 1) % 6;
      const double hz = frequency(chords[harmony][voice] + voicing_[voice]);
      frequencies_[bank][voice * 2] = hz * (0.9985 + drift);
      frequencies_[bank][voice * 2 + 1] = hz * (1.0015 - drift);
    }
  }
  bass_frequencies_[bank_] = frequency(roots[chord_]);
  bass_frequencies_[1 - bank_] = frequency(roots[(chord_ + 1) % 6]);
}

void AmbientSynth::render(float* stereo, int frames, float target_gain, float target_depth,
                          float target_julia) {
  target_gain = std::isfinite(target_gain) ? std::clamp(target_gain, 0.0f, 1.0f) : 0;
  target_depth = std::isfinite(target_depth) ? std::clamp(target_depth, 0.0f, 1.0f) : 0;
  target_julia = std::isfinite(target_julia) ? std::clamp(target_julia, 0.0f, 1.0f) : 0;
  for (int i = 0; i < frames; ++i) {
    const int chord_frame = static_cast<int>(frame_ % chord_frames);
    if (chord_frame == 0 && frame_ != 0) {
      chord_ = (chord_ + 1) % 6;
      bank_ = 1 - bank_;
    }
    if (frame_ % control_frames == 0) update_controls(target_depth, target_julia);
    gain_ += (target_gain - gain_) * 0.00045f;
    if (target_gain == 0 && gain_ < 0.000001f) gain_ = 0;
    const float blend = smooth((chord_frame / static_cast<double>(sample_rate) - 16) / 16);
    float left = 0, right = 0;
    for (int bank = 0; bank < 2; ++bank) {
      const float weight = bank == bank_ ? 1 - blend : blend;
      for (int voice = 0; voice < 5; ++voice) {
        const float a = pad(phases_[bank][voice * 2], frequencies_[bank][voice * 2], brightness_);
        const float b = pad(phases_[bank][voice * 2 + 1], frequencies_[bank][voice * 2 + 1], brightness_);
        left += weight * pad_levels_[voice] * (a + b * 0.22f);
        right += weight * pad_levels_[voice] * (b + a * 0.22f);
      }
    }
    pad_low_left_ += (left - pad_low_left_) * filter_;
    pad_low_right_ += (right - pad_low_right_) * filter_;
    left = pad_low_left_;
    right = pad_low_right_;
    const float bass = (wave(bass_phases_[bank_], bass_frequencies_[bank_]) * (1 - blend) +
        wave(bass_phases_[1 - bank_], bass_frequencies_[1 - bank_]) * blend) * bass_level_;
    left += bass;
    right += bass;

    if (frame_ == next_lead_frame_) {
      const auto step = lead_step_++;
      if (step % 8 == 0)
        lead_spacing_ = std::uniform_int_distribution<int>(sample_rate * 38 / 10, sample_rate * 45 / 10)(random_);
      next_lead_frame_ += lead_spacing_;
      int pitch = theme[step % 8];
      // Keep the four-note opening; vary only its quiet answering phrase.
      constexpr int answers[] = {72, 70, 67};
      if (step % 8 == 6) pitch = answers[std::uniform_int_distribution<int>(0, 2)(random_)];
      if (step % 8 == 4 && std::bernoulli_distribution(.25)(random_)) pitch = 79;
      if (pitch >= 0) {
        auto& note = lead_notes_[step % lead_notes_.size()];
        note.frequency = frequency(pitch + lead_octave_);
        note.age = 0;
        note.pan = std::uniform_real_distribution<float>(.40f, .60f)(random_);
        note.expression = std::uniform_real_distribution<float>(.82f, 1.0f)(random_);
      }
    }
    if (frame_ % (sample_rate * 2) == 0) {
      const auto step = frame_ / (sample_rate * 2);
      const int pitch = figure[(step + figure_offset_) % 8];
      if (pitch >= 0) {
        auto& note = figure_notes_[step % figure_notes_.size()];
        note.frequency = frequency(pitch);
        note.age = 0;
        note.pan = step % 2 == 0 ? 0.28f : 0.72f;
      }
    }
    if (frame_ % (sample_rate * 16) == 0) {
      const auto step = frame_ / (sample_rate * 16);
      auto& note = air_notes_[step % air_notes_.size()];
      note.frequency = frequency(air[(step + air_offset_) % 6]);
      note.age = 0;
      note.pan = step % 2 == 0 ? 0.22f : 0.78f;
    }
    for (auto& note : lead_notes_) {
      if (note.age < 0) continue;
      const float env = envelope(note.age, 1.3, 2.8, 7.2);
      const float fundamental = wave(note.phase, note.frequency * (1 + lead_vibrato_ * env));
      const float tone = fundamental + env * (0.26f * sample(note.phase * 2) +
          0.07f * sample(note.phase * 3));
      const float value = tone * env * lead_level_ * note.expression;
      left += value * (1 - note.pan);
      right += value * note.pan;
      if (++note.age >= sample_rate * 10) note.age = -1;
    }
    for (auto& note : figure_notes_) {
      if (note.age < 0) continue;
      const float env = envelope(note.age, 0.30, 0.45, 4.55);
      const float value = (wave(note.phase, note.frequency) + 0.12f * sample(note.phase * 2)) *
          env * figure_level_;
      left += value * (1 - note.pan);
      right += value * note.pan;
      if (++note.age >= sample_rate * 5) note.age = -1;
    }
    for (auto& note : air_notes_) {
      if (note.age < 0) continue;
      const float env = envelope(note.age, 4.0, 5.0, 13.0);
      wave(note.phase, note.frequency * (1 + air_drift_));
      const float value = sample(note.phase + 0.055 * sample(note.phase * 2)) * env * air_level_;
      left += value * (1 - note.pan);
      right += value * note.pan;
      if (++note.age >= sample_rate * 18) note.age = -1;
    }

    const float echo_l = echo_left_[(echo_index_ + 48000 - 31200) % 48000];
    const float echo_r = echo_right_[(echo_index_ + 48000 - 40800) % 48000];
    echo_low_left_ += (echo_r - echo_low_left_) * 0.055f;
    echo_low_right_ += (echo_l - echo_low_right_) * 0.055f;
    echo_left_[echo_index_] = left + echo_low_left_ * 0.48f;
    echo_right_[echo_index_] = right + echo_low_right_ * 0.48f;
    echo_index_ = (echo_index_ + 1) % 48000;
    const auto tail = reverberate(diffuse(left + echo_l * 0.24f, 0),
                                diffuse(right + echo_r * 0.24f, 1));
    low_left_ += (left + echo_l * 0.22f + tail[0] * 0.85f - low_left_) * 0.20f;
    low_right_ += (right + echo_r * 0.22f + tail[1] * 0.85f - low_right_) * 0.20f;
    const float opening = smooth(frame_ / (sample_rate * 8.0));
    const float musical_left = std::tanh(low_left_ * 1.35f) * 0.8f * opening;
    const float musical_right = std::tanh(low_right_ * 1.35f) * 0.8f * opening;
    // Stereo musical energy before volume; exclude the constant binaural tones.
    musical_energy_ += ((musical_left * musical_left + musical_right * musical_right) * .5f -
                        musical_energy_) * .000173596f;
    // Binaural carriers remain channel-specific, outside music crossfeed/reverb.
    const float binaural_left = wave(binaural_phases_[0], 200) * 0.009f;
    const float binaural_right = wave(binaural_phases_[1], 210) * 0.009f;
    stereo[i * 2] = (musical_left + binaural_left * opening) * gain_;
    stereo[i * 2 + 1] = (musical_right + binaural_right * opening) * gain_;
    // Muting changes only output gain: phrases, controls, and tails keep moving.
    ++frame_;
  }
}
