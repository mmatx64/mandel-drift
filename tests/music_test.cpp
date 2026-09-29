#include "ambient_music.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool okay, const char* message) {
  if (!okay) throw std::runtime_error(message);
}
void wav(const char* path, const std::vector<float>& data) {
  std::ofstream out(path, std::ios::binary);
  const auto u16 = [&](std::uint16_t value) { out.put(static_cast<char>(value)); out.put(static_cast<char>(value >> 8)); };
  const auto u32 = [&](std::uint32_t value) { u16(static_cast<std::uint16_t>(value)); u16(static_cast<std::uint16_t>(value >> 16)); };
  const auto bytes = static_cast<std::uint32_t>(data.size() * 2);
  out.write("RIFF", 4); u32(36 + bytes); out.write("WAVEfmt ", 8); u32(16);
  u16(1); u16(2); u32(48000); u32(192000); u16(4); u16(16);
  out.write("data", 4); u32(bytes);
  for (float value : data) u16(static_cast<std::uint16_t>(static_cast<std::int16_t>(std::lround(value * 32767))));
  require(static_cast<bool>(out), "Could not write preview WAV");
}

int main(int argc, char** argv) {
  try {
    AmbientSynth synth;
    std::vector<float> samples(48000 * 2);
    std::vector<float> preview;
    double energy = 0, difference = 0;
    float peak = 0, jump = 0, previous = 0;
    for (int second = 0; second < 130; ++second) {
      synth.render(samples.data(), 48000, 1, second < 65 ? 0 : 1);
      for (size_t i = 0; i < samples.size(); i += 2) {
        require(std::isfinite(samples[i]) && std::isfinite(samples[i + 1]), "Non-finite audio");
        peak = std::max({peak, std::abs(samples[i]), std::abs(samples[i + 1])});
        jump = std::max(jump, std::abs(samples[i] - previous));
        previous = samples[i];
        energy += samples[i] * samples[i];
        difference += std::abs(samples[i] - samples[i + 1]);
      }
      if (argc > 1 && second < 40)
        for (float sample : samples) preview.push_back(sample * 0.6f);
    }
    require(peak < 0.8f && peak > 0.05f, "Audio range or clipping");
    require(energy / (48000 * 130) > 0.0001, "Unexpected silence");
    require(difference > 1, "Stereo image missing");
    require(jump < 0.04f, "Discontinuity across notes or chords");
    synth.render(samples.data(), 48000, 0, 1);
    require(std::all_of(samples.end() - 2000, samples.end(), [](float x) { return x == 0; }), "Mute did not settle to exact silence");
    synth.render(samples.data(), 48000, .35f, 0);
    require(std::abs(samples.front()) < 0.001f, "Unmute did not fade in");
    AmbientSynth a, b;
    std::vector<float> whole(20000), split(20000);
    a.render(whole.data(), 10000, .5f, .5f);
    for (int offset = 0; offset < 10000; offset += 100) b.render(split.data() + offset * 2, 100, .5f, .5f);
    require(whole == split, "Synthesis depends on callback buffer size");
    if (argc > 1) wav(argv[1], preview);

    SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "dummy");
    {
      AmbientMusic music;
      music.set_controls(true, false, 35);
      require(music.start(), SDL_GetError());
      SDL_Delay(150);
      require(music.rendered_frames() > 0 && music.available(), "Audio callback did not supply samples");
      music.set_controls(true, true, 35);
      SDL_Delay(50);
      music.set_controls(false, false, 35);
    }
    SDL_Quit();
    SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "nonexistent-test-driver");
    {
      AmbientMusic missing;
      require(!missing.start() && !missing.available(), "Missing audio device must fail gracefully");
    }
    SDL_Quit();
    std::cout << "130 seconds synthesized; peak=" << peak << ", maximum step=" << jump
              << "; mute, unmute, stereo, buffer independence, callback and unavailable device passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
