#include "app_settings.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

void require(bool okay, const char* message) {
  if (!okay) throw std::runtime_error(message);
}

int main() {
  const auto folder = std::filesystem::temp_directory_path() /
      ("mandel-settings-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    std::filesystem::create_directories(folder);
    const auto path = folder / L"settings-\u00e9.ini";
    require(settings_text(load_settings(path)) == settings_text(AppSettings{}), "Missing file defaults");
    AppSettings changed;
    changed.auto_palette = changed.shuffle = changed.paused = changed.menu_visible = true;
    changed.endless_dive = changed.bloom = changed.rotation = false;
    changed.rotation_direction = -1;
    changed.palette = 7;
    changed.window_mode = 2;
    changed.music = false;
    changed.muted = true;
    changed.volume = 73;
    require(save_settings(path, changed), "Save preferences");
    require(settings_text(load_settings(path)) == settings_text(changed), "Reload every preference");
    require(save_settings(path, AppSettings{}), "Replace existing file");
    require(settings_text(load_settings(path)) == settings_text(AppSettings{}), "Reload replacement");
    {
      std::ofstream invalid(path);
      invalid << "\xef\xbb\xbf[MandelDrift]\n"
              << "bloom=0\nshuffle=1 ; comment\nrotation_direction=-1\n"
              << "palette=999\nwindow_mode=-1\nauto_palette=2\npaused=1junk\n"
              << "rotation=99999999999999999999\nendless_dive=\nunknown=42\n"
              << "music=7\nmuted=-1\nvolume=101\nvolume=-2\n"
              << "[OtherApp]\nbloom=1\n";
    }
    AppSettings expected;
    expected.bloom = false;
    expected.shuffle = true;
    expected.rotation_direction = -1;
    require(settings_text(load_settings(path)) == settings_text(expected), "Malformed and unknown values");
    {
      std::ofstream legacy(path);
      legacy << "[MandelDrift]\nwindow_mode=1\n";
    }
    require(load_settings(path).window_mode == 0, "Legacy borderless opens windowed");
    require(!save_settings(folder / "missing" / "settings.ini", changed), "Unwritable location reported");
    for (const auto& file : std::filesystem::directory_iterator(folder))
      require(file.path() == path, "No leftover temporary files");
    std::filesystem::remove(path);
    std::filesystem::remove(folder);
    std::cout << "Settings: defaults, round-trip, replacement, Unicode paths, invalid input, and write failures passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
