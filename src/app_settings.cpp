#define NOMINMAX
#include <windows.h>

#include "app_settings.h"

#include <charconv>
#include <fstream>
#include <sstream>
#include <string_view>

namespace {
std::string_view trim(std::string_view value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == value.npos) return {};
  return value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1);
}
}

AppSettings load_settings(const std::filesystem::path& path) {
  AppSettings result;
  std::ifstream input(path);
  std::string line;
  bool app_section = false;
  while (std::getline(input, line)) {
    std::string_view value = trim(line);
    if (value.substr(0, 3) == "\xef\xbb\xbf") value.remove_prefix(3);
    value = trim(value.substr(0, value.find_first_of(";#")));
    if (value.empty()) continue;
    if (value.front() == '[') {
      app_section = value == "[MandelDrift]";
      continue;
    }
    if (!app_section) continue;
    const auto split = value.find('=');
    if (split == value.npos) continue;
    const auto key = trim(value.substr(0, split));
    value = trim(value.substr(split + 1));
    int number = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) continue;
    if (number == 0 || number == 1) {
      if (key == "auto_palette") result.auto_palette = number != 0;
      else if (key == "shuffle") result.shuffle = number != 0;
      else if (key == "endless_dive") result.endless_dive = number != 0;
      else if (key == "bloom") result.bloom = number != 0;
      else if (key == "music") result.music = number != 0;
      else if (key == "muted") result.muted = number != 0;
      else if (key == "rotation") result.rotation = number != 0;
      else if (key == "paused") result.paused = number != 0;
      else if (key == "menu_visible") result.menu_visible = number != 0;
    }
    if (key == "rotation_direction" && (number == -1 || number == 1)) result.rotation_direction = number;
    if (key == "palette" && number >= 0 && number < 8) result.palette = number;
    if (key == "volume" && number >= 0 && number <= 100) result.volume = number;
    if (key == "window_mode" && number >= 0 && number <= 2)
      result.window_mode = number == 2 ? 2 : 0; // Retire legacy borderless mode.
  }
  return result;
}

std::string settings_text(const AppSettings& settings) {
  std::ostringstream text;
  text << "; Mandel Drift preferences. Toggles: 0 = off, 1 = on.\n"
       << "; palette: 0..7; window_mode: 0 = windowed, 2 = fullscreen\n"
       << "[MandelDrift]\n"
       << "auto_palette=" << settings.auto_palette << '\n'
       << "shuffle=" << settings.shuffle << '\n'
       << "endless_dive=" << settings.endless_dive << '\n'
       << "bloom=" << settings.bloom << '\n'
       << "music=" << settings.music << '\n'
       << "muted=" << settings.muted << '\n'
       << "volume=" << settings.volume << '\n'
       << "rotation=" << settings.rotation << '\n'
       << "rotation_direction=" << settings.rotation_direction << '\n'
       << "paused=" << settings.paused << '\n'
       << "menu_visible=" << settings.menu_visible << '\n'
       << "palette=" << settings.palette << '\n'
       << "window_mode=" << settings.window_mode << '\n';
  return text.str();
}

bool save_settings(const std::filesystem::path& path, const AppSettings& settings) {
  // Replace only after the full file is written, preserving the old file on failure.
  auto temporary = path;
  temporary += L"." + std::to_wstring(GetCurrentProcessId()) + L".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) return false;
  output << settings_text(settings);
  output.close();
  if (output && MoveFileExW(temporary.c_str(), path.c_str(),
                           MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return true;
  std::error_code ignored;
  std::filesystem::remove(temporary, ignored);
  return false;
}
