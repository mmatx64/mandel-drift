#pragma once

#include <filesystem>
#include <string>

struct AppSettings {
  bool auto_palette = false;
  bool shuffle = false;
  bool endless_dive = true;
  bool bloom = true;
  bool music = true;
  bool muted = false;
  int volume = 35;
  bool rotation = true;
  bool paused = false;
  bool menu_visible = false;
  int rotation_direction = 1;
  int palette = 0;
  int window_mode = 0; // 0 = windowed, 2 = fullscreen (preserves existing INI values).
};

AppSettings load_settings(const std::filesystem::path& path);
std::string settings_text(const AppSettings& settings);
bool save_settings(const std::filesystem::path& path, const AppSettings& settings);
