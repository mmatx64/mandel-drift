#pragma once
#include <string>
#include <vector>

class SettingsOverlay {
 public:
  explicit SettingsOverlay(bool help = false);
  ~SettingsOverlay();
  SettingsOverlay(const SettingsOverlay&) = delete;
  SettingsOverlay& operator=(const SettingsOverlay&) = delete;

  void draw(int width, int height, float opacity,
            bool auto_palette, bool shuffle, bool endless_dive, bool bloom,
            bool music, bool muted, int volume, bool audio_available,
            float settings_progress = 0.0f, double fps = 0.0, double render_ms = 0.0);
  // Mouse coordinates and dimensions are in drawable pixels, with the origin at top left.
  int hit_test(float mouse_x, float mouse_y, int width, int height) const;
  int volume_at(float mouse_x, int width, int height) const;
  bool close_hit(float x, float y, int width, int height, float settings_progress = 0.0f) const;

 private:
  bool help_ = false;
  void update_texture(bool auto_palette, bool shuffle, bool endless_dive, bool bloom,
                      bool music, bool muted, int volume, bool audio_available);
  void update_performance(double fps, double render_ms);
  std::vector<unsigned char> performance_background_;
  std::wstring performance_text_;
  unsigned int program_ = 0;
  unsigned int vao_ = 0;
  unsigned int texture_ = 0;
  int rect_location_ = -1;
  int opacity_location_ = -1;
  bool values_valid_ = false;
  bool auto_palette_ = false;
  bool shuffle_ = false;
  bool endless_dive_ = false;
  bool bloom_ = true;
  bool music_ = true, muted_ = false, audio_available_ = true;
  int volume_ = 35;
};
