#define NOMINMAX
#include <windows.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>
#include <SDL3/SDL_opengl_glext.h>

#include "settings_overlay.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cwchar>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 1200;
constexpr int kHeight = 250;
constexpr int kPerformanceLeft = 215, kPerformanceTop = 19;
constexpr int kPerformanceWidth = 420, kPerformanceHeight = 32;
constexpr float kSliderLeft = 480, kSliderWidth = 590;

template <typename T> T gl_proc(const char* name) {
  auto* address = SDL_GL_GetProcAddress(name);
  if (!address) throw std::runtime_error(std::string("OpenGL function unavailable: ") + name);
  return reinterpret_cast<T>(address);
}

struct GL {
  PFNGLCREATESHADERPROC create_shader = gl_proc<PFNGLCREATESHADERPROC>("glCreateShader");
  PFNGLSHADERSOURCEPROC shader_source = gl_proc<PFNGLSHADERSOURCEPROC>("glShaderSource");
  PFNGLCOMPILESHADERPROC compile_shader = gl_proc<PFNGLCOMPILESHADERPROC>("glCompileShader");
  PFNGLGETSHADERIVPROC get_shader_iv = gl_proc<PFNGLGETSHADERIVPROC>("glGetShaderiv");
  PFNGLGETSHADERINFOLOGPROC get_shader_log = gl_proc<PFNGLGETSHADERINFOLOGPROC>("glGetShaderInfoLog");
  PFNGLDELETESHADERPROC delete_shader = gl_proc<PFNGLDELETESHADERPROC>("glDeleteShader");
  PFNGLCREATEPROGRAMPROC create_program = gl_proc<PFNGLCREATEPROGRAMPROC>("glCreateProgram");
  PFNGLATTACHSHADERPROC attach_shader = gl_proc<PFNGLATTACHSHADERPROC>("glAttachShader");
  PFNGLLINKPROGRAMPROC link_program = gl_proc<PFNGLLINKPROGRAMPROC>("glLinkProgram");
  PFNGLGETPROGRAMIVPROC get_program_iv = gl_proc<PFNGLGETPROGRAMIVPROC>("glGetProgramiv");
  PFNGLGETPROGRAMINFOLOGPROC get_program_log = gl_proc<PFNGLGETPROGRAMINFOLOGPROC>("glGetProgramInfoLog");
  PFNGLDELETEPROGRAMPROC delete_program = gl_proc<PFNGLDELETEPROGRAMPROC>("glDeleteProgram");
  PFNGLUSEPROGRAMPROC use_program = gl_proc<PFNGLUSEPROGRAMPROC>("glUseProgram");
  PFNGLGENVERTEXARRAYSPROC gen_vertex_arrays = gl_proc<PFNGLGENVERTEXARRAYSPROC>("glGenVertexArrays");
  PFNGLBINDVERTEXARRAYPROC bind_vertex_array = gl_proc<PFNGLBINDVERTEXARRAYPROC>("glBindVertexArray");
  PFNGLDELETEVERTEXARRAYSPROC delete_vertex_arrays = gl_proc<PFNGLDELETEVERTEXARRAYSPROC>("glDeleteVertexArrays");
  PFNGLACTIVETEXTUREPROC active_texture = gl_proc<PFNGLACTIVETEXTUREPROC>("glActiveTexture");
  PFNGLGETUNIFORMLOCATIONPROC get_uniform_location = gl_proc<PFNGLGETUNIFORMLOCATIONPROC>("glGetUniformLocation");
  PFNGLUNIFORM1IPROC uniform_1i = gl_proc<PFNGLUNIFORM1IPROC>("glUniform1i");
  PFNGLUNIFORM1FPROC uniform_1f = gl_proc<PFNGLUNIFORM1FPROC>("glUniform1f");
  PFNGLUNIFORM4FPROC uniform_4f = gl_proc<PFNGLUNIFORM4FPROC>("glUniform4f");
};

GL& gl() { static GL api; return api; }

constexpr const char* kVertex = R"(#version 330 core
out vec2 uv;
void main() {
  vec2 corners[3] = vec2[3](vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
  vec2 position = corners[gl_VertexID];
  uv = position * .5 + .5;
  gl_Position = vec4(position, 0., 1.);
})";

constexpr const char* kFragment = R"(#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D menu_texture;
uniform vec4 rect;
uniform float opacity;
void main() {
  vec2 point = (uv - rect.xy) / rect.zw;
  if (any(lessThan(point, vec2(0.))) || any(greaterThan(point, vec2(1.)))) discard;
  color = texture(menu_texture, vec2(point.x, 1. - point.y));
  color.a *= opacity;
})";

GLuint shader(GLenum kind, const char* source) {
  GLuint value = gl().create_shader(kind);
  gl().shader_source(value, 1, &source, nullptr);
  gl().compile_shader(value);
  GLint okay = GL_FALSE;
  gl().get_shader_iv(value, GL_COMPILE_STATUS, &okay);
  if (!okay) {
    std::array<char, 2048> log{};
    gl().get_shader_log(value, static_cast<GLsizei>(log.size()), nullptr, log.data());
    gl().delete_shader(value);
    throw std::runtime_error(std::string("Menu shader: ") + log.data());
  }
  return value;
}

GLuint menu_program() {
  GLuint vertex = shader(GL_VERTEX_SHADER, kVertex);
  GLuint fragment = shader(GL_FRAGMENT_SHADER, kFragment);
  GLuint program = gl().create_program();
  gl().attach_shader(program, vertex);
  gl().attach_shader(program, fragment);
  gl().link_program(program);
  gl().delete_shader(vertex);
  gl().delete_shader(fragment);
  GLint okay = GL_FALSE;
  gl().get_program_iv(program, GL_LINK_STATUS, &okay);
  if (!okay) {
    std::array<char, 2048> log{};
    gl().get_program_log(program, static_cast<GLsizei>(log.size()), nullptr, log.data());
    gl().delete_program(program);
    throw std::runtime_error(std::string("Menu program: ") + log.data());
  }
  return program;
}

struct Color { unsigned char r, g, b, a; };
using Image = std::vector<unsigned char>;

void pixel(Image& image, int x, int y, Color color, int image_width = kWidth) {
  const int image_height = static_cast<int>(image.size() / (image_width * 4));
  if (x < 0 || x >= image_width || y < 0 || y >= image_height || !color.a) return;
  const size_t index = (static_cast<size_t>(y) * image_width + x) * 4;
  const float source = color.a / 255.0f;
  const float destination = image[index + 3] / 255.0f;
  const float out = source + destination * (1.0f - source);
  if (out <= 0.0f) return;
  image[index] = static_cast<unsigned char>((color.r * source + image[index] * destination * (1.0f - source)) / out);
  image[index + 1] = static_cast<unsigned char>((color.g * source + image[index + 1] * destination * (1.0f - source)) / out);
  image[index + 2] = static_cast<unsigned char>((color.b * source + image[index + 2] * destination * (1.0f - source)) / out);
  image[index + 3] = static_cast<unsigned char>(out * 255.0f);
}

float rounded_distance(float x, float y, float left, float top,
                       float width, float height, float radius) {
  const float qx = std::abs(x - (left + width * .5f)) - width * .5f + radius;
  const float qy = std::abs(y - (top + height * .5f)) - height * .5f + radius;
  return std::hypot(std::max(qx, 0.0f), std::max(qy, 0.0f)) +
         std::min(std::max(qx, qy), 0.0f) - radius;
}

void rounded(Image& image, float left, float top, float width, float height,
             float radius, Color color) {
  const int x0 = std::max(0, static_cast<int>(left - 2));
  const int y0 = std::max(0, static_cast<int>(top - 2));
  const int x1 = std::min(kWidth, static_cast<int>(left + width + 2));
  const int y1 = std::min(kHeight, static_cast<int>(top + height + 2));
  for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) {
    const float coverage = std::clamp(.5f - rounded_distance(x + .5f, y + .5f,
        left, top, width, height, radius), 0.0f, 1.0f);
    pixel(image, x, y, {color.r, color.g, color.b,
                        static_cast<unsigned char>(color.a * coverage)});
  }
}

void text(Image& image, int x, int y, const wchar_t* words, int size, Color color,
          int image_width = kWidth) {
  constexpr int width = 1200;
  constexpr int height = 52;
  BITMAPINFO info{};
  info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  info.bmiHeader.biWidth = width;
  info.bmiHeader.biHeight = -height;
  info.bmiHeader.biPlanes = 1;
  info.bmiHeader.biBitCount = 32;
  info.bmiHeader.biCompression = BI_RGB;
  void* bits = nullptr;
  HDC dc = CreateCompatibleDC(nullptr);
  if (!dc) throw std::runtime_error("Could not create menu text context.");
  HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
  if (!bitmap) { DeleteDC(dc); throw std::runtime_error("Could not create menu text bitmap."); }
  HGDIOBJ old_bitmap = SelectObject(dc, bitmap);
  std::memset(bits, 0, static_cast<size_t>(width) * height * 4);
  HFONT font = CreateFontW(-size, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                           DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                           ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Segoe UI");
  HGDIOBJ old_font = SelectObject(dc, font);
  SetBkMode(dc, TRANSPARENT);
  SetTextColor(dc, RGB(255, 255, 255));
  TextOutW(dc, 0, 0, words, static_cast<int>(wcslen(words)));
  const auto* source = static_cast<const unsigned char*>(bits);
  const int image_height = static_cast<int>(image.size() / (image_width * 4));
  for (int py = 0; py < height && y + py < image_height; ++py) {
    for (int px = 0; px < width && x + px < image_width; ++px) {
      const size_t index = (static_cast<size_t>(py) * width + px) * 4;
      const unsigned char coverage = std::max({source[index], source[index + 1], source[index + 2]});
      pixel(image, x + px, y + py, {color.r, color.g, color.b,
            static_cast<unsigned char>(coverage * color.a / 255)}, image_width);
    }
  }
  SelectObject(dc, old_font);
  SelectObject(dc, old_bitmap);
  DeleteObject(font);
  DeleteObject(bitmap);
  DeleteDC(dc);
}

void toggle(Image& image, int x, bool enabled, int top = 89) {
  const Color accent = enabled ? Color{25, 239, 247, 255} : Color{247, 57, 180, 255};
  rounded(image, static_cast<float>(x - 5), static_cast<float>(top - 5), 104, 52, 26,
          {accent.r, accent.g, accent.b, 34});
  rounded(image, static_cast<float>(x), static_cast<float>(top), 94, 42, 21,
          {accent.r, accent.g, accent.b, 180});
  rounded(image, static_cast<float>(x + 2), static_cast<float>(top + 2), 90, 38, 19,
          enabled ? Color{7, 90, 114, 240} : Color{42, 15, 63, 245});
  rounded(image, static_cast<float>(enabled ? x + 56 : x + 6), static_cast<float>(top + 6), 30, 30, 15,
          enabled ? Color{231, 255, 255, 255} : Color{223, 198, 227, 255});
}

struct Placement { float left, top, width, height; };
Placement placement(int width, int height, float settings_progress = 0.0f) {
  const float margin = std::max(16.0f, height * .035f);
  const float bar_width = std::min({1500.0f, width * .68f,
                                    std::max(1.0f, (height - 2 * margin - 12) * kWidth / (2 * kHeight)),
                                    std::max(1.0f, width - 28.0f)});
  const float bar_height = bar_width * kHeight / kWidth;
  return {(width - bar_width) * .5f, height - bar_height - margin -
          settings_progress * (bar_height + std::max(10.0f, bar_width * .015f)),
          bar_width, bar_height};
}

}  // namespace

SettingsOverlay::SettingsOverlay(bool help) : help_(help), program_(menu_program()) {
  gl().gen_vertex_arrays(1, &vao_);
  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  rect_location_ = gl().get_uniform_location(program_, "rect");
  opacity_location_ = gl().get_uniform_location(program_, "opacity");
  gl().use_program(program_);
  gl().uniform_1i(gl().get_uniform_location(program_, "menu_texture"), 0);
}

SettingsOverlay::~SettingsOverlay() {
  if (texture_) glDeleteTextures(1, &texture_);
  if (vao_) gl().delete_vertex_arrays(1, &vao_);
  if (program_) gl().delete_program(program_);
}

void SettingsOverlay::update_texture(bool auto_palette, bool shuffle, bool endless_dive, bool bloom,
                                     bool music, bool muted, int volume, bool audio_available) {
  if (values_valid_ && auto_palette == auto_palette_ && shuffle == shuffle_ &&
      endless_dive == endless_dive_ && bloom == bloom_ && music == music_ &&
      muted == muted_ && volume == volume_ && audio_available == audio_available_) return;
  values_valid_ = true;
  auto_palette_ = auto_palette;
  shuffle_ = shuffle;
  endless_dive_ = endless_dive;
  bloom_ = bloom;
  music_ = music;
  muted_ = muted;
  volume_ = volume;
  audio_available_ = audio_available;
  Image image(static_cast<size_t>(kWidth) * kHeight * 4, 0);
  for (int y = 0; y < kHeight; ++y) for (int x = 0; x < kWidth; ++x) {
    const float distance = rounded_distance(x + .5f, y + .5f, 7, 7, 1186, kHeight - 14, 29);
    const float outer = std::clamp(9.0f - distance, 0.0f, 9.0f) / 9.0f;
    if (distance > 0 && distance < 9) {
      const float mix = x / static_cast<float>(kWidth);
      pixel(image, x, y, {static_cast<unsigned char>(29 + mix * 225),
                          static_cast<unsigned char>(226 - mix * 160), 244,
                          static_cast<unsigned char>(30 * outer * outer)});
    }
    if (distance <= .5f) {
      pixel(image, x, y, {20, 7, 47, 225});
      if (distance > -2.0f) {
        const float mix = x / static_cast<float>(kWidth);
        pixel(image, x, y, {static_cast<unsigned char>(22 + mix * 230),
                            static_cast<unsigned char>(226 - mix * 160), 246, 210});
      }
    }
  }
  rounded(image, 37, 57, 1126, 1, 0, {103, 69, 165, 110});
  if (help_) {
    const Color label{247, 240, 255, 255};
    const Color secondary{206, 196, 239, 255};
    text(image, 47, 19, L"K E Y B O A R D   S H O R T C U T S", 19, secondary);
    text(image, 967, 27, L"? / ESC TO CLOSE", 14, {208, 119, 216, 230});
    const wchar_t* keys[] = {L"M", L"F", L"S", L"A", L"O", L"P", L"B", L"E", L"?"};
    const wchar_t* labels[] = {L"Mute / unmute music", L"Windowed / fullscreen", L"Settings menu",
      L"Auto palette", L"Shuffle order", L"Next palette", L"Bloom", L"Endless dive", L"Help"};
    for (int i = 0; i < 9; ++i) {
      const int x = 47 + (i / 3) * 377, y = 72 + (i % 3) * 34;
      rounded(image, static_cast<float>(x), static_cast<float>(y), 32, 27, 5, {90, 67, 125, 255});
      rounded(image, static_cast<float>(x + 1), static_cast<float>(y + 1), 30, 25, 4, {33, 17, 60, 255});
      text(image, x + 8, y + 3, keys[i], 17, {157, 238, 250, 255});
      text(image, x + 44, y + 2, labels[i], 20, label);
    }
    rounded(image, 37, 181, 1126, 1, 0, {103, 69, 165, 110});
    text(image, 47, 191, L"Shift+P  Previous palette     - / =  Volume     Space  Pause camera", 16, secondary);
    text(image, 47, 215, L"R  Pause rotation     Shift+R  Reverse rotation     F12  Screenshot     Esc  Close panel / exit", 16, secondary);
  } else {
    rounded(image, 316, 76, 1, 65, 0, {155, 123, 205, 100});
    rounded(image, 600, 76, 1, 65, 0, {155, 123, 205, 100});
    rounded(image, 884, 76, 1, 65, 0, {155, 123, 205, 100});
    text(image, 47, 19, L"S E T T I N G S", 19, {206, 196, 239, 255});
    text(image, 1057, 27, L"S TO CLOSE", 14, {208, 119, 216, 230});
    text(image, 47, 95, L"Auto Palette", 21, {247, 240, 255, 255});
    text(image, 331, 95, L"Shuffle Order", 21, {247, 240, 255, 255});
    text(image, 615, 95, L"Endless Dive", 21, {247, 240, 255, 255});
    text(image, 910, 95, L"Bloom", 21, {247, 240, 255, 255});
    toggle(image, 202, auto_palette);
    toggle(image, 486, shuffle);
    toggle(image, 770, endless_dive);
    toggle(image, 1054, bloom);
    rounded(image, 37, 155, 1126, 1, 0, {103, 69, 165, 110});
    text(image, 47, 190, L"Music", 21, {247, 240, 255, 255});
    toggle(image, 202, music, 181);
    text(image, 331, 190, L"Volume", 21, {247, 240, 255, 255});
    const int displayed_volume = muted ? 0 : std::clamp(volume, 0, 100);
    const float slider_value = kSliderLeft + kSliderWidth * displayed_volume / 100.0f;
    rounded(image, kSliderLeft, 198, kSliderWidth, 8, 4, {82, 53, 115, 240});
    if (displayed_volume > 0)
      rounded(image, kSliderLeft, 198, slider_value - kSliderLeft, 8, 4, {25, 211, 232, 240});
    rounded(image, slider_value - 9, 193, 18, 18, 9, {231, 255, 255, 255});
    const std::wstring percent = std::to_wstring(displayed_volume) + L"%";
    text(image, 1100, 190, percent.c_str(), 21, {247, 240, 255, 255});
    if (music && !audio_available) text(image, 47, 220, L"Audio unavailable", 13, {247, 130, 190, 255});
  }
  gl().active_texture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kWidth, kHeight, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, image.data());
  if (!help_) {
    // Retain just the clean header patch. Changing telemetry never redraws
    // the panel, controls, or help texture.
    performance_background_.resize(kPerformanceWidth * kPerformanceHeight * 4);
    for (int y = 0; y < kPerformanceHeight; ++y)
      std::copy_n(image.data() + ((y + kPerformanceTop) * kWidth + kPerformanceLeft) * 4,
          kPerformanceWidth * 4, performance_background_.data() + y * kPerformanceWidth * 4);
    performance_text_.clear();
  }
}

void SettingsOverlay::update_performance(double fps, double render_ms) {
  if (help_) return;
  wchar_t label[80]{};
  if (std::isfinite(fps) && fps > 0 && std::isfinite(render_ms) && render_ms > 0)
    std::swprintf(label, std::size(label), L"|  %.0f FPS  |  render %.1f ms", fps, render_ms);
  else
    std::swprintf(label, std::size(label), L"|  -- FPS  |  render -- ms");
  if (performance_text_ == label) return;
  performance_text_ = label;
  Image patch = performance_background_;
  text(patch, 0, 0, label, 19, {157, 238, 250, 255}, kPerformanceWidth);
  gl().active_texture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, kPerformanceLeft, kPerformanceTop,
      kPerformanceWidth, kPerformanceHeight, GL_RGBA, GL_UNSIGNED_BYTE, patch.data());
}

void SettingsOverlay::draw(int width, int height, float opacity,
                           bool auto_palette, bool shuffle, bool endless_dive, bool bloom,
                           bool music, bool muted, int volume, bool audio_available, float settings_progress,
                           double fps, double render_ms) {
  if (opacity <= 0.0f) return;
  update_texture(auto_palette, shuffle, endless_dive, bloom, music, muted, volume, audio_available);
  update_performance(fps, render_ms);
  const Placement where = placement(width, height, help_ ? settings_progress : 0.0f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  gl().use_program(program_);
  gl().uniform_4f(rect_location_, where.left / width,
                  1.0f - (where.top + where.height) / height,
                  where.width / width, where.height / height);
  gl().uniform_1f(opacity_location_, opacity);
  gl().active_texture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_);
  gl().bind_vertex_array(vao_);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  glDisable(GL_BLEND);
}

bool SettingsOverlay::close_hit(float x, float y, int width, int height, float settings_progress) const {
  if (width <= 0 || height <= 0) return false;
  const Placement where = placement(width, height, help_ ? settings_progress : 0.0f);
  x = (x - where.left) * kWidth / where.width;
  y = (y - where.top) * kHeight / where.height;
  return x >= (help_ ? 950 : 1035) && x < 1170 && y >= 12 && y < 57;
}

int SettingsOverlay::hit_test(float mouse_x, float mouse_y, int width, int height) const {
  if (width <= 0 || height <= 0) return -1;
  const Placement where = placement(width, height);
  const float x = (mouse_x - where.left) * kWidth / where.width;
  const float y = (mouse_y - where.top) * kHeight / where.height;
  if (x < 32 || x >= 1170) return -1;
  if (y >= 171 && y < 234) {
    if (x < 317) return 4;
    if (x >= kSliderLeft - 20 && x <= kSliderLeft + kSliderWidth + 15) return 5;
    return -1;
  }
  if (y < 65 || y >= 151) return -1;
  if (x < 317) return 0;
  if (x < 601) return 1;
  if (x < 885) return 2;
  return 3;
}

int SettingsOverlay::volume_at(float mouse_x, int width, int height) const {
  const Placement where = placement(width, height);
  const float x = (mouse_x - where.left) * kWidth / where.width;
  return std::clamp(static_cast<int>(std::lround((x - kSliderLeft) * 100 / kSliderWidth)), 0, 100);
}
