#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_opengl.h>
#include <SDL3/SDL_opengl_glext.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include "mandelbrot.h"
#include "settings_overlay.h"
#include "app_settings.h"
#include "ambient_music.h"
#include "bloom_shaders.h"
#include "frame_budget.h"
#include "trip_effects.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr Uint64 kFrameIntervalNS = 16'666'667;  // At most 60 frames per second.

template <typename T>
T load_gl(const char* name) {
  auto proc = SDL_GL_GetProcAddress(name);
  if (!proc) throw std::runtime_error(std::string("OpenGL function unavailable: ") + name);
  return reinterpret_cast<T>(proc);
}

struct GLApi {
  PFNGLGENQUERIESPROC GenQueries = load_gl<PFNGLGENQUERIESPROC>("glGenQueries");
  PFNGLDELETEQUERIESPROC DeleteQueries = load_gl<PFNGLDELETEQUERIESPROC>("glDeleteQueries");
  PFNGLBEGINQUERYPROC BeginQuery = load_gl<PFNGLBEGINQUERYPROC>("glBeginQuery");
  PFNGLENDQUERYPROC EndQuery = load_gl<PFNGLENDQUERYPROC>("glEndQuery");
  PFNGLGETQUERYOBJECTIVPROC GetQueryObjectiv = load_gl<PFNGLGETQUERYOBJECTIVPROC>("glGetQueryObjectiv");
  PFNGLGETQUERYOBJECTUI64VPROC GetQueryObjectui64v = load_gl<PFNGLGETQUERYOBJECTUI64VPROC>("glGetQueryObjectui64v");
  PFNGLCREATESHADERPROC CreateShader = load_gl<PFNGLCREATESHADERPROC>("glCreateShader");
  PFNGLSHADERSOURCEPROC ShaderSource = load_gl<PFNGLSHADERSOURCEPROC>("glShaderSource");
  PFNGLCOMPILESHADERPROC CompileShader = load_gl<PFNGLCOMPILESHADERPROC>("glCompileShader");
  PFNGLGETSHADERIVPROC GetShaderiv = load_gl<PFNGLGETSHADERIVPROC>("glGetShaderiv");
  PFNGLGETSHADERINFOLOGPROC GetShaderInfoLog = load_gl<PFNGLGETSHADERINFOLOGPROC>("glGetShaderInfoLog");
  PFNGLDELETESHADERPROC DeleteShader = load_gl<PFNGLDELETESHADERPROC>("glDeleteShader");
  PFNGLCREATEPROGRAMPROC CreateProgram = load_gl<PFNGLCREATEPROGRAMPROC>("glCreateProgram");
  PFNGLATTACHSHADERPROC AttachShader = load_gl<PFNGLATTACHSHADERPROC>("glAttachShader");
  PFNGLLINKPROGRAMPROC LinkProgram = load_gl<PFNGLLINKPROGRAMPROC>("glLinkProgram");
  PFNGLGETPROGRAMIVPROC GetProgramiv = load_gl<PFNGLGETPROGRAMIVPROC>("glGetProgramiv");
  PFNGLGETPROGRAMINFOLOGPROC GetProgramInfoLog = load_gl<PFNGLGETPROGRAMINFOLOGPROC>("glGetProgramInfoLog");
  PFNGLDELETEPROGRAMPROC DeleteProgram = load_gl<PFNGLDELETEPROGRAMPROC>("glDeleteProgram");
  PFNGLUSEPROGRAMPROC UseProgram = load_gl<PFNGLUSEPROGRAMPROC>("glUseProgram");
  PFNGLGENVERTEXARRAYSPROC GenVertexArrays = load_gl<PFNGLGENVERTEXARRAYSPROC>("glGenVertexArrays");
  PFNGLBINDVERTEXARRAYPROC BindVertexArray = load_gl<PFNGLBINDVERTEXARRAYPROC>("glBindVertexArray");
  PFNGLDELETEVERTEXARRAYSPROC DeleteVertexArrays = load_gl<PFNGLDELETEVERTEXARRAYSPROC>("glDeleteVertexArrays");
  PFNGLGENBUFFERSPROC GenBuffers = load_gl<PFNGLGENBUFFERSPROC>("glGenBuffers");
  PFNGLBINDBUFFERPROC BindBuffer = load_gl<PFNGLBINDBUFFERPROC>("glBindBuffer");
  PFNGLBUFFERDATAPROC BufferData = load_gl<PFNGLBUFFERDATAPROC>("glBufferData");
  PFNGLDELETEBUFFERSPROC DeleteBuffers = load_gl<PFNGLDELETEBUFFERSPROC>("glDeleteBuffers");
  PFNGLACTIVETEXTUREPROC ActiveTexture = load_gl<PFNGLACTIVETEXTUREPROC>("glActiveTexture");
  PFNGLGETUNIFORMLOCATIONPROC GetUniformLocation = load_gl<PFNGLGETUNIFORMLOCATIONPROC>("glGetUniformLocation");
  PFNGLUNIFORM1IPROC Uniform1i = load_gl<PFNGLUNIFORM1IPROC>("glUniform1i");
  PFNGLUNIFORM1FPROC Uniform1f = load_gl<PFNGLUNIFORM1FPROC>("glUniform1f");
  PFNGLUNIFORM2FPROC Uniform2f = load_gl<PFNGLUNIFORM2FPROC>("glUniform2f");
  PFNGLUNIFORM4FPROC Uniform4f = load_gl<PFNGLUNIFORM4FPROC>("glUniform4f");
  PFNGLGENFRAMEBUFFERSPROC GenFramebuffers = load_gl<PFNGLGENFRAMEBUFFERSPROC>("glGenFramebuffers");
  PFNGLBINDFRAMEBUFFERPROC BindFramebuffer = load_gl<PFNGLBINDFRAMEBUFFERPROC>("glBindFramebuffer");
  PFNGLFRAMEBUFFERTEXTURE2DPROC FramebufferTexture2D = load_gl<PFNGLFRAMEBUFFERTEXTURE2DPROC>("glFramebufferTexture2D");
  PFNGLCHECKFRAMEBUFFERSTATUSPROC CheckFramebufferStatus = load_gl<PFNGLCHECKFRAMEBUFFERSTATUSPROC>("glCheckFramebufferStatus");
  PFNGLDELETEFRAMEBUFFERSPROC DeleteFramebuffers = load_gl<PFNGLDELETEFRAMEBUFFERSPROC>("glDeleteFramebuffers");
  PFNGLBLITFRAMEBUFFERPROC BlitFramebuffer = load_gl<PFNGLBLITFRAMEBUFFERPROC>("glBlitFramebuffer");
};

void check_cuda(cudaError_t result, const char* action) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(result));
  }
}

constexpr const char* kVertexShader = R"GLSL(#version 330 core
out vec2 uv;
void main() {
  vec2 corners[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  vec2 pos = corners[gl_VertexID];
  uv = pos * 0.5 + 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}
)GLSL";

constexpr const char* kTemporalFragmentShader = R"GLSL(#version 330 core
in vec2 uv;
out vec4 frag_color;
uniform sampler2D current_color;
uniform sampler2D history_color;
uniform vec4 previous_matrix;
uniform vec2 previous_translation;
uniform vec2 pixel_step;
uniform float history_weight;

void main() {
  vec3 now_color = texture(current_color, uv).rgb;
  vec2 centered = uv - vec2(0.5);
  vec2 old_uv = vec2(previous_matrix.x * centered.x +
                     previous_matrix.y * centered.y,
                     previous_matrix.z * centered.x +
                     previous_matrix.w * centered.y) + previous_translation + vec2(0.5);
  if (history_weight <= 0.0 || any(lessThan(old_uv, vec2(0.0))) ||
      any(greaterThan(old_uv, vec2(1.0)))) {
    frag_color = vec4(now_color, 1.0);
    return;
  }
  vec3 lower = now_color;
  vec3 upper = now_color;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      vec3 neighbor = texture(current_color, uv + vec2(x, y) * pixel_step).rgb;
      lower = min(lower, neighbor);
      upper = max(upper, neighbor);
    }
  }
  vec3 old_color = clamp(texture(history_color, old_uv).rgb, lower, upper);
  float disagreement = length(old_color - now_color);
  float motion = length((old_uv - uv) / pixel_step);
  float confidence = exp(-8.0 * disagreement) / (1.0 + 0.15 * motion);
  frag_color = vec4(mix(now_color, old_color, history_weight * confidence), 1.0);
}
)GLSL";

constexpr const char* kFragmentShader = R"GLSL(#version 330 core
in vec2 uv;
out vec4 frag_color;
uniform sampler2D escape_texture;
uniform vec2 escape_size;
uniform int palette_from;
uniform int palette_to;
uniform float palette_blend;
uniform float palette_time;
uniform vec4 trip; // wave amount, folding amount, symmetry count, rotation
uniform float musical_breath;

vec3 ramp(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
  float u = fract(t) * 3.0;
  if (u < 1.0) return mix(a, b, smoothstep(0.0, 1.0, u));
  if (u < 2.0) return mix(b, c, smoothstep(0.0, 1.0, u - 1.0));
  return mix(c, d, smoothstep(0.0, 1.0, u - 2.0));
}

vec3 palette(float t, int id) {
  if (id == 0) return ramp(t,
      vec3(0.025, 0.005, 0.11), vec3(0.38, 0.02, 0.58),
      vec3(1.0, 0.14, 0.52), vec3(0.12, 0.96, 0.96));
  if (id == 1) return ramp(t,
      vec3(0.035, 0.008, 0.09), vec3(0.45, 0.01, 0.42),
      vec3(1.0, 0.25, 0.29), vec3(1.0, 0.80, 0.35));
  if (id == 2) return ramp(t,
      vec3(0.008, 0.02, 0.12), vec3(0.11, 0.25, 0.66),
      vec3(0.35, 0.95, 1.0), vec3(0.76, 0.99, 0.94));
  if (id == 3) return ramp(t,
      vec3(0.015, 0.005, 0.10), vec3(0.26, 0.04, 0.62),
      vec3(0.72, 0.23, 0.98), vec3(1.0, 0.75, 0.98));
  if (id == 4) return ramp(t,
      vec3(0.05, 0.01, 0.16), vec3(0.28, 0.66, 0.78),
      vec3(1.0, 0.55, 0.67), vec3(1.0, 0.91, 0.64));
  if (id == 5) return ramp(t,
      vec3(0.008, 0.03, 0.10), vec3(0.13, 0.52, 0.25),
      vec3(0.73, 1.0, 0.17), vec3(0.99, 0.14, 0.48));
  if (id == 6) return ramp(t,
      vec3(0.0, 0.025, 0.10), vec3(0.04, 0.30, 0.65),
      vec3(0.0, 0.90, 0.90), vec3(0.86, 0.16, 0.68));
  return ramp(t,
      vec3(0.02, 0.0, 0.09), vec3(0.19, 0.08, 0.35),
      vec3(0.79, 0.28, 0.13), vec3(1.0, 0.91, 0.48));
}

vec3 color_escape(float escape) {
  if (escape < 0.0) {
    return vec3(0.004, 0.003, 0.018);
  } else {
    float phase = escape * 0.028 + palette_time * 0.035;
    if (trip.x > 0.0)
      phase += trip.x * (0.12 * sin(escape * 0.017 - palette_time * 0.24) +
                          0.045 * sin(escape * 0.043 + palette_time * 0.11));
    // The fade endpoints occupy most frames. Avoid evaluating a second
    // palette for each of the four color-first interpolation samples.
    float blend = smoothstep(0.0, 1.0, palette_blend);
    if (blend == 0.0) return palette(phase, palette_from);
    if (blend == 1.0 || palette_from == palette_to) return palette(phase, palette_to);
    vec3 color_a = palette(phase, palette_from);
    vec3 color_b = palette(phase, palette_to);
    return mix(color_a, color_b, blend);
  }
}

void main() {
  // Escape counts are nonlinear palette inputs; interpolate colors instead.
  ivec2 size = ivec2(escape_size);
  vec2 sample_uv = uv;
  if (trip.y > 0.0) {
    float aspect = escape_size.x / escape_size.y;
    vec2 centered = (uv - 0.5) * vec2(aspect, 1.0);
    float radius = length(centered);
    float angle = radius > 0.000001 ? atan(centered.y, centered.x) : 0.0;
    float sector = 6.28318530718 / trip.z;
    float folded = abs(mod(angle - trip.w + sector * 0.5, sector) - sector * 0.5) + trip.w;
    // Keep rotated corners inside the source image, avoiding clamped streaks.
    float fit = 0.48 * min(aspect, 1.0) / length(vec2(aspect, 1.0) * 0.5);
    vec2 target = radius * fit * vec2(cos(folded), sin(folded)) / vec2(aspect, 1.0);
    sample_uv = 0.5 + mix(centered / vec2(aspect, 1.0), target, trip.y);
  }
  vec2 pos = sample_uv * vec2(size) - 0.5;
  ivec2 p = ivec2(floor(pos));
  vec2 f = fract(pos);
  vec3 a = color_escape(texelFetch(escape_texture, clamp(p, ivec2(0), size - 1), 0).r);
  vec3 b = color_escape(texelFetch(escape_texture, clamp(p + ivec2(1,0), ivec2(0), size - 1), 0).r);
  vec3 c = color_escape(texelFetch(escape_texture, clamp(p + ivec2(0,1), ivec2(0), size - 1), 0).r);
  vec3 d = color_escape(texelFetch(escape_texture, clamp(p + ivec2(1,1), ivec2(0), size - 1), 0).r);
  frag_color = vec4(mix(mix(a,b,f.x), mix(c,d,f.x), f.y) * (1.0 + musical_breath * 0.16), 1.0);
}
)GLSL";

GLuint compile_shader(GLApi& gl, GLenum kind, const char* source) {
  GLuint shader = gl.CreateShader(kind);
  gl.ShaderSource(shader, 1, &source, nullptr);
  gl.CompileShader(shader);
  GLint success = GL_FALSE;
  gl.GetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (success == GL_FALSE) {
    std::array<char, 4096> log{};
    gl.GetShaderInfoLog(shader, static_cast<GLsizei>(log.size()), nullptr, log.data());
    gl.DeleteShader(shader);
    throw std::runtime_error(std::string("Shader compilation failed: ") + log.data());
  }
  return shader;
}

GLuint create_program(GLApi& gl, const char* fragment_source = kFragmentShader) {
  GLuint vertex = compile_shader(gl, GL_VERTEX_SHADER, kVertexShader);
  GLuint fragment = compile_shader(gl, GL_FRAGMENT_SHADER, fragment_source);
  GLuint program = gl.CreateProgram();
  gl.AttachShader(program, vertex);
  gl.AttachShader(program, fragment);
  gl.LinkProgram(program);
  gl.DeleteShader(vertex);
  gl.DeleteShader(fragment);
  GLint success = GL_FALSE;
  gl.GetProgramiv(program, GL_LINK_STATUS, &success);
  if (success == GL_FALSE) {
    std::array<char, 4096> log{};
    gl.GetProgramInfoLog(program, static_cast<GLsizei>(log.size()), nullptr, log.data());
    gl.DeleteProgram(program);
    throw std::runtime_error(std::string("Shader link failed: ") + log.data());
  }
  return program;
}

struct Camera {
  double x;
  double y;
  double span;
  double rotation = 0.0;
  JuliaMorph julia{};
};

Camera camera_at(double seconds) {
  // Some adjacent shots hold their zoom while tracking sideways across detail.
  constexpr std::array<Camera, 13> route{{
      {-0.65, 0.00, 3.2}, {-0.82, -0.11, 2.4},
      {-0.75, 0.10, 0.75}, {-0.76, 0.130, 0.11},
      {-0.72, 0.135, 0.11}, {-0.743643, 0.131826, 0.0045},
      {-0.7425, 0.1325, 0.0045}, {-0.7436, 0.1318, 0.09},
      {-0.55, 0.10, 2.4}, {-1.18, -0.08, 0.85},
      {-1.2506, 0.019, 0.10}, {-1.21, 0.035, 0.10},
      {-0.65, 0.00, 3.2},
  }};
  constexpr double segment_seconds = 9.0;
  constexpr int segments = static_cast<int>(route.size()) - 1;
  const double position = std::fmod(seconds / segment_seconds, static_cast<double>(segments));
  const int index = static_cast<int>(position);
  const double t = position - index;
  const Camera& a = route[index];
  const Camera& b = route[index + 1];

  const auto tangent = [&](int waypoint) {
    const int center = waypoint == segments ? 0 : waypoint;
    const Camera& current = route[center];
    const Camera& previous = route[(center + segments - 1) % segments];
    const Camera& next = route[(center + 1) % segments];
    double dx = 0.35 * (next.x - previous.x);
    double dy = 0.35 * (next.y - previous.y);
    const double length = std::hypot(dx, dy);
    const double limit = current.span * 0.18;
    if (length > limit) {
      dx *= limit / length;
      dy *= limit / length;
    }
    return std::array<double, 2>{dx, dy};
  };
  const auto ta = tangent(index);
  const auto tb = tangent(index + 1);
  const auto hermite = [t](double p0, double p1, double m0, double m1) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * p0 +
           (t3 - 2.0 * t2 + t) * m0 +
           (-2.0 * t3 + 3.0 * t2) * p1 +
           (t3 - t2) * m1;
  };
  const double ease = t * t * (3.0 - 2.0 * t);
  const double span = std::exp(std::log(a.span) +
                               (std::log(b.span) - std::log(a.span)) * ease);
  // The small orbit keeps the frame alive even where zoom changes direction.
  return {hermite(a.x, b.x, ta[0], tb[0]) + span * 0.035 * std::sin(seconds * 0.19),
          hermite(a.y, b.y, ta[1], tb[1]) + span * 0.025 * std::sin(seconds * 0.127 + 1.3),
          span};
}

// An endless sequence of long dives. Each one returns to the same wide view,
// so switching to a different detail site never jumps across the complex plane.
class DiveJourney {
 public:
  void advance(double seconds) {
    phase_ += seconds;
    elapsed_ += seconds;
    while (phase_ >= cycle_duration()) {
      phase_ -= cycle_duration();
      std::uniform_int_distribution<int> next_site(0, static_cast<int>(kSites.size()) - 2);
      int choice = next_site(rng_);
      if (choice >= site_) ++choice;
      site_ = choice;
      if (kSites[site_].min_span <= .00003) ++deep_visit_;
    }
  }

  Camera frame() const {
    const DiveSite& site = kSites[site_];
    const double dive_seconds = dive_duration(site);
    double span = site.min_span;
    if (phase_ < dive_seconds) {
      const double t = smooth(phase_ / dive_seconds);
      span = std::exp(std::log(kWideSpan) +
                      (std::log(site.min_span) - std::log(kWideSpan)) * t);
    } else if (phase_ >= dive_seconds + hold_duration()) {
      const double t = smooth((phase_ - dive_seconds - hold_duration()) / pullback_seconds());
      span = std::exp(std::log(site.min_span) +
                      (std::log(kWideSpan) - std::log(site.min_span)) * t);
    }

    const double retreat = std::pow(span / kWideSpan, 1.15);
    double x = site.x + (kWideX - site.x) * retreat;
    double y = site.y + (kWideY - site.y) * retreat;
    x += span * (0.035 * std::sin(elapsed_ * 0.19) +
                 0.025 * std::sin(elapsed_ * 0.073 + 0.4));
    y += span * 0.03 * std::sin(elapsed_ * 0.127 + 1.3);
    if (phase_ >= dive_seconds && phase_ < dive_seconds + hold_duration()) {
      const double hold = (phase_ - dive_seconds) / hold_duration();
      x += span * 0.12 * std::sin(hold * 3.141592653589793);
    }
    Camera camera{x, y, span};
    if (julia_hold())
      camera.julia = julia_morph(julia_hold_amount(phase_ - dive_seconds), span, site.x, site.y,
                                 phase_ - dive_seconds);
    // The wider Julia silhouette stays centered, independent of the local
    // camera orbit and sideways motion used at a deep Mandelbrot hold.
    const double centered = julia_center_amount(camera.julia.amount, span * camera.julia.seed_scale);
    camera.julia.seed_shift_x -= centered * (x - site.x) * camera.julia.seed_scale;
    camera.julia.seed_shift_y -= centered * (y - site.y) * camera.julia.seed_scale;
    return camera;
  }

  std::array<double, 2> reference_point() const {
    return {kSites[site_].x, kSites[site_].y};
  }

  double dive_seconds() const { return dive_duration(kSites[site_]); }
  // Match the dive's log-zoom speed instead of compressing deep returns into
  // a fixed 32 seconds, especially after the quiet Julia-to-Mandelbrot return.
  double pullback_seconds() const { return dive_seconds(); }
  bool julia_hold() const { return minimum_span() <= .00003 && deep_visit_ % 2 == 0; }
  double hold_duration() const { return julia_hold() ? kJuliaHoldSeconds : kHoldSeconds; }
  double cycle_duration() const { return dive_seconds() + hold_duration() + pullback_seconds(); }
  double minimum_span() const { return kSites[site_].min_span; }
  bool diving() const { return phase_ < dive_duration(kSites[site_]); }

 private:
  struct DiveSite {
    double x;
    double y;
    double min_span;
  };
  static constexpr std::array<DiveSite, 4> kSites{{
      {-0.743643887037151, 0.131825904205330, 0.00003},
      {-0.743643887037151, -0.131825904205330, 0.00003},
      {-1.2506, 0.019, 0.012},
      {-1.2506, -0.019, 0.012},
  }};
  static constexpr double kWideX = -0.65;
  static constexpr double kWideY = 0.0;
  static constexpr double kWideSpan = 3.2;
  static constexpr double kHoldSeconds = 10.0;

  static double dive_duration(const DiveSite& site) {
    return std::clamp(8.2 * std::log(kWideSpan / site.min_span), 75.0, 180.0);
  }
  static double smooth(double t) { return t * t * (3.0 - 2.0 * t); }

  std::mt19937 rng_{std::random_device{}()};
  int site_ = 0;
  unsigned int deep_visit_ = 0;
  double phase_ = 0.0;
  double elapsed_ = 0.0;
};

Camera blend_camera(const Camera& from, const Camera& to, double t) {
  const double ease = t * t * (3.0 - 2.0 * t);
  Camera camera{from.x + (to.x - from.x) * ease,
                from.y + (to.y - from.y) * ease,
                std::exp(std::log(from.span) +
                         (std::log(to.span) - std::log(from.span)) * ease)};
  const double amount = from.julia.amount + (to.julia.amount - from.julia.amount) * ease;
  if (amount > 0) {
    const JuliaMorph& anchor = from.julia.amount > 0 ? from.julia : to.julia;
    const double seed_span = std::exp(std::log(from.span * from.julia.seed_scale) +
        (std::log(to.span * to.julia.seed_scale) - std::log(from.span * from.julia.seed_scale)) * ease);
    camera.julia = {amount, seed_span / camera.span, anchor.x, anchor.y};
    // Interpolate absolute seed centers, then express them relative to the
    // shared anchor. This also makes repeated mode toggles continuous.
    const double seed_from_x = from.julia.amount > 0 ? from.julia.x + from.julia.seed_shift_x +
        (from.x - from.julia.x) * from.julia.seed_scale : from.x;
    const double seed_from_y = from.julia.amount > 0 ? from.julia.y + from.julia.seed_shift_y +
        (from.y - from.julia.y) * from.julia.seed_scale : from.y;
    const double seed_to_x = to.julia.amount > 0 ? to.julia.x + to.julia.seed_shift_x +
        (to.x - to.julia.x) * to.julia.seed_scale : to.x;
    const double seed_to_y = to.julia.amount > 0 ? to.julia.y + to.julia.seed_shift_y +
        (to.y - to.julia.y) * to.julia.seed_scale : to.y;
    camera.julia.seed_shift_x = seed_from_x + (seed_to_x - seed_from_x) * ease -
        anchor.x - (camera.x - anchor.x) * camera.julia.seed_scale;
    camera.julia.seed_shift_y = seed_from_y + (seed_to_y - seed_from_y) * ease -
        anchor.y - (camera.y - anchor.y) * camera.julia.seed_scale;
    camera.julia.parameter_shift_x = from.julia.parameter_shift_x +
        (to.julia.parameter_shift_x - from.julia.parameter_shift_x) * ease;
    camera.julia.parameter_shift_y = from.julia.parameter_shift_y +
        (to.julia.parameter_shift_y - from.julia.parameter_shift_y) * ease;
  }
  return camera;
}

// Travel between sites only after pulling back far enough to see the distance.
// Each leg has a bounded log-zoom/pan speed, independent of starting depth.
class CameraTransition {
 public:
  void start(const Camera& from, const Camera& to) {
    points_[0] = from;
    points_[3] = to;
    const double distance = std::hypot(to.x - from.x, to.y - from.y);
    const double wide = std::max({from.span, to.span, distance * 2.0});
    points_[1] = {from.x, from.y, wide};
    points_[2] = {to.x, to.y, wide};
    durations_[0] = std::max(0.5, 1.5 * std::log(wide / from.span) / 0.65);
    durations_[1] = std::max(0.5, 1.5 * distance / wide / 0.18);
    durations_[2] = std::max(0.5, 1.5 * std::log(wide / to.span) / 0.65);
    elapsed_ = 0.0;
    active_ = true;
  }
  bool active() const { return active_; }
  Camera advance(double dt) {
    elapsed_ += dt;
    double time = elapsed_;
    for (int leg = 0; leg < 3; ++leg) {
      if (time < durations_[leg])
        return blend_camera(points_[leg], points_[leg + 1], time / durations_[leg]);
      time -= durations_[leg];
    }
    active_ = false;
    return points_[3];
  }
 private:
  std::array<Camera, 4> points_{};
  std::array<double, 3> durations_{};
  double elapsed_ = 0.0;
  bool active_ = false;
};

class Renderer {
 public:
  Renderer() : program_(create_program(gl_)),
               temporal_program_(create_program(gl_, kTemporalFragmentShader)),
               bloom_extract_program_(create_program(gl_, kBloomExtractShader)),
               bloom_blur_program_(create_program(gl_, kBloomBlurShader)),
               bloom_composite_program_(create_program(gl_, kBloomCompositeShader)) {
    unsigned int count = 0;
    int devices[8]{};
    check_cuda(cudaGLGetDevices(&count, devices, 8, cudaGLDeviceListAll),
               "Find CUDA device for OpenGL window");
    if (count == 0) throw std::runtime_error("OpenGL is not running on an NVIDIA CUDA device.");
    check_cuda(cudaSetDevice(devices[0]), "Select NVIDIA GPU");
    for (auto& timing : frame_timings_) {
      check_cuda(cudaEventCreate(&timing.start), "Create CUDA start timer");
      check_cuda(cudaEventCreate(&timing.stop), "Create CUDA stop timer");
    }
    gl_.GenQueries(4, post_queries_);

    gl_.GenVertexArrays(1, &vao_);
    gl_.UseProgram(program_);
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "escape_texture"), 0);
    palette_from_location_ = gl_.GetUniformLocation(program_, "palette_from");
    palette_to_location_ = gl_.GetUniformLocation(program_, "palette_to");
    palette_blend_location_ = gl_.GetUniformLocation(program_, "palette_blend");
    time_location_ = gl_.GetUniformLocation(program_, "palette_time");
    trip_location_ = gl_.GetUniformLocation(program_, "trip");
    breath_location_ = gl_.GetUniformLocation(program_, "musical_breath");
    escape_size_location_ = gl_.GetUniformLocation(program_, "escape_size");
    gl_.UseProgram(temporal_program_);
    gl_.Uniform1i(gl_.GetUniformLocation(temporal_program_, "current_color"), 0);
    gl_.Uniform1i(gl_.GetUniformLocation(temporal_program_, "history_color"), 1);
    matrix_location_ = gl_.GetUniformLocation(temporal_program_, "previous_matrix");
    translation_location_ = gl_.GetUniformLocation(temporal_program_, "previous_translation");
    pixel_step_location_ = gl_.GetUniformLocation(temporal_program_, "pixel_step");
    history_weight_location_ = gl_.GetUniformLocation(temporal_program_, "history_weight");
    gl_.UseProgram(bloom_extract_program_);
    gl_.Uniform1i(gl_.GetUniformLocation(bloom_extract_program_, "source_color"), 0);
    gl_.UseProgram(bloom_blur_program_);
    gl_.Uniform1i(gl_.GetUniformLocation(bloom_blur_program_, "source_color"), 0);
    bloom_direction_location_ = gl_.GetUniformLocation(bloom_blur_program_, "direction");
    gl_.UseProgram(bloom_composite_program_);
    gl_.Uniform1i(gl_.GetUniformLocation(bloom_composite_program_, "scene_color"), 0);
    gl_.Uniform1i(gl_.GetUniformLocation(bloom_composite_program_, "glow_color"), 1);
  }

  ~Renderer() {
    gl_.DeleteQueries(4, post_queries_);
    release_image();
    release_history();
    if (reference_orbit_) cudaFree(reference_orbit_);
    if (fold_detail_) cudaFree(fold_detail_);
    if (vao_) gl_.DeleteVertexArrays(1, &vao_);
    if (program_) gl_.DeleteProgram(program_);
    if (temporal_program_) gl_.DeleteProgram(temporal_program_);
    if (bloom_extract_program_) gl_.DeleteProgram(bloom_extract_program_);
    if (bloom_blur_program_) gl_.DeleteProgram(bloom_blur_program_);
    if (bloom_composite_program_) gl_.DeleteProgram(bloom_composite_program_);
    for (auto& timing : frame_timings_) {
      if (timing.start) cudaEventDestroy(timing.start);
      if (timing.stop) cudaEventDestroy(timing.stop);
    }
  }

  void resize(int width, int height, int capacity_width, int capacity_height) {
    // Keep the CUDA registration and texture storage across quality changes.
    // Only a change of output dimensions needs a new allocation.
    if (capacity_width == capacity_width_ && capacity_height == capacity_height_) {
      width_ = width;
      height_ = height;
      return;
    }
    release_image();
    width_ = width;
    height_ = height;
    capacity_width_ = capacity_width;
    capacity_height_ = capacity_height;
    gl_.GenBuffers(1, &pbo_);
    gl_.BindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    gl_.BufferData(GL_PIXEL_UNPACK_BUFFER,
                   static_cast<GLsizeiptr>(capacity_width) * capacity_height * sizeof(float),
                   nullptr, GL_STREAM_DRAW);
    gl_.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    check_cuda(cudaGraphicsGLRegisterBuffer(&cuda_resource_, pbo_,
                                            cudaGraphicsRegisterFlagsWriteDiscard),
               "Share image buffer with CUDA");

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, capacity_width, capacity_height, 0, GL_RED, GL_FLOAT, nullptr);
    ensure_history(capacity_width, capacity_height);
  }

  float draw(const Camera& camera, int palette_from, int palette_to,
             float palette_blend, float palette_time,
             int output_width, int output_height,
             const std::array<double, 2>& journey_reference, bool bloom_enabled,
             TripEffects effects, float fold_angle, float musical_breath, bool visual_probe = false) {
    const Uint64 draw_start = SDL_GetTicksNS();
    timing_sample_ready_ = false;
    const bool morphing = camera.julia.amount > 0;
    const bool wide_julia = morphing && camera.span * camera.julia.seed_scale >= kJuliaFloatSpan;
    const bool deep_zoom = camera.span < 0.0045 || morphing;
    // Release unused scratch between dives, but avoid allocation churn near
    // the precision switch. cudaFree orders release after outstanding work.
    update_queue_retention(morphing ? std::min(camera.span, .0045) : camera.span, draw_start);
    const int iterations = std::clamp(
        260 + static_cast<int>(58.0 * std::log2(3.2 / camera.span)),
        260, deep_zoom ? 2200 : 850);
    int reference_length = 0;
    double reference_offset_x = 0.0;
    double reference_offset_y = 0.0;
    if (deep_zoom && !wide_julia) {
      if (!reference_orbit_) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&reference_orbit_),
                              2201 * sizeof(float2)), "Allocate reference orbit");
      }
      const bool site_nearby = std::hypot(camera.x - journey_reference[0],
                                         camera.y - journey_reference[1]) < camera.span * 0.45;
      const double reference_x = morphing ? camera.julia.x : (site_nearby ? journey_reference[0] : camera.x);
      const double reference_y = morphing ? camera.julia.y : (site_nearby ? journey_reference[1] : camera.y);
      reference_offset_x = reference_x - camera.x;
      reference_offset_y = reference_y - camera.y;
      const double seed_shift_x = morphing ? camera.julia.seed_shift_x : 0;
      const double seed_shift_y = morphing ? camera.julia.seed_shift_y : 0;
      const double parameter_shift_x = morphing ? camera.julia.parameter_shift_x : 0;
      const double parameter_shift_y = morphing ? camera.julia.parameter_shift_y : 0;
      if (!cached_reference_valid_ || reference_x != cached_reference_x_ ||
          reference_y != cached_reference_y_ || seed_shift_x != cached_seed_shift_x_ ||
          seed_shift_y != cached_seed_shift_y_ || parameter_shift_x != cached_parameter_shift_x_ ||
          parameter_shift_y != cached_parameter_shift_y_) {
        std::vector<float2> reference;
        reference.reserve(2201);
        double zx = 0.0, zy = 0.0;
        for (int index = 0; index <= 2200; ++index) {
          reference.push_back(float2{static_cast<float>(zx), static_cast<float>(zy)});
          if (zx * zx + zy * zy > 256.0) break;
          if (index == 0) {
            zx = reference_x + seed_shift_x;
            zy = reference_y + seed_shift_y;
          } else {
            const double next_x = zx * zx - zy * zy + reference_x + parameter_shift_x;
            zy = 2.0 * zx * zy + reference_y + parameter_shift_y;
            zx = next_x;
          }
        }
        cached_reference_length_ = static_cast<int>(reference.size());
        check_cuda(cudaMemcpy(reference_orbit_, reference.data(),
                              reference.size() * sizeof(float2), cudaMemcpyHostToDevice),
                   "Upload reference orbit");
        cached_reference_x_ = reference_x;
        cached_reference_y_ = reference_y;
        cached_seed_shift_x_ = seed_shift_x;
        cached_seed_shift_y_ = seed_shift_y;
        cached_parameter_shift_x_ = parameter_shift_x;
        cached_parameter_shift_y_ = parameter_shift_y;
        cached_reference_valid_ = true;
      }
      reference_length = cached_reference_length_;
      // Shallow scenes and short references never use the queued fallback.
      // Retain full capacity across quality changes and short shallow visits.
      if (reference_length > iterations && !fallback_queue_.pixels) {
        allocate_fallback_queue();
      }
    }
    if (effects.folding > 0 && !fold_detail_)
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&fold_detail_), sizeof(float4)), "Allocate fold detail summary");
    check_cuda(cudaGraphicsMapResources(1, &cuda_resource_), "Map shared image");
    float* device_pixels = nullptr;
    size_t bytes = 0;
    cudaError_t result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&device_pixels), &bytes, cuda_resource_);
    if (result != cudaSuccess || bytes < static_cast<size_t>(width_) * height_ * sizeof(float)) {
      cudaGraphicsUnmapResources(1, &cuda_resource_);
      check_cuda(result, "Get shared image pointer");
      throw std::runtime_error("Shared image buffer is too small.");
    }

    // Mapping orders earlier OpenGL work before this frame's CUDA work.
    // Collect completed CUDA/GL pairs without ever waiting for telemetry.
    collect_frame_timings(output_width, output_height);
    const bool timing = query_count_ < 4;
    auto& frame_timing = frame_timings_[query_write_];
    if (timing) check_cuda(cudaEventRecord(frame_timing.start), "Start CUDA timer");
    result = morphing
        ? launch_julia_morph(device_pixels, width_, height_, camera.x, camera.y, camera.span,
                             camera.rotation, camera.julia, reference_orbit_, reference_length,
                             iterations, reference_length > iterations ? fallback_queue_ : MandelbrotFallbackQueue{})
        : deep_zoom ? launch_mandelbrot_perturbed(device_pixels, width_, height_,
                                     camera.x, camera.y, camera.span, camera.rotation,
                                     reference_offset_x, reference_offset_y,
                                     reference_orbit_, reference_length, iterations, nullptr,
                                     reference_length > iterations ? fallback_queue_ : MandelbrotFallbackQueue{})
        : launch_mandelbrot(device_pixels, width_, height_,
                            static_cast<float>(camera.x),
                            static_cast<float>(camera.y),
                            static_cast<float>(camera.span),
                            static_cast<float>(camera.rotation), iterations);
    const bool probe_fold = effects.folding > 0 &&
        (!fold_detail_ || draw_start - fold_probe_time_ >= 500'000'000ULL);
    if (effects.folding == 0) {
      fold_detail_weight_ = fold_detail_target_ = 0;
      fold_probe_time_ = 0; // A later fold must establish fresh scene detail.
    }
    if (probe_fold) {
      check_cuda(measure_fold_detail(device_pixels, width_, height_, effects.folds, fold_angle, fold_detail_),
                 "Measure kaleidoscope detail");
      fold_probe_time_ = draw_start;
    }
    cudaError_t timer_result = timing ? cudaEventRecord(frame_timing.stop) : cudaSuccess;
    cudaError_t unmap_result = cudaGraphicsUnmapResources(1, &cuda_resource_);
    check_cuda(result, "Render Mandelbrot frame");
    check_cuda(timer_result, "Stop CUDA timer");
    check_cuda(unmap_result, "Unmap shared image");
    // Unmapping orders these CUDA writes before subsequent OpenGL use.
    // Only the infrequent fold probe needs a CPU readback of this frame.
    if (probe_fold) {
      float4 detail;
      check_cuda(cudaMemcpy(&detail, fold_detail_, sizeof(detail), cudaMemcpyDeviceToHost), "Read fold detail summary");
      fold_detail_target_ = folding_detail_weight(detail.x, detail.y, detail.z, detail.w);
    }
    const double effect_dt = previous_effect_time_ ? std::min((draw_start - previous_effect_time_) / 1.0e9, .1) : 0;
    previous_effect_time_ = draw_start;
    fold_detail_weight_ += (fold_detail_target_ - fold_detail_weight_) * static_cast<float>(1 - std::exp(-effect_dt / .6));
    if (!visual_probe) effects.folding *= fold_detail_weight_;
    if (effects.folding < .001f) effects.folding = 0;
    if (timing) {
      frame_timing.pre_post_ms = (SDL_GetTicksNS() - draw_start) / 1.0e6;
      frame_timing.width = width_;
      frame_timing.height = height_;
      frame_timing.output_width = output_width;
      frame_timing.output_height = output_height;
      gl_.BeginQuery(GL_TIME_ELAPSED, post_queries_[query_write_]);
    }

    gl_.ActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    gl_.BindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RED, GL_FLOAT, nullptr);
    gl_.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    ensure_history(output_width, output_height);
    gl_.ActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    gl_.BindFramebuffer(GL_FRAMEBUFFER, current_fbo_);
    glViewport(0, 0, output_width, output_height);
    glClear(GL_COLOR_BUFFER_BIT);
    gl_.UseProgram(program_);
    gl_.Uniform1i(palette_from_location_, palette_from);
    gl_.Uniform1i(palette_to_location_, palette_to);
    gl_.Uniform1f(palette_blend_location_, palette_blend);
    gl_.Uniform1f(time_location_, palette_time);
    gl_.Uniform4f(trip_location_, effects.waves, effects.folding, static_cast<float>(effects.folds), fold_angle);
    gl_.Uniform1f(breath_location_, musical_breath);
    gl_.Uniform2f(escape_size_location_, static_cast<float>(width_), static_cast<float>(height_));
    gl_.BindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    const int write_index = 1 - history_index_;
    gl_.BindFramebuffer(GL_FRAMEBUFFER, history_fbo_[write_index]);
    gl_.UseProgram(temporal_program_);
    gl_.ActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, current_color_);
    gl_.ActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, history_color_[history_index_]);
    float weight = history_valid_ ? 0.78f : 0.0f;
    // A changing nonlinear fold cannot use the camera's affine reprojection.
    // Suppress stale history through folding and its first normal frame.
    if (effects.folding > 0 || previous_folding_ > 0) weight = 0.0f;
    // Parameter morphs do not follow affine camera motion. Frozen Julia views
    // can accumulate history again; changing parameters and both edges cannot.
    if (camera.julia.amount != previous_camera_.julia.amount ||
        camera.julia.seed_scale != previous_camera_.julia.seed_scale ||
        camera.julia.seed_shift_x != previous_camera_.julia.seed_shift_x ||
        camera.julia.seed_shift_y != previous_camera_.julia.seed_shift_y ||
        camera.julia.parameter_shift_x != previous_camera_.julia.parameter_shift_x ||
        camera.julia.parameter_shift_y != previous_camera_.julia.parameter_shift_y) weight = 0.0f;
    double m00 = 1.0, m01 = 0.0, m10 = 0.0, m11 = 1.0;
    double tx = 0.0, ty = 0.0;
    if (history_valid_) {
      const double aspect = output_height / static_cast<double>(output_width);
      const double ratio = camera.span / previous_camera_.span;
      const double delta = camera.rotation - previous_camera_.rotation;
      const double cosine = std::cos(delta);
      const double sine = std::sin(delta);
      m00 = ratio * cosine;
      m01 = -ratio * sine * aspect;
      m10 = ratio * sine / aspect;
      m11 = ratio * cosine;
      const double dx = camera.x - previous_camera_.x;
      const double dy = camera.y - previous_camera_.y;
      const double old_cosine = std::cos(previous_camera_.rotation);
      const double old_sine = std::sin(previous_camera_.rotation);
      tx = (old_cosine * dx + old_sine * dy) / previous_camera_.span;
      ty = (-old_sine * dx + old_cosine * dy) / (previous_camera_.span * aspect);
      if (ratio < 0.8 || ratio > 1.25 || std::abs(tx) > 0.25 ||
          std::abs(ty) > 0.25) weight = 0.0f;
    }
    gl_.Uniform4f(matrix_location_, static_cast<float>(m00), static_cast<float>(m01),
                  static_cast<float>(m10), static_cast<float>(m11));
    gl_.Uniform2f(translation_location_, static_cast<float>(tx), static_cast<float>(ty));
    gl_.Uniform2f(pixel_step_location_, 1.0f / output_width, 1.0f / output_height);
    gl_.Uniform1f(history_weight_location_, weight);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    history_index_ = write_index;
    history_valid_ = true;
    previous_camera_ = camera;
    previous_folding_ = effects.folding;
    present(bloom_enabled);
    if (timing) {
      gl_.EndQuery(GL_TIME_ELAPSED);
      frame_timing.submit_ms = (SDL_GetTicksNS() - draw_start) / 1.0e6;
      query_write_ = (query_write_ + 1) % 4;
      ++query_count_;
    }
    return kernel_ms_;
  }

  double work_ms() const { return work_ms_; }
  double post_ms() const { return post_ms_; }
  bool timing_sample_ready() const { return timing_sample_ready_; }
  float folding_amount() const { return previous_folding_; }

  // Bloom stays outside temporal history, avoiding glow accumulation and trails.
  // Re-presenting also allows an exact same-frame comparison in the smoke test.
  void present(bool bloom_enabled) {
    gl_.BindVertexArray(vao_);
    if (bloom_enabled) {
      glViewport(0, 0, bloom_width_, bloom_height_);
      gl_.BindFramebuffer(GL_FRAMEBUFFER, bloom_fbo_[0]);
      gl_.UseProgram(bloom_extract_program_);
      gl_.ActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, history_color_[history_index_]);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      gl_.UseProgram(bloom_blur_program_);
      for (int pass = 0; pass < 2; ++pass) {
        gl_.BindFramebuffer(GL_FRAMEBUFFER, bloom_fbo_[1 - pass]);
        glBindTexture(GL_TEXTURE_2D, bloom_color_[pass]);
        gl_.Uniform2f(bloom_direction_location_, pass == 0 ? 1.0f / bloom_width_ : 0.0f,
                      pass == 1 ? 1.0f / bloom_height_ : 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
      }
      gl_.BindFramebuffer(GL_FRAMEBUFFER, 0);
      glViewport(0, 0, history_width_, history_height_);
      gl_.UseProgram(bloom_composite_program_);
      glBindTexture(GL_TEXTURE_2D, history_color_[history_index_]);
      gl_.ActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, bloom_color_[0]);
      glDrawArrays(GL_TRIANGLES, 0, 3);
    } else {
      gl_.BindFramebuffer(GL_READ_FRAMEBUFFER, history_fbo_[history_index_]);
      gl_.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
      gl_.BlitFramebuffer(0, 0, history_width_, history_height_,
                          0, 0, history_width_, history_height_, GL_COLOR_BUFFER_BIT, GL_NEAREST);
      gl_.BindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    gl_.ActiveTexture(GL_TEXTURE0);
  }

  void check_deep_frame(const Camera& camera, double maximum_span) const {
    if (camera.span >= maximum_span) {
      throw std::runtime_error("Deep zoom smoke frame did not reach its target depth.");
    }
    std::vector<float> pixels(static_cast<size_t>(capacity_width_) * capacity_height_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, pixels.data());
    int escaped = 0;
    float minimum = 1e9f, maximum = -1e9f;
    const size_t active_pixels = static_cast<size_t>(width_) * height_;
    for (size_t index = 0; index < active_pixels; index += std::max<size_t>(1, active_pixels / 4096)) {
      const float value = pixels[(index / width_) * capacity_width_ + index % width_];
      if (!std::isfinite(value)) throw std::runtime_error("Deep zoom contains non-finite pixels.");
      if (value >= 0.0f) {
        ++escaped;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
      }
    }
    if (escaped < 10 || maximum - minimum < 10.0f) {
      throw std::runtime_error("Deep zoom frame lost its fractal detail.");
    }
  }

  void check_fallback_queue_lifecycle() {
    if (!fallback_queue_.pixels || !fallback_queue_.count)
      throw std::runtime_error("Deep frame did not allocate its fallback queue.");
    queue_idle_since_ = 0;
    update_queue_retention(0.009, 1'000'000'000ULL);
    update_queue_retention(0.006, 2'500'000'000ULL); // Threshold visit cancels release.
    update_queue_retention(0.009, 3'000'000'000ULL);
    update_queue_retention(0.009, 4'999'999'999ULL);
    if (!fallback_queue_.pixels || !fallback_queue_.count)
      throw std::runtime_error("Fallback queue released before the idle grace period.");
    update_queue_retention(0.009, 5'000'000'000ULL);
    if (fallback_queue_.pixels || fallback_queue_.count || queue_idle_since_)
      throw std::runtime_error("Idle fallback queue was not released.");
    allocate_fallback_queue(); // The next deep frame must reuse the new allocation.
  }

 private:
  void collect_frame_timings(int output_width, int output_height) {
    while (query_count_ > 0) {
      const auto& timing = frame_timings_[query_read_];
      const auto result = cudaEventQuery(timing.stop);
      if (result == cudaErrorNotReady) break;
      check_cuda(result, "Query CUDA timer");
      GLint ready = GL_FALSE;
      gl_.GetQueryObjectiv(post_queries_[query_read_], GL_QUERY_RESULT_AVAILABLE, &ready);
      if (!ready) break;
      float kernel_ms = 0;
      check_cuda(cudaEventElapsedTime(&kernel_ms, timing.start, timing.stop), "Read CUDA timer");
      GLuint64 elapsed = 0;
      gl_.GetQueryObjectui64v(post_queries_[query_read_], GL_QUERY_RESULT, &elapsed);
      // A delayed sample from another resolution must not steer today's scale.
      // Pair GPU stages from the same frame, including execution that happened
      // after the CPU finished submitting work. Avoid counting a CPU wait twice.
      if (timing.width == width_ && timing.height == height_ &&
          timing.output_width == output_width && timing.output_height == output_height) {
        kernel_ms_ = kernel_ms;
        post_ms_ = elapsed / 1.0e6;
        work_ms_ = std::max(timing.submit_ms,
            std::max(timing.pre_post_ms, static_cast<double>(kernel_ms)) + post_ms_);
        timing_sample_ready_ = true;
      }
      query_read_ = (query_read_ + 1) % 4;
      --query_count_;
    }
  }

  void allocate_fallback_queue() {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&fallback_queue_.pixels),
        static_cast<size_t>(capacity_width_) * capacity_height_ * sizeof(int)),
        "Allocate fallback pixel queue");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&fallback_queue_.count), sizeof(unsigned int)),
        "Allocate fallback counter");
  }

  void update_queue_retention(double span, Uint64 now) {
    if (fallback_queue_.pixels && span >= 0.009) {
      if (!queue_idle_since_) queue_idle_since_ = now;
      if (now - queue_idle_since_ >= 2'000'000'000ULL) release_fallback_queue();
    } else {
      queue_idle_since_ = 0;
    }
  }
  void create_color_target(GLuint& framebuffer, GLuint& texture,
                           int width, int height) {
    gl_.ActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    gl_.GenFramebuffers(1, &framebuffer);
    gl_.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    gl_.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texture, 0);
    if (gl_.CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      throw std::runtime_error("Could not create temporal color target.");
    }
  }

  void release_history() {
    gl_.BindFramebuffer(GL_FRAMEBUFFER, 0);
    if (current_fbo_) gl_.DeleteFramebuffers(1, &current_fbo_);
    if (current_color_) glDeleteTextures(1, &current_color_);
    for (int index = 0; index < 2; ++index) {
      if (history_fbo_[index]) gl_.DeleteFramebuffers(1, &history_fbo_[index]);
      if (history_color_[index]) glDeleteTextures(1, &history_color_[index]);
      history_fbo_[index] = history_color_[index] = 0;
      if (bloom_fbo_[index]) gl_.DeleteFramebuffers(1, &bloom_fbo_[index]);
      if (bloom_color_[index]) glDeleteTextures(1, &bloom_color_[index]);
      bloom_fbo_[index] = bloom_color_[index] = 0;
    }
    current_fbo_ = current_color_ = 0;
    history_width_ = history_height_ = 0;
    history_valid_ = false;
    bloom_width_ = bloom_height_ = 0;
  }

  void ensure_history(int width, int height) {
    if (width == history_width_ && height == history_height_) return;
    release_history();
    create_color_target(current_fbo_, current_color_, width, height);
    bloom_width_ = std::max(1, (width + 3) / 4);
    bloom_height_ = std::max(1, (height + 3) / 4);
    for (int index = 0; index < 2; ++index) {
      create_color_target(history_fbo_[index], history_color_[index], width, height);
      create_color_target(bloom_fbo_[index], bloom_color_[index], bloom_width_, bloom_height_);
    }
    gl_.BindFramebuffer(GL_FRAMEBUFFER, 0);
    history_width_ = width;
    history_height_ = height;
  }

  void release_fallback_queue() {
    if (fallback_queue_.pixels) cudaFree(fallback_queue_.pixels);
    if (fallback_queue_.count) cudaFree(fallback_queue_.count);
    fallback_queue_ = {};
    queue_idle_since_ = 0;
  }

  void release_image() {
    release_fallback_queue();
    if (cuda_resource_) {
      glFinish();
      cudaGraphicsUnregisterResource(cuda_resource_);
      cuda_resource_ = nullptr;
    }
    if (texture_) glDeleteTextures(1, &texture_);
    if (pbo_) gl_.DeleteBuffers(1, &pbo_);
    texture_ = pbo_ = 0;
    width_ = height_ = 0;
  }

  GLApi gl_;
  GLuint program_ = 0;
  GLuint temporal_program_ = 0;
  GLuint bloom_extract_program_ = 0;
  GLuint bloom_blur_program_ = 0;
  GLuint bloom_composite_program_ = 0;
  GLuint bloom_fbo_[2]{};
  GLuint bloom_color_[2]{};
  int bloom_width_ = 0;
  int bloom_height_ = 0;
  GLint bloom_direction_location_ = -1;
  GLuint vao_ = 0;
  GLuint pbo_ = 0;
  GLuint texture_ = 0;
  int capacity_width_ = 0;
  int capacity_height_ = 0;
  GLuint post_queries_[4]{};
  int query_read_ = 0, query_write_ = 0, query_count_ = 0;
  double post_ms_ = 2.0, work_ms_ = 0.0;
  float kernel_ms_ = 0;
  bool timing_sample_ready_ = false;
  GLint escape_size_location_ = -1;
  GLuint current_fbo_ = 0;
  GLuint current_color_ = 0;
  GLuint history_fbo_[2]{};
  GLuint history_color_[2]{};
  int history_width_ = 0;
  int history_height_ = 0;
  int history_index_ = 0;
  bool history_valid_ = false;
  Camera previous_camera_{};
  cudaGraphicsResource* cuda_resource_ = nullptr;
  struct FrameTiming {
    cudaEvent_t start = nullptr, stop = nullptr;
    double pre_post_ms = 0, submit_ms = 0;
    int width = 0, height = 0, output_width = 0, output_height = 0;
  };
  FrameTiming frame_timings_[4]{};
  float2* reference_orbit_ = nullptr;
  float4* fold_detail_ = nullptr;
  Uint64 fold_probe_time_ = 0, previous_effect_time_ = 0;
  float fold_detail_target_ = 0, fold_detail_weight_ = 0;
  MandelbrotFallbackQueue fallback_queue_{};
  Uint64 queue_idle_since_ = 0;
  GLint palette_from_location_ = -1;
  GLint palette_to_location_ = -1;
  GLint palette_blend_location_ = -1;
  GLint time_location_ = -1;
  GLint trip_location_ = -1;
  GLint breath_location_ = -1;
  float previous_folding_ = 0;
  GLint matrix_location_ = -1;
  GLint translation_location_ = -1;
  GLint pixel_step_location_ = -1;
  GLint history_weight_location_ = -1;
  int width_ = 0;
  int height_ = 0;
  bool cached_reference_valid_ = false;
  double cached_reference_x_ = 0.0;
  double cached_reference_y_ = 0.0;
  int cached_reference_length_ = 0;
  double cached_seed_shift_x_ = 0, cached_seed_shift_y_ = 0;
  double cached_parameter_shift_x_ = 0, cached_parameter_shift_y_ = 0;
};

// Preserve the saved fullscreen value; legacy borderless (1) loads as windowed.
enum class WindowMode { Windowed = 0, Fullscreen = 2 };

void update_cursor(WindowMode mode, bool focused, bool menu_visible) {
  const bool should_show = mode != WindowMode::Fullscreen || !focused || menu_visible;
  if (SDL_CursorVisible() == should_show) return;
  const bool success = should_show ? SDL_ShowCursor() : SDL_HideCursor();
  if (!success) throw std::runtime_error(SDL_GetError());
}

void toggle_window_mode(SDL_Window* window, WindowMode& mode) {
  if (mode == WindowMode::Windowed) {
    if (!SDL_SetWindowFullscreen(window, true)) throw std::runtime_error(SDL_GetError());
    mode = WindowMode::Fullscreen;
  } else {
    if (!SDL_SetWindowFullscreen(window, false)) throw std::runtime_error(SDL_GetError());
    SDL_SyncWindow(window);
    if (!SDL_SetWindowBordered(window, true)) throw std::runtime_error(SDL_GetError());
    mode = WindowMode::Windowed;
  }
}

const char* mode_name(WindowMode mode) {
  switch (mode) {
    case WindowMode::Windowed: return "Windowed";
    case WindowMode::Fullscreen: return "Fullscreen";
  }
  return "Unknown";
}

constexpr std::array<const char*, 8> kPaletteNames{{
    "Neon dusk", "Sunset chrome", "Cyber ice", "Ultraviolet",
    "Vaporwave", "Acid arcade", "Electric ocean", "Gilded night"}};
constexpr int kPaletteCount = static_cast<int>(kPaletteNames.size());
constexpr double kPaletteFadeSeconds = 1.5;

void save_screenshot(int width, int height, const char* path) {
  std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 4);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
  const size_t stride = static_cast<size_t>(width) * 4;
  // Flip in place: a second full-size image doubled screenshot RAM use.
  for (int y = 0; y < height / 2; ++y) {
    auto first = pixels.begin() + y * stride;
    auto opposite = pixels.begin() + (height - 1 - y) * stride;
    std::swap_ranges(first, first + stride, opposite);
  }
  SDL_Surface* surface = SDL_CreateSurfaceFrom(
      width, height, SDL_PIXELFORMAT_RGBA32, pixels.data(), static_cast<int>(stride));
  if (!surface) throw std::runtime_error(SDL_GetError());
  const bool saved = SDL_SaveBMP(surface, path);
  SDL_DestroySurface(surface);
  if (!saved) throw std::runtime_error(SDL_GetError());
}

std::string save_user_screenshot(int width, int height) {
  const char* pictures = SDL_GetUserFolder(SDL_FOLDER_PICTURES);
  if (!pictures) throw std::runtime_error(SDL_GetError());
  const auto directory = std::filesystem::u8path(pictures) / "MandelDrift";
  std::filesystem::create_directories(directory);
  const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  const auto path = directory / ("mandel-" + std::to_string(stamp) + ".bmp");
  const std::string filename = path.u8string();
  save_screenshot(width, height, filename.c_str());
  return filename;
}

int run(bool smoke_test, bool cycle_test, bool motion_test = false, bool motion_return = false) {
  if (!SDL_Init(SDL_INIT_VIDEO)) throw std::runtime_error(SDL_GetError());
  const char* base_path = SDL_GetBasePath();
  if (!base_path) throw std::runtime_error(SDL_GetError());
  const auto settings_path = std::filesystem::u8path(base_path) / "settings.ini";
  // Automated runs use defaults and never read or overwrite personal preferences.
  const bool persist_settings = !smoke_test && !cycle_test;
  const AppSettings preferences = persist_settings ? load_settings(settings_path) : AppSettings{};
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  SDL_Window* window = SDL_CreateWindow("Mandel Drift", 1280, 720,
      SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
  if (!window) throw std::runtime_error(SDL_GetError());
  SDL_GLContext context = SDL_GL_CreateContext(window);
  if (!context) throw std::runtime_error(SDL_GetError());
  if (!SDL_GL_MakeCurrent(window, context)) throw std::runtime_error(SDL_GetError());
  SDL_GL_SetSwapInterval(1);

  {
    Renderer renderer;
    SettingsOverlay settings;
    SettingsOverlay help(true);
    WindowMode mode = WindowMode::Windowed;
    if (preferences.window_mode == 2) toggle_window_mode(window, mode);
    if (preferences.window_mode != 0) SDL_SyncWindow(window);
    if (cycle_test) {
      toggle_window_mode(window, mode);
      SDL_SyncWindow(window);
    }
    bool window_focused = (SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS) != 0;
    bool menu_visible = preferences.menu_visible;
    double menu_progress = menu_visible ? 1.0 : 0.0;
    bool help_visible = false;
    double help_progress = 0.0;
    update_cursor(mode, window_focused, menu_visible);
    bool running = true;
    bool camera_paused = preferences.paused;
    bool dive_mode = preferences.endless_dive;
    bool bloom_enabled = preferences.bloom;
    bool music_enabled = preferences.music;
    bool music_muted = preferences.muted;
    int music_volume = preferences.volume;
    bool volume_dragging = false;
    AmbientMusic music;
    music.set_controls(music_enabled, music_muted, music_volume);
    if (persist_settings && music_enabled && !music.start())
      std::fprintf(stderr, "Audio unavailable: %s\n", SDL_GetError());
    DiveJourney journey;
    // Review the last 118 seconds of either Julia ramp and its quiet endpoint.
    // Separate real-time excerpts cover both ramps without speeding up time.
    const double motion_offset = motion_test ? journey.dive_seconds() +
        (motion_return ? kJuliaHoldSeconds : kJuliaPlateauStart + kJuliaQuietSeconds) - 120.0 : 0.0;
    if (motion_test) journey.advance(motion_offset);
    std::FILE* motion_file = nullptr;
    if (motion_test && fopen_s(&motion_file, "motion-trace.csv", "w") != 0)
      throw std::runtime_error("Could not create camera motion trace.");
    std::unique_ptr<std::FILE, decltype(&std::fclose)> motion_trace(motion_file, &std::fclose);
    if (motion_test) {
      std::fprintf(motion_trace.get(), "wall_seconds,journey_seconds,stage,wall_dt,camera_dt,base_span,seed_span,julia_amount,base_log_speed,seed_log_speed,seed_pan_spans_per_second,seed_center_x,seed_center_y,parameter_x,parameter_y,rotation,render_width,render_height,kernel_ms,work_ms\n");
      std::fprintf(stderr, "Motion review: 120 real-time seconds, journey offset %.6f, fullscreen; positive log speed = pullback\n", motion_offset);
    }
    const double test_cycle_duration = journey.cycle_duration();
    double cycle_elapsed = 0.0;
    double next_cycle_log = 0.0;
    size_t cycle_capture = 0;
    const std::array<double, 10> capture_times = motion_test
        ? std::array<double, 10>{{0, 10, 30, 50, 56, 62, 76, 82, 94, 114}}
        : std::array<double, 10>{{
        journey.dive_seconds() * 0.34,
        journey.dive_seconds() * 0.62,
        journey.dive_seconds() * 0.78, journey.dive_seconds() - 7.0,
        journey.dive_seconds(), journey.dive_seconds() + kJuliaQuietSeconds + kJuliaRampSeconds * .5,
        journey.dive_seconds() + kJuliaHoldSeconds * .5, test_cycle_duration - journey.pullback_seconds(),
        test_cycle_duration - journey.pullback_seconds() * .5, test_cycle_duration + 1.0}};
    Camera last_camera = dive_mode ? journey.frame() : camera_at(0.0);
    CameraTransition transition;
    bool rotation_enabled = preferences.rotation;
    int rotation_direction = preferences.rotation_direction;
    bool auto_palette = preferences.auto_palette;
    bool random_palette_order = preferences.shuffle;
    int palette_from = preferences.palette;
    int palette_to = preferences.palette;
    double palette_fade = kPaletteFadeSeconds;
    std::mt19937 palette_rng(std::random_device{}());
    std::array<int, kPaletteCount - 1> palette_bag{};
    size_t palette_bag_index = palette_bag.size();
    double camera_time = 0.0;
    double rotation_angle = 0.0;
    if (motion_test) rotation_angle = motion_offset * .055;
    double color_time = 0.0;
    double trip_time = 0.0;
    if (motion_test) trip_time = motion_offset;
    double previous_motion_base_span = 0, previous_motion_seed_span = 0;
    double previous_motion_seed_x = 0, previous_motion_seed_y = 0;
    const auto motion_wall_start = std::chrono::steady_clock::now();
    float musical_breath = 0;
    double palette_timer = 0.0;
    FrameBudget frame_budget;
    std::vector<double> cycle_intervals;
    int last_output_w = 0;
    int last_output_h = 0;
    double title_timer = 0.0;
    int title_frames = 0;
    int total_frames = 0;
    int valid_timing_samples = 0;
    bool screenshot_requested = false;
    bool smoke_blend_saved = false;
    bool smoke_final_saved = false;
    bool smoke_pullback_pending = false;
    bool smoke_pullback_saved = false;
    auto previous = std::chrono::steady_clock::now();
    Uint64 previous_frame_start = 0;
    Uint64 pacing_interval_sum = 0;
    Uint64 pacing_min_interval = UINT64_MAX;
    int pacing_samples = 0;
    std::string last_saved_settings;
    bool settings_warning_shown = false;
    const auto save_preferences = [&]() {
      if (!persist_settings) return;
      AppSettings current;
      current.auto_palette = auto_palette;
      current.shuffle = random_palette_order;
      current.endless_dive = dive_mode;
      current.bloom = bloom_enabled;
      current.music = music_enabled;
      current.muted = music_muted;
      current.volume = music_volume;
      current.rotation = rotation_enabled;
      current.rotation_direction = rotation_direction;
      current.paused = camera_paused;
      current.menu_visible = menu_visible;
      current.palette = palette_to;
      current.window_mode = static_cast<int>(mode);
      const std::string serialized = settings_text(current);
      if (serialized == last_saved_settings) return;
      if (!save_settings(settings_path, current) && !settings_warning_shown) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_WARNING, "Settings could not be saved",
            "Mandel Drift could not write settings.ini beside the app. Move the app to a writable folder to keep your preferences.", window);
        settings_warning_shown = true;
      }
      // Retry on the next change, rather than repeatedly writing every frame.
      last_saved_settings = serialized;
    };
    save_preferences();

    const auto next_shuffled_palette = [&]() {
      if (palette_bag_index >= palette_bag.size()) {
        size_t index = 0;
        for (int value = 0; value < kPaletteCount; ++value) {
          if (value != palette_to) palette_bag[index++] = value;
        }
        std::shuffle(palette_bag.begin(), palette_bag.end(), palette_rng);
        palette_bag_index = 0;
      }
      return palette_bag[palette_bag_index++];
    };

    if (smoke_test) {
      std::array<bool, kPaletteCount> seen{};
      seen[palette_to] = true;
      for (int i = 1; i < kPaletteCount; ++i) {
        const int next = next_shuffled_palette();
        if (seen[next]) throw std::runtime_error("Random palette order repeated early.");
        seen[next] = true;
        palette_to = next;
      }
      palette_to = 0;
      palette_bag_index = palette_bag.size();

      const Camera deep{-0.743643887037151, 0.131825904205330, 1e-9};
      const Camera wide{-0.65, 0.0, 3.2};
      for (bool reverse : {false, true}) {
        CameraTransition probe;
        Camera prior = reverse ? wide : deep;
        const Camera destination = reverse ? deep : wide;
        probe.start(prior, destination);
        int frames = 0;
        while (probe.active() && frames++ < 20000) {
          const Camera paused = probe.advance(0.0);
          if (paused.x != prior.x || paused.y != prior.y ||
              std::abs(std::log(paused.span / prior.span)) > 1e-12)
            throw std::runtime_error("Paused transition moved.");
          const Camera next = probe.advance(1.0 / 60.0);
          if (std::abs(std::log(next.span / prior.span)) > 0.65 / 60.0 + 1e-9 ||
              std::hypot(next.x - prior.x, next.y - prior.y) / prior.span > 0.18 / 60.0 + 1e-9)
            throw std::runtime_error("Mode transition exceeded its motion limit.");
          prior = next;
          if (frames == 90) probe.start(prior, destination); // Rapid re-toggle origin.
        }
        if (probe.active() || prior.x != destination.x || prior.y != destination.y ||
            prior.span != destination.span)
          throw std::runtime_error("Mode transition failed to arrive.");
      }

      DiveJourney continuity_probe;
      continuity_probe.advance(continuity_probe.cycle_duration() - 0.001);
      const Camera before = continuity_probe.frame();
      continuity_probe.advance(0.002);
      const Camera after = continuity_probe.frame();
      if (std::hypot(after.x - before.x, after.y - before.y) > 0.001 ||
          std::abs(std::log(after.span / before.span)) > 0.001) {
        throw std::runtime_error("Endless dive changed abruptly at a scene boundary.");
      }
      DiveJourney julia_probe;
      unsigned int deep_visits = 0;
      for (int cycle = 0; cycle < 24; ++cycle) {
        // Independently sample the visible camera span over both legs. This
        // catches a return-speed regression even if their endpoints still join.
        const double dive = julia_probe.dive_seconds();
        const double pullback = julia_probe.pullback_seconds();
        double previous_dive_span = 3.2;
        double previous_return_span = 3.2;
        for (int sample = 0; sample <= 120; ++sample) {
          const double t = dive * sample / 120;
          auto inward = julia_probe;
          auto outward = julia_probe;
          inward.advance(t);
          outward.advance(dive + julia_probe.hold_duration() + pullback - t);
          const double inward_span = inward.frame().span;
          const double outward_span = outward.frame().span;
          if (std::abs(std::log(inward_span / outward_span)) > 1e-10)
            throw std::runtime_error("Pullback no longer matches the dive's zoom progression.");
          if (sample > 0 && (std::abs(std::log(inward_span / previous_dive_span)) / (dive / 120) > .185 ||
                            std::abs(std::log(outward_span / previous_return_span)) / (dive / 120) > .185))
            throw std::runtime_error("Journey zoom exceeded its gradual camera speed.");
          previous_dive_span = inward_span;
          previous_return_span = outward_span;
        }
        if (julia_probe.minimum_span() <= .00003) {
          const bool expected = deep_visits++ % 2 == 0;
          if (julia_probe.julia_hold() != expected)
            throw std::runtime_error("Julia excursions did not alternate deep holds.");
        } else if (julia_probe.julia_hold()) {
          throw std::runtime_error("Julia appeared at a shallow site.");
        }
        // Measure actual renderer seed coordinates, including the local orbit,
        // over the complete excursion at 60 Hz, not just the base camera span.
        if (julia_probe.julia_hold()) {
          auto motion = julia_probe;
          motion.advance(dive);
          auto seed_position = [](const Camera& camera) {
            return std::array<double, 2>{{camera.julia.x + camera.julia.seed_shift_x +
                (camera.x - camera.julia.x) * camera.julia.seed_scale,
                camera.julia.y + camera.julia.seed_shift_y +
                (camera.y - camera.julia.y) * camera.julia.seed_scale}};
          };
          Camera previous_camera = motion.frame();
          double peak_zoom = 0, peak_pan = 0;
          constexpr double step = 1.0 / 60;
          for (int sample = 0; sample < static_cast<int>(kJuliaHoldSeconds * 60); ++sample) {
            motion.advance(step);
            const Camera current = motion.frame();
            const double old_span = previous_camera.span * previous_camera.julia.seed_scale;
            const double new_span = current.span * current.julia.seed_scale;
            const auto old_center = seed_position(previous_camera);
            const auto new_center = seed_position(current);
            peak_zoom = std::max(peak_zoom, std::abs(std::log(new_span / old_span)) / step);
            peak_pan = std::max(peak_pan, std::hypot(new_center[0] - old_center[0],
                new_center[1] - old_center[1]) / (std::sqrt(old_span * new_span) * step));
            if (peak_zoom > kJuliaMaxLogZoomSpeed || peak_pan > kJuliaMaxPanSpeed)
              throw std::runtime_error("Julia visible zoom or centering exceeded its gradual camera speed.");
            previous_camera = current;
          }
          std::fprintf(stderr, "Julia motion cycle %d: peak log zoom %.6f/sec, pan %.6f view widths/sec\n",
              cycle, peak_zoom, peak_pan);
        }
        const double hold_midpoint = julia_probe.hold_duration() * .5;
        julia_probe.advance(julia_probe.dive_seconds() + hold_midpoint);
        const Camera held = julia_probe.frame();
        if (held.julia.amount != (julia_probe.julia_hold() ? 1.0 : 0.0))
          throw std::runtime_error("Julia hold endpoint was not reached.");
        julia_probe.advance(0);
        if (julia_probe.frame().julia.amount != held.julia.amount)
          throw std::runtime_error("Paused Julia hold advanced.");
        CameraTransition mode_probe;
        mode_probe.start(held, camera_at(0));
        const Camera start = mode_probe.advance(0);
        if (std::abs(start.julia.amount - held.julia.amount) > 1e-12)
          throw std::runtime_error("Mode switch cut the Julia excursion.");
        for (int i = 0; mode_probe.active() && i < 10000; ++i) mode_probe.advance(.05);
        if (mode_probe.active() || mode_probe.advance(0).julia.amount != 0)
          throw std::runtime_error("Mode switch failed to leave Julia.");
        // Pass the return edge before moving to the next randomized site.
        julia_probe.advance(julia_probe.cycle_duration() - julia_probe.dive_seconds() - hold_midpoint);
      }
    }

    while (running) {
      // Include rendering and VSync in the budget; sleep only for time left over.
      // Anchor to the actual start so a slow frame never causes a catch-up burst.
      Uint64 frame_start = SDL_GetTicksNS();
      if (previous_frame_start != 0) {
        const Uint64 deadline = previous_frame_start + kFrameIntervalNS;
        while (frame_start < deadline) {
          SDL_DelayPrecise(deadline - frame_start);
          frame_start = SDL_GetTicksNS();
        }
        if (cycle_test) cycle_intervals.push_back((frame_start - previous_frame_start) / 1.0e6);
        if (smoke_test) {
          const Uint64 interval = frame_start - previous_frame_start;
          pacing_interval_sum += interval;
          pacing_min_interval = std::min(pacing_min_interval, interval);
          ++pacing_samples;
        }
      }
      previous_frame_start = frame_start;
      if (smoke_test) {
        // Exercise the actual shortcuts through the normal SDL event handler.
        SDL_Keycode key = 0;
        if (total_frames == 8 || total_frames == 24 || total_frames == 44 || total_frames == 49)
          key = SDLK_F;
        if (total_frames == 18 || total_frames == 22) key = SDLK_S;
        if (total_frames == 19) key = SDLK_QUESTION;
        if (total_frames == 21 || total_frames == 27 || total_frames == 28) key = SDLK_ESCAPE;
        if (total_frames == 25) key = SDLK_SLASH;
        if (total_frames == 26) key = SDLK_S;
        if (key) {
          SDL_Event press{};
          press.type = SDL_EVENT_KEY_DOWN;
          press.key.key = key;
          if (total_frames == 25) press.key.mod = SDL_KMOD_SHIFT;
          if (!SDL_PushEvent(&press)) throw std::runtime_error(SDL_GetError());
        }
      }
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT) running = false;
        if (event.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
          volume_dragging = false;
          window_focused = false;
          update_cursor(mode, window_focused, menu_visible || menu_progress > 0.0 || help_visible || help_progress > 0.0);
        }
        if (event.type == SDL_EVENT_WINDOW_FOCUS_GAINED) {
          window_focused = true;
          update_cursor(mode, window_focused, menu_visible || menu_progress > 0.0 || help_visible || help_progress > 0.0);
        }
        if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_LEFT)
          volume_dragging = false;
        if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT) {
          int ww = 0, wh = 0, dw = 0, dh = 0;
          SDL_GetWindowSize(window, &ww, &wh);
          SDL_GetWindowSizeInPixels(window, &dw, &dh);
          if (ww > 0 && wh > 0) {
            const float x = event.button.x * dw / ww, y = event.button.y * dh / wh;
            const float stack = static_cast<float>(menu_progress * menu_progress * (3.0 - 2.0 * menu_progress));
            if (help_visible && help.close_hit(x, y, dw, dh, stack)) help_visible = false;
            else if (menu_visible && settings.close_hit(x, y, dw, dh)) menu_visible = false;
          }
        }
        if (event.type == SDL_EVENT_MOUSE_MOTION && volume_dragging && menu_visible) {
          int window_w = 0, window_h = 0, drawable_w = 0, drawable_h = 0;
          SDL_GetWindowSize(window, &window_w, &window_h);
          SDL_GetWindowSizeInPixels(window, &drawable_w, &drawable_h);
          if (window_w > 0 && drawable_w > 0 && drawable_h > 0)
            music_volume = settings.volume_at(event.motion.x * drawable_w / window_w, drawable_w, drawable_h);
        }
        if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT &&
            menu_visible && menu_progress > 0.4) {
          int window_w = 0, window_h = 0, drawable_w = 0, drawable_h = 0;
          SDL_GetWindowSize(window, &window_w, &window_h);
          SDL_GetWindowSizeInPixels(window, &drawable_w, &drawable_h);
          if (window_w > 0 && window_h > 0) {
            const int control = settings.hit_test(
                event.button.x * drawable_w / window_w,
                event.button.y * drawable_h / window_h, drawable_w, drawable_h);
            if (control == 0) {
              auto_palette = !auto_palette;
              palette_timer = 0.0;
            } else if (control == 1) {
              random_palette_order = !random_palette_order;
              palette_bag_index = palette_bag.size();
            } else if (control == 2) {
              dive_mode = !dive_mode;
              transition.start(last_camera, dive_mode ? journey.frame() : camera_at(camera_time));
            } else if (control == 3) {
              bloom_enabled = !bloom_enabled;
            } else if (control == 4) {
              music_enabled = !music_enabled;
              if (music_enabled && persist_settings && !music.start())
                std::fprintf(stderr, "Audio unavailable: %s\n", SDL_GetError());
            } else if (control == 5) {
              volume_dragging = true;
              music_volume = settings.volume_at(event.button.x * drawable_w / window_w, drawable_w, drawable_h);
            } else if (control == 6) {
              music_muted = !music_muted;
            }
          }
        }
        if (event.type != SDL_EVENT_KEY_DOWN || event.key.repeat) continue;
        // SDL may report the unshifted slash or the translated question mark.
        if (event.key.key == SDLK_QUESTION ||
            (event.key.key == SDLK_SLASH && (event.key.mod & SDL_KMOD_SHIFT))) {
          help_visible = !help_visible;
          update_cursor(mode, window_focused, true);
          continue;
        }
        switch (event.key.key) {
          case SDLK_ESCAPE:
            if (help_visible) help_visible = false;
            else if (menu_visible) menu_visible = false;
            else running = false;
            break;
          case SDLK_S:
            menu_visible = !menu_visible;
            break;
          case SDLK_B:
            bloom_enabled = !bloom_enabled;
            break;
          case SDLK_M:
            if (!music_enabled) {
              music_enabled = true;
              music_muted = false;
              if (persist_settings && !music.start())
                std::fprintf(stderr, "Audio unavailable: %s\n", SDL_GetError());
            } else {
              music_muted = !music_muted;
            }
            break;
          case SDLK_MINUS: music_volume = std::max(0, music_volume - 5); break;
          case SDLK_EQUALS: music_volume = std::min(100, music_volume + 5); break;
          case SDLK_F:
            toggle_window_mode(window, mode);
            break;
          case SDLK_P:
            palette_from = palette_to;
            palette_to = (palette_to + kPaletteCount +
                          ((event.key.mod & SDL_KMOD_SHIFT) ? -1 : 1)) % kPaletteCount;
            palette_fade = 0.0;
            palette_timer = 0.0;
            palette_bag_index = palette_bag.size();
            break;
          case SDLK_A:
            auto_palette = !auto_palette;
            palette_timer = 0.0;
            break;
          case SDLK_O:
            random_palette_order = !random_palette_order;
            palette_bag_index = palette_bag.size();
            break;
          case SDLK_E:
            dive_mode = !dive_mode;
            transition.start(last_camera, dive_mode ? journey.frame() : camera_at(camera_time));
            break;
          case SDLK_R:
            if (event.key.mod & SDL_KMOD_SHIFT) rotation_direction = -rotation_direction;
            else rotation_enabled = !rotation_enabled;
            break;
          case SDLK_SPACE: camera_paused = !camera_paused; break;
          case SDLK_F12: screenshot_requested = true; break;
          default: break;
        }
        update_cursor(mode, window_focused, menu_visible || menu_progress > 0.0 || help_visible || help_progress > 0.0);
      }
      if (!menu_visible || !running) volume_dragging = false;
      music.set_controls(music_enabled, music_muted, music_volume);
      if (!volume_dragging) save_preferences();
      if (!running) {
        if (smoke_test) throw std::runtime_error("Smoke test exited before completing its checks.");
        break;
      }

      if (smoke_test && (total_frames == 8 || total_frames == 24 || total_frames == 44 || total_frames == 49)) {
        SDL_SyncWindow(window);
        const SDL_WindowFlags flags = SDL_GetWindowFlags(window);
        window_focused = (flags & SDL_WINDOW_INPUT_FOCUS) != 0;
        update_cursor(mode, window_focused, menu_visible || menu_progress > 0.0 || help_visible || help_progress > 0.0);
        const bool correct =
            (mode == WindowMode::Fullscreen && (flags & SDL_WINDOW_FULLSCREEN) != 0) ||
            (mode == WindowMode::Windowed &&
             (flags & (SDL_WINDOW_BORDERLESS | SDL_WINDOW_FULLSCREEN)) == 0);
        if (!correct) throw std::runtime_error("Window mode smoke check failed.");
        const bool expected_fullscreen = total_frames == 8 || total_frames == 44;
        if ((mode == WindowMode::Fullscreen) != expected_fullscreen)
          throw std::runtime_error("F shortcut did not toggle window mode.");
        if (SDL_CursorVisible() == (mode == WindowMode::Fullscreen && window_focused)) {
          throw std::runtime_error("Cursor visibility smoke check failed.");
        }
      }
      if (smoke_test && total_frames == 18) {
        if (!menu_visible || screenshot_requested)
          throw std::runtime_error("S shortcut did not open settings.");
        menu_progress = 1.0;
        update_cursor(mode, window_focused, true);
        if (!SDL_CursorVisible()) throw std::runtime_error("Menu did not reveal cursor.");
      }
      if (smoke_test && total_frames == 22) {
        if (menu_visible || screenshot_requested)
          throw std::runtime_error("S shortcut did not close settings.");
        menu_progress = 0.0;
        update_cursor(mode, window_focused, false);
      }
      if (smoke_test && total_frames == 36) {
        journey.advance(journey.dive_seconds() * 0.78);
        rotation_angle = 0.75;
      }
      if (smoke_test && (total_frames == 19 || total_frames == 25 || total_frames == 26)) {
        if (!help_visible || menu_visible != (total_frames != 25))
          throw std::runtime_error("Help/settings shortcut state mismatch.");
        help_progress = 1.0;
        menu_progress = menu_visible ? 1.0 : 0.0;
      }
      if (smoke_test && (total_frames == 21 || total_frames == 27 || total_frames == 28)) {
        if (help_visible || menu_visible != (total_frames != 28))
          throw std::runtime_error("Escape did not close panels in order.");
        help_progress = 0.0;
        menu_progress = menu_visible ? 1.0 : 0.0;
      }
      if (smoke_test && total_frames == 46)
        journey.advance(journey.dive_seconds() * 0.21);
      if (smoke_test && total_frames == 50) {
        palette_from = palette_to;
        palette_to = 4;
        palette_fade = 0.0;
        camera_paused = true;
      }

      auto now = std::chrono::steady_clock::now();
      const double wall_dt = std::chrono::duration<double>(now - previous).count();
      const double dt = smoke_test ? 1.0 / 60.0 :
          std::clamp(wall_dt, 0.0, 0.1);
      previous = now;
      trip_time += dt;
      TripEffects effects = trip_effects(trip_time);
      // Freeze all palette motion while folding, including its fade ramps.
      const float target_breath = music_enabled && !music_muted && music.available()
          ? std::clamp(music.envelope() * 12.0f, 0.0f, 1.0f) : 0.0f;
      musical_breath += (target_breath - musical_breath) * static_cast<float>(1.0 - std::exp(-dt / 0.9));
      if (cycle_test) cycle_elapsed += dt;
      menu_progress = std::clamp(menu_progress + (menu_visible ? 1.0 : -1.0) * dt / 0.22,
                                 0.0, 1.0);
      help_progress = std::clamp(help_progress + (help_visible ? 1.0 : -1.0) * dt / 0.22, 0.0, 1.0);
      update_cursor(mode, window_focused, menu_progress > 0.0 || help_progress > 0.0);
      if (!camera_paused) {
        if (!transition.active()) {
          if (dive_mode) journey.advance(dt);
          else camera_time += dt;
        }
        if (rotation_enabled) rotation_angle += dt * 0.055 * rotation_direction;
      }
      Camera camera = dive_mode ? journey.frame() : camera_at(camera_time);
      if (transition.active()) camera = transition.advance(camera_paused ? 0.0 : dt);
      last_camera = camera;
      camera.rotation = rotation_angle;
      effects.folding *= folding_depth_weight(camera.span, dive_mode ? journey.minimum_span() : .00003);
      if ((dive_mode && !journey.diving()) || transition.active() || camera.julia.amount > 0) effects.folding = 0;
      effects.waves *= static_cast<float>(1.0 - camera.julia.amount);
      if (effects.folding == 0) color_time += dt;
      if (effects.folding == 0) palette_fade = std::min(kPaletteFadeSeconds, palette_fade + dt);
      if (auto_palette && effects.folding == 0) {
        palette_timer += dt;
        if (palette_timer >= 25.0) {
          palette_from = palette_to;
          palette_to = random_palette_order ? next_shuffled_palette()
                                            : (palette_to + 1) % kPaletteCount;
          palette_fade = 0.0;
          palette_timer = 0.0;
        }
      }

      int output_w = 0;
      int output_h = 0;
      if (!SDL_GetWindowSizeInPixels(window, &output_w, &output_h) ||
          output_w <= 0 || output_h <= 0) {
        SDL_Delay(50);
        continue;
      }
      if (output_w != last_output_w || output_h != last_output_h) {
        last_output_w = output_w;
        last_output_h = output_h;
        frame_budget.resize(output_w, output_h);
      }
      const double quality_scale = frame_budget.scale();
      const int render_w = std::clamp(static_cast<int>(std::lround(output_w * quality_scale)), 1, output_w);
      const int render_h = std::clamp(static_cast<int>(std::lround(output_h * quality_scale)), 1, output_h);
      renderer.resize(render_w, render_h, output_w, output_h);
      music.set_depth(static_cast<float>(std::clamp(std::log2(3.2 / camera.span) / 17.0, 0.0, 1.0)));
      const float kernel_ms = renderer.draw(camera, palette_from, palette_to,
                                            static_cast<float>(palette_fade / kPaletteFadeSeconds),
                                            static_cast<float>(color_time),
                                            output_w, output_h, journey.reference_point(), bloom_enabled,
                                            effects, static_cast<float>(std::fmod(trip_time * 0.025, 6.28318530718)), musical_breath);
      const double frame_work_ms = renderer.work_ms();
      const bool frame_timing_ready = renderer.timing_sample_ready();
      if (motion_test) {
        const double seed_span = camera.span * camera.julia.seed_scale;
        const double seed_x = camera.julia.amount > 0 ? camera.julia.x + camera.julia.seed_shift_x +
            (camera.x - camera.julia.x) * camera.julia.seed_scale : camera.x;
        const double seed_y = camera.julia.amount > 0 ? camera.julia.y + camera.julia.seed_shift_y +
            (camera.y - camera.julia.y) * camera.julia.seed_scale : camera.y;
        const double parameter_x = camera.julia.amount > 0 ? camera.julia.x + camera.julia.parameter_shift_x +
            (camera.x - camera.julia.x) * (1 - camera.julia.amount) : camera.x;
        const double parameter_y = camera.julia.amount > 0 ? camera.julia.y + camera.julia.parameter_shift_y +
            (camera.y - camera.julia.y) * (1 - camera.julia.amount) : camera.y;
        const bool has_previous = previous_motion_seed_span > 0 && wall_dt > 0;
        const double base_speed = has_previous ? std::log(camera.span / previous_motion_base_span) / wall_dt : 0;
        const double seed_speed = has_previous ? std::log(seed_span / previous_motion_seed_span) / wall_dt : 0;
        const double pan_speed = has_previous ? std::hypot(seed_x - previous_motion_seed_x, seed_y - previous_motion_seed_y) /
            (std::sqrt(seed_span * previous_motion_seed_span) * wall_dt) : 0;
        const double journey_time = motion_offset + cycle_elapsed;
        const char* stage = journey_time < journey.dive_seconds() ? "dive" :
            journey_time < journey.dive_seconds() + journey.hold_duration() ? "hold" : "pullback";
        std::fprintf(motion_trace.get(), "%.9f,%.9f,%s,%.9f,%.9f,%.17g,%.17g,%.9f,%.9f,%.9f,%.9f,%.17g,%.17g,%.17g,%.17g,%.9f,%d,%d,%.6f,%.6f\n",
            std::chrono::duration<double>(now - motion_wall_start).count(), journey_time, stage, wall_dt, dt,
            camera.span, seed_span, camera.julia.amount, base_speed, seed_speed, pan_speed, seed_x, seed_y,
            parameter_x, parameter_y, camera.rotation, render_w, render_h, kernel_ms, frame_work_ms);
        previous_motion_base_span = camera.span;
        previous_motion_seed_span = seed_span;
        previous_motion_seed_x = seed_x;
        previous_motion_seed_y = seed_y;
        if (total_frames % 60 == 0) std::fflush(motion_trace.get());
      }
      if (frame_timing_ready) {
        ++valid_timing_samples;
        if (smoke_test && (!std::isfinite(kernel_ms) || kernel_ms <= 0 ||
            !std::isfinite(frame_work_ms) || frame_work_ms < kernel_ms ||
            !std::isfinite(renderer.post_ms()) || renderer.post_ms() < 0))
          throw std::runtime_error("Asynchronous GPU timing produced an invalid frame budget.");
      }
      if (smoke_test && total_frames == 30) {
        save_screenshot(output_w, output_h, "smoke-bloom-on.bmp");
        renderer.present(false);
        save_screenshot(output_w, output_h, "smoke-bloom-off.bmp");
        renderer.present(true);
      }
      if (cycle_test && cycle_elapsed >= next_cycle_log) {
        std::fprintf(stderr, "cycle %.2f span %.6g render %dx%d output %dx%d kernel %.2fms work %.2fms post %.2fms fold %.3f julia %.3f\n",
                     cycle_elapsed, camera.span, render_w, render_h, output_w, output_h, kernel_ms,
                     renderer.work_ms(), renderer.post_ms(), renderer.folding_amount(), camera.julia.amount);
        std::fflush(stderr);
        next_cycle_log += 1.0;
      }
      if (cycle_test && cycle_capture < capture_times.size() &&
          cycle_elapsed >= capture_times[cycle_capture]) {
        char path[64];
        std::snprintf(path, sizeof(path), "cycle-%02d.bmp", static_cast<int>(cycle_capture));
        save_screenshot(output_w, output_h, path);
        if (cycle_capture == 4) screenshot_requested = true;
        ++cycle_capture;
      }
      if (cycle_test && (motion_test
          ? std::chrono::duration<double>(now - motion_wall_start).count() >= 120.0
          : cycle_elapsed >= test_cycle_duration + 2.0)) running = false;
      if (smoke_test && (total_frames == 37 || total_frames == 45 || total_frames == 47)) {
        std::fprintf(stderr, "frame %d span %.3g render %dx%d output %dx%d kernel %.2fms scale %.2f\n",
                     total_frames, camera.span, render_w, render_h,
                     output_w, output_h, kernel_ms, quality_scale);
      }
      const float menu_opacity = static_cast<float>(menu_progress * menu_progress *
                                                    (3.0 - 2.0 * menu_progress));
      settings.draw(output_w, output_h, menu_opacity,
                    auto_palette, random_palette_order, dive_mode, bloom_enabled,
                    music_enabled, music_muted, music_volume, smoke_test || music.available());
      const float help_opacity = static_cast<float>(help_progress * help_progress * (3.0 - 2.0 * help_progress));
      help.draw(output_w, output_h, help_opacity, false, false, false, false,
                false, false, 0, true, menu_opacity);
      if (smoke_test && total_frames == 45) renderer.check_deep_frame(camera, journey.minimum_span() * 20.0);
      if (smoke_test && total_frames == 47) renderer.check_deep_frame(camera, journey.minimum_span() * 1.1);
      if (smoke_test && total_frames == 47) renderer.check_fallback_queue_lifecycle();
      if (smoke_test && total_frames == 47) {
        // Exercise every visual branch without waiting for the full sequence.
        const double probes[] = {20.0, 68.0, 164.0, 260.0};
        const char* names[] = {"smoke-waves.bmp", "smoke-fold-3.bmp", "smoke-fold-5.bmp", "smoke-fold-7.bmp"};
        for (int probe = 0; probe < 4; ++probe) {
          const auto visual = trip_effects(probes[probe]);
          renderer.draw(camera, palette_from, palette_to,
              static_cast<float>(palette_fade / kPaletteFadeSeconds), 20.0f,
              output_w, output_h, journey.reference_point(), bloom_enabled,
              visual, static_cast<float>(std::fmod(probes[probe] * 0.025, 6.28318530718)), 0.45f, true);
          save_screenshot(output_w, output_h, names[probe]);
        }
        const char* julia_names[] = {"smoke-julia-start.bmp", "smoke-julia-in.bmp",
                                    "smoke-julia-hold.bmp", "smoke-julia-out.bmp", "smoke-julia-return.bmp"};
        const double hold_times[] = {0, kJuliaQuietSeconds + kJuliaRampSeconds * .5,
            kJuliaHoldSeconds * .5, kJuliaPlateauEnd + kJuliaRampSeconds * .5, kJuliaHoldSeconds};
        for (int probe = 0; probe < 5; ++probe) {
          DiveJourney held_journey;
          held_journey.advance(held_journey.dive_seconds() + hold_times[probe]);
          Camera held = held_journey.frame();
          held.rotation = camera.rotation;
          renderer.draw(held, palette_from, palette_to,
              static_cast<float>(palette_fade / kPaletteFadeSeconds), static_cast<float>(color_time),
              output_w, output_h, held_journey.reference_point(), bloom_enabled, {}, 0, 0);
          renderer.check_deep_frame(held, held_journey.minimum_span() * 1.1);
          save_screenshot(output_w, output_h, julia_names[probe]);
          if (probe == 2) {
            CameraTransition exit_probe;
            exit_probe.start(held, camera_at(0));
            Camera leaving = exit_probe.advance(4.0);
            leaving.rotation = camera.rotation;
            renderer.draw(leaving, palette_from, palette_to,
                static_cast<float>(palette_fade / kPaletteFadeSeconds), static_cast<float>(color_time),
                output_w, output_h, held_journey.reference_point(), bloom_enabled, {}, 0, 0);
            save_screenshot(output_w, output_h, "smoke-julia-mode-exit.bmp");
          }
        }
        renderer.draw(camera, palette_from, palette_to,
            static_cast<float>(palette_fade / kPaletteFadeSeconds), static_cast<float>(color_time),
            output_w, output_h, journey.reference_point(), bloom_enabled,
            effects, static_cast<float>(std::fmod(trip_time * 0.025, 6.28318530718)), musical_breath);
      }
      if (screenshot_requested) {
        // Automated captures stay in the test directory, never in Pictures.
        const std::string filename = persist_settings
            ? save_user_screenshot(output_w, output_h) : "cycle-screenshot.bmp";
        if (!persist_settings) save_screenshot(output_w, output_h, filename.c_str());
        if (cycle_test) std::fprintf(stderr, "Saved screenshot: %s\n", filename.c_str());
        SDL_SetWindowTitle(window, ("Saved screenshot: " + filename).c_str());
        title_timer = -3.0;
        screenshot_requested = false;
      }
      if (smoke_test) {
        const char* capture = nullptr;
        switch (total_frames) {
          case 18: capture = "smoke-menu.bmp"; break;
          case 19: capture = "smoke-help-settings.bmp"; break;
          case 25: capture = "smoke-help.bmp"; break;
          case 30: capture = "smoke.bmp"; break;
          case 45: capture = "smoke-travel.bmp"; break;
          case 47: capture = "smoke-deep.bmp"; break;
          default: break;
        }
        if (capture) save_screenshot(output_w, output_h, capture);
      }
      if (smoke_test && total_frames > 50 && !smoke_blend_saved &&
          palette_fade >= kPaletteFadeSeconds * 0.5) {
        save_screenshot(output_w, output_h, "smoke-blend.bmp");
        smoke_blend_saved = true;
      }
      if (smoke_test && total_frames > 50 && !smoke_final_saved &&
          palette_fade >= kPaletteFadeSeconds) {
        save_screenshot(output_w, output_h, "smoke-vapor.bmp");
        smoke_final_saved = true;
      }
      if (smoke_test && smoke_pullback_pending && !smoke_pullback_saved) {
        save_screenshot(output_w, output_h, "smoke-pullback.bmp");
        smoke_pullback_saved = true;
      }
      SDL_GL_SwapWindow(window);
      if (smoke_test && glGetError() != GL_NO_ERROR)
        throw std::runtime_error("OpenGL error during smoke test.");
      ++total_frames;
      if (smoke_test && smoke_pullback_saved) running = false;
      else if (smoke_test && smoke_final_saved && !smoke_pullback_pending) {
        journey.advance(journey.hold_duration() + journey.pullback_seconds() * .5);
        smoke_pullback_pending = true;
      }
      if (smoke_test && total_frames >= 1000) throw std::runtime_error("Palette fade smoke check timed out.");

      if (frame_timing_ready) frame_budget.observe(kernel_ms, frame_work_ms);

      title_timer += dt;
      ++title_frames;
      if (title_timer >= 1.0) {
        char title[256];
        std::snprintf(title, sizeof(title),
            "Mandel Drift | %s | %s%s%s | %s | roll %s | %.0f FPS | render %.1f ms | %.0f%%",
            mode_name(mode), kPaletteNames[palette_to], auto_palette ? " (auto" : "",
            auto_palette ? (random_palette_order ? ", random)" : ")") : "",
            dive_mode ? "Dive" : "Tour",
            rotation_enabled ? "on" : "off",
            title_frames / title_timer, frame_budget.work_ms(), quality_scale * 100.0);
        SDL_SetWindowTitle(window, title);
        title_timer = 0.0;
        title_frames = 0;
      }
    }
    if (cycle_test && !cycle_intervals.empty()) {
      std::sort(cycle_intervals.begin(), cycle_intervals.end());
      double sum = 0.0;
      int late = 0;
      for (double ms : cycle_intervals) { sum += ms; if (ms > 20.0) ++late; }
      std::fprintf(stderr, "Frame intervals: %zu samples, %.2f FPS, median %.2fms, p95 %.2fms, p99 %.2fms, >20ms %.2f%%\n",
          cycle_intervals.size(), cycle_intervals.size() * 1000.0 / sum,
          cycle_intervals[cycle_intervals.size() / 2],
          cycle_intervals[static_cast<size_t>((cycle_intervals.size() - 1) * 0.95)],
          cycle_intervals[static_cast<size_t>((cycle_intervals.size() - 1) * 0.99)],
          late * 100.0 / cycle_intervals.size());
    }
    if (smoke_test || cycle_test) {
      if (valid_timing_samples < 8)
        throw std::runtime_error("Asynchronous GPU timing failed to recycle its frame slots.");
      std::fprintf(stderr, "Async GPU timing: %d valid frame samples\n", valid_timing_samples);
    }
    if (smoke_test && pacing_samples > 0) {
      std::fprintf(stderr, "60 FPS cap: %d intervals, shortest %.3fms, average %.2f FPS\n",
                   pacing_samples, pacing_min_interval / 1.0e6,
                   pacing_samples * 1.0e9 / pacing_interval_sum);
      if (pacing_min_interval < kFrameIntervalNS)
        throw std::runtime_error("60 FPS pacing check failed.");
    }
  }

  SDL_ShowCursor();
  SDL_GL_DestroyContext(context);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const bool smoke_test = argc > 1 && std::string(argv[1]) == "--smoke-test";
  const bool motion_test = argc > 1 && std::string(argv[1]) == "--motion-test";
  const bool cycle_test = motion_test || (argc > 1 && std::string(argv[1]) == "--cycle-test");
  try {
    const bool motion_return = motion_test && argc > 2 && std::string(argv[2]) == "return";
    return run(smoke_test, cycle_test, motion_test, motion_return);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Mandel Drift: %s\n", error.what());
    if (!smoke_test && !cycle_test) SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Mandel Drift", error.what(), nullptr);
    SDL_Quit();
    return 1;
  }
}
