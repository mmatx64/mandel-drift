#pragma once

// Threshold before downsampling so small bright details contribute consistently.
constexpr const char* kBloomExtractShader = R"GLSL(#version 330 core
in vec2 uv;
out vec4 frag_color;
uniform sampler2D source_color;
vec3 highlight(vec2 point) {
  vec3 color = texture(source_color, point).rgb;
  float peak = max(color.r, max(color.g, color.b));
  return color * smoothstep(0.55, 0.95, peak);
}
void main() {
  vec2 step = 1.0 / vec2(textureSize(source_color, 0));
  vec3 color = highlight(uv + step * vec2(-1.0, -1.0));
  color += highlight(uv + step * vec2(1.0, -1.0));
  color += highlight(uv + step * vec2(-1.0, 1.0));
  color += highlight(uv + step * vec2(1.0, 1.0));
  frag_color = vec4(color * 0.25, 1.0);
}
)GLSL";

constexpr const char* kBloomBlurShader = R"GLSL(#version 330 core
in vec2 uv;
out vec4 frag_color;
uniform sampler2D source_color;
uniform vec2 direction;
void main() {
  vec3 color = texture(source_color, uv).rgb * 0.227027;
  color += texture(source_color, uv + direction * 1.384615).rgb * 0.316216;
  color += texture(source_color, uv - direction * 1.384615).rgb * 0.316216;
  color += texture(source_color, uv + direction * 3.230769).rgb * 0.070270;
  color += texture(source_color, uv - direction * 3.230769).rgb * 0.070270;
  frag_color = vec4(color, 1.0);
}
)GLSL";

constexpr const char* kBloomCompositeShader = R"GLSL(#version 330 core
in vec2 uv;
out vec4 frag_color;
uniform sampler2D scene_color;
uniform sampler2D glow_color;
void main() {
  vec3 scene = texture(scene_color, uv).rgb;
  vec3 glow = texture(glow_color, uv).rgb;
  // A restrained screen blend preserves saturated highlights and crisp detail.
  frag_color = vec4(scene + 0.22 * glow * (1.0 - scene), 1.0);
}
)GLSL";
