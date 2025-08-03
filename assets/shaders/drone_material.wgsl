struct Material {
    color: vec4<f32>,
    emissive: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> material: Material;

// Vertex shader
struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec3<f32>, // Change to vec3<f32>
};

@vertex
fn vertex(vin: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vin.position;
    out.color = material.color;
    out.emissive = material.emissive.rgb; // Convert to vec3<f32>
    return out;
}

// Fragment shader
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(in.color.rgb + in.emissive, 1.0); // Ensure proper blending
}