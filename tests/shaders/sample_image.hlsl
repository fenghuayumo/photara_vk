[[vk::binding(0, 0)]] Texture2D<float4> image;
[[vk::binding(1, 0)]] SamplerState image_sampler;
[[vk::binding(2, 0)]] RWByteAddressBuffer output;
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    output.Store4(0, asuint(image.SampleLevel(image_sampler, float2(0.5,0.5), 0)));
}
