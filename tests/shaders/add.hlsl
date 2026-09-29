// Adds `addend` to each float in the storage buffer. Used by the photara_vk
// runtime round-trip test.

struct Push
{
    uint count;
    float addend;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer values;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.count) return;
    const uint offset = id.x * 4u;
    const float value = asfloat(values.Load(offset));
    values.Store(offset, asuint(value + pc.addend));
}
