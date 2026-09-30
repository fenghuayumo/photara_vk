struct Push
{
    float4 origin_tmin;
    float4 direction_tmax;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure scene;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> hits;
[[vk::push_constant]] ConstantBuffer<Push> push_constants;

[numthreads(1, 1, 1)]
void main()
{
    RayDesc ray;
    ray.Origin = push_constants.origin_tmin.xyz;
    ray.TMin = push_constants.origin_tmin.w;
    ray.Direction = push_constants.direction_tmax.xyz;
    ray.TMax = push_constants.direction_tmax.w;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
    query.Proceed();
    hits[0] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 1u : 0u;
}
