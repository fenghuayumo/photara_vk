float4 main(uint id : SV_VertexID) : SV_Position
{
    const float2 pos[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0),
    };
    return float4(pos[id], 0.0, 1.0);
}
