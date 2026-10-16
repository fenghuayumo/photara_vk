#include <photara_vk/photara_vk.hpp>
#include <stdexcept>
int main() {
    // Exercises installed headers and linked API without requiring a GPU.
    photara::vk::PipelineCache cache;
    try {
        (void)cache.data();
    } catch (const std::logic_error&) {
        return 0;
    }
    return 1;
}
