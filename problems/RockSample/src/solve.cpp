#include "solve.hpp"

#include "RockSampleModel.hpp"          // for RockSampleModel
#include "RockSampleOptions.hpp"        // for RockSampleOptions
#include "TextSerializer.hpp"           // for TextSerializer
int main(int argc, const char* argv[]) {
    RockSampleOptions options;
    return solve<RockSampleModel, TextSerializer>(argc, argv, &options);
}