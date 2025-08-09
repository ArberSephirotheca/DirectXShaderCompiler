#include "HLSLProgramGenerator.h"
#include <sstream>

namespace minihlsl {
namespace fuzzer {

// Minimal implementation of serializeProgramToString
std::string serializeProgramToString(const interpreter::Program& program) {
    std::stringstream ss;
    
    // Add global buffer declarations
    for (const auto& buffer : program.globalBuffers) {
        ss << buffer.bufferType << "<";
        ss << interpreter::HLSLTypeInfo::toString(buffer.elementType);
        ss << "> " << buffer.name;
        ss << " : register(" << (buffer.isReadWrite ? "u" : "t") 
           << buffer.registerIndex << ");\n";
    }
    
    if (!program.globalBuffers.empty()) {
        ss << "\n";
    }
    
    // Add thread configuration
    ss << "[numthreads(" << program.numThreadsX << ", " 
       << program.numThreadsY << ", " 
       << program.numThreadsZ << ")]\n";
    
    // Add WaveSize attribute if specified
    if (program.waveSizeMin > 0 || program.waveSizeMax > 0 || program.waveSizePreferred > 0) {
        ss << "[WaveSize(";
        if (program.waveSizeMin > 0) ss << program.waveSizeMin;
        if (program.waveSizeMax > 0) ss << ", " << program.waveSizeMax;
        if (program.waveSizePreferred > 0) ss << ", " << program.waveSizePreferred;
        ss << ")]\n";
    }
    
    ss << "void main(";
    
    // Add function parameters with semantics
    bool first = true;
    for (const auto& param : program.entryInputs.parameters) {
        if (!first) {
            ss << ",\n          ";
        }
        first = false;
        
        // Output type
        ss << interpreter::HLSLTypeInfo::toString(param.type);
        ss << " " << param.name;
        
        // Output semantic
        if (param.semantic != interpreter::HLSLSemantic::None) {
            ss << " : " << interpreter::HLSLSemanticInfo::toString(param.semantic);
        }
    }
    
    ss << ") {\n";
    
    // Add statements
    for (const auto& stmt : program.statements) {
        ss << "  " << stmt->toString() << "\n";
    }
    
    ss << "}\n";
    
    return ss.str();
}

} // namespace fuzzer
} // namespace minihlsl