#pragma once

#include "MiniHLSLInterpreter.h"
#include "MiniHLSLInterpreterFuzzer.h"

namespace minihlsl {
namespace fuzzer {

// TraceCaptureInterpreter - extends MiniHLSLInterpreter with trace capture
class TraceCaptureInterpreter : public minihlsl::interpreter::MiniHLSLInterpreter {
private:
  ExecutionTrace trace_;
  
  // Thread hierarchy tracking
  void recordThreadHierarchy(const interpreter::ThreadgroupContext& tg);
  
  // Block tracking
  void recordBlockExecution(uint32_t blockId, const interpreter::DynamicExecutionBlock& block,
                           const interpreter::ThreadgroupContext& tg);
  
  // Wave operation tracking
  void recordWaveOperation(const std::string& opName, 
                          const std::map<interpreter::WaveId, std::set<interpreter::LaneId>>& participants,
                          const std::map<interpreter::LaneId, interpreter::Value>& inputs,
                          const std::map<interpreter::LaneId, interpreter::Value>& outputs,
                          uint32_t blockId);
  
  // Additional trace capture methods
  void onBarrier(interpreter::ThreadgroupContext& tg);

protected:
  // Hook methods (these may or may not be virtual in base class)
  void onStatementExecute(interpreter::LaneContext &lane, interpreter::WaveContext &wave,
                          interpreter::ThreadgroupContext &tg);
  
  void onStatementComplete(interpreter::LaneContext &lane, interpreter::WaveContext &wave,
                          interpreter::ThreadgroupContext &tg);
  
  void onWaveOpSync(interpreter::WaveContext &wave, interpreter::ThreadgroupContext &tg,
                    const interpreter::SyncPointState &syncState);
  
  void onControlFlow(interpreter::LaneContext &lane, interpreter::WaveContext &wave,
                     interpreter::ThreadgroupContext &tg, const interpreter::Statement *stmt,
                     bool branchTaken);
  
  void onVariableAccess(interpreter::LaneContext &lane, interpreter::WaveContext &wave,
                        interpreter::ThreadgroupContext &tg, const std::string &name,
                        bool isWrite, const interpreter::Value &value);

public:
  TraceCaptureInterpreter();
  
  // Execute program and capture trace
  interpreter::ExecutionResult executeAndCaptureTrace(const interpreter::Program& program,
                                                     const interpreter::ThreadOrdering& ordering,
                                                     uint32_t waveSize = 32);
  
  // Get captured trace
  const ExecutionTrace& getTrace() const { return trace_; }
  ExecutionTrace* getTrace() { return &trace_; }
  
  // Clear trace for new execution
  void clearTrace() { trace_ = ExecutionTrace(); }
};

} // namespace fuzzer
} // namespace minihlsl