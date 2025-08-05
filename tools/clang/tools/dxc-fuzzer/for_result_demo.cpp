#include "MiniHLSLInterpreter.h"
#include <iostream>

using namespace minihlsl::interpreter;

int main() {
    std::cout << "=== Phase-based Result ForStmt Demo ===" << std::endl;
    
    // Create a simple for loop: for (i = 0; i < 3; i++) { x = x + i; }
    LaneContext lane;
    WaveContext wave;
    ThreadgroupContext tg;
    
    lane.laneId = 0;
    lane.isActive = true;
    lane.variables["x"] = Value(0); // Initialize x to 0
    
    // Create for loop components
    auto init = std::make_unique<LiteralExpr>(Value(0));           // i = 0
    auto condition = std::make_unique<BinaryExpr>(                 // i < 3
        BinaryOp::Lt,
        std::make_unique<VariableExpr>("i"),
        std::make_unique<LiteralExpr>(Value(3))
    );
    auto increment = std::make_unique<BinaryExpr>(                 // i + 1
        BinaryOp::Add,
        std::make_unique<VariableExpr>("i"),
        std::make_unique<LiteralExpr>(Value(1))
    );
    
    // Create loop body: x = x + i
    auto bodyExpr = std::make_unique<BinaryExpr>(
        BinaryOp::Add,
        std::make_unique<VariableExpr>("x"),
        std::make_unique<VariableExpr>("i")
    );
    auto bodyStmt = std::make_unique<AssignStmt>("x", std::move(bodyExpr));
    
    std::vector<std::unique_ptr<Statement>> body;
    body.push_back(std::move(bodyStmt));
    
    // Create the for statement
    auto forStmt = std::make_unique<ForStmt>("i", std::move(init), std::move(condition), 
                                           std::move(increment), std::move(body));
    
    std::cout << "\n1. Executing Exception-based ForStmt:" << std::endl;
    std::cout << "   Initial x = " << lane.variables["x"].asInt() << std::endl;
    
    try {
        forStmt->execute(lane, wave, tg);
        std::cout << "   Exception-based execution completed" << std::endl;
        std::cout << "   Final x = " << lane.variables["x"].asInt() << std::endl;
        std::cout << "   Final i = " << lane.variables["i"].asInt() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "   Exception caught: " << e.what() << std::endl;
    }
    
    std::cout << "\n2. Executing Result-based ForStmt:" << std::endl;
    
    // Reset for Result-based test
    lane.variables["x"] = Value(0);
    lane.variables.erase("i");
    
    // Re-create the for statement since it was moved
    auto init2 = std::make_unique<LiteralExpr>(Value(0));
    auto condition2 = std::make_unique<BinaryExpr>(
        BinaryOp::Lt,
        std::make_unique<VariableExpr>("i"),
        std::make_unique<LiteralExpr>(Value(3))
    );
    auto increment2 = std::make_unique<BinaryExpr>(
        BinaryOp::Add,
        std::make_unique<VariableExpr>("i"),
        std::make_unique<LiteralExpr>(Value(1))
    );
    
    auto bodyExpr2 = std::make_unique<BinaryExpr>(
        BinaryOp::Add,
        std::make_unique<VariableExpr>("x"),
        std::make_unique<VariableExpr>("i")
    );
    auto bodyStmt2 = std::make_unique<AssignStmt>("x", std::move(bodyExpr2));
    
    std::vector<std::unique_ptr<Statement>> body2;
    body2.push_back(std::move(bodyStmt2));
    
    auto forStmt2 = std::make_unique<ForStmt>("i", std::move(init2), std::move(condition2), 
                                            std::move(increment2), std::move(body2));
    
    std::cout << "   Initial x = " << lane.variables["x"].asInt() << std::endl;
    
    // Execute using Result-based approach
    auto result = forStmt2->execute_result(lane, wave, tg);
    if (result.is_ok()) {
        std::cout << "   Result-based execution: SUCCESS" << std::endl;
        std::cout << "   Final x = " << lane.variables["x"].asInt() << std::endl;
        std::cout << "   Final i = " << lane.variables["i"].asInt() << std::endl;
    } else {
        std::cout << "   Result-based execution: ERROR" << std::endl;
        return 1;
    }
    
    std::cout << "\n3. Result-based ForStmt with Break:" << std::endl;
    
    // Reset variables
    lane.variables["x"] = Value(0);
    lane.variables.erase("i");
    
    // Create for loop with break inside
    auto init3 = std::make_unique<LiteralExpr>(Value(0));
    auto condition3 = std::make_unique<BinaryExpr>(
        BinaryOp::Lt,
        std::make_unique<VariableExpr>("i"),
        std::make_unique<LiteralExpr>(Value(10))
    );
    auto increment3 = std::make_unique<BinaryExpr>(
        BinaryOp::Add,
        std::make_unique<VariableExpr>("i"),
        std::make_unique<LiteralExpr>(Value(1))
    );
    
    // Create body with conditional break
    std::vector<std::unique_ptr<Statement>> body3;
    
    // Add x = x + i
    auto bodyExpr3 = std::make_unique<BinaryExpr>(
        BinaryOp::Add,
        std::make_unique<VariableExpr>("x"),
        std::make_unique<VariableExpr>("i")
    );
    body3.push_back(std::make_unique<AssignStmt>("x", std::move(bodyExpr3)));
    
    // Add break when i >= 2
    auto breakCondition = std::make_unique<BinaryExpr>(
        BinaryOp::Ge,
        std::make_unique<VariableExpr>("i"),
        std::make_unique<LiteralExpr>(Value(2))
    );
    
    std::vector<std::unique_ptr<Statement>> breakBlock;
    breakBlock.push_back(std::make_unique<BreakStmt>());
    
    auto ifBreak = std::make_unique<IfStmt>(std::move(breakCondition), std::move(breakBlock));
    body3.push_back(std::move(ifBreak));
    
    auto forStmt3 = std::make_unique<ForStmt>("i", std::move(init3), std::move(condition3), 
                                            std::move(increment3), std::move(body3));
    
    std::cout << "   Initial x = " << lane.variables["x"].asInt() << std::endl;
    
    auto result3 = forStmt3->execute_result(lane, wave, tg);
    if (result3.is_ok()) {
        std::cout << "   Result-based execution with break: SUCCESS" << std::endl;
        std::cout << "   Final x = " << lane.variables["x"].asInt() << " (should be 0+1+2=3)" << std::endl;
        std::cout << "   Final i = " << lane.variables["i"].asInt() << " (should be 3)" << std::endl;
    } else {
        std::cout << "   Result-based execution with break: completed via control flow" << std::endl;
    }
    
    std::cout << "\n=== Phase-based Result Demo completed! ===" << std::endl;
    std::cout << "Key insight: Each phase (init, condition, body, increment) is now" << std::endl;
    std::cout << "a separate Result-returning function, enabling clean error handling" << std::endl;
    std::cout << "and composable control flow without exceptions!" << std::endl;
    
    return 0;
}