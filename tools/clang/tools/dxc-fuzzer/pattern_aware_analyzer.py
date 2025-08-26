#!/usr/bin/env python3
"""
Pattern-Aware Test Result Analyzer
Integrates test metadata with failure analysis for accurate pattern identification
"""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

@dataclass
class TestMetadata:
    """Parsed test metadata"""
    test_name: str
    pattern_id: str
    pattern_description: str
    control_flow_type: str  # for, while, if, switch, nested
    nesting_depth: int
    wave_operations: List[Dict]  # type, placement, condition
    mutations: List[Dict]  # type, target, description
    thread_config: Dict  # numThreads, waveSize
    
    @classmethod
    def from_json(cls, data: Dict, test_name: str):
        """Create from metadata JSON"""
        pattern = data.get('pattern', {})
        return cls(
            test_name=test_name,
            pattern_id=pattern.get('id', 'unknown'),
            pattern_description=pattern.get('description', ''),
            control_flow_type=pattern.get('type', 'unknown'),
            nesting_depth=pattern.get('depth', 0),
            wave_operations=data.get('waveOperations', []),
            mutations=data.get('mutations', []),
            thread_config=data.get('threadConfig', {})
        )

@dataclass
class PatternFailure:
    """Failure analysis with pattern context"""
    pattern_id: str
    failure_type: str  # mismatch category
    confidence: float
    description: str
    evidence: Dict
    pattern_specific_insights: List[str]

class PatternAwareAnalyzer:
    """Analyzer that understands test patterns and their expected behaviors"""
    
    def __init__(self):
        # Pattern-specific analysis rules
        self.pattern_rules = self._init_pattern_rules()
    
    def _init_pattern_rules(self):
        """Initialize pattern-specific analysis rules"""
        return {
            # Basic patterns
            'P01': {
                'name': 'Simple if statement',
                'expected_behavior': 'Lanes meeting condition participate once',
                'common_failures': ['incorrect_condition', 'missing_participants']
            },
            'P02': {
                'name': 'If-else statement',
                'expected_behavior': 'Exactly one branch executes per lane',
                'common_failures': ['both_branches_execute', 'no_branch_executes']
            },
            'P03': {
                'name': 'Nested if statements',
                'expected_behavior': 'Inner condition only evaluated if outer is true',
                'common_failures': ['incorrect_nesting', 'condition_hoisting']
            },
            
            # Loop patterns
            'P10': {
                'name': 'Simple for loop',
                'expected_behavior': 'Fixed iterations with consistent participation',
                'common_failures': ['incomplete_iterations', 'extra_iterations']
            },
            'P11': {
                'name': 'For loop with break',
                'expected_behavior': 'Early exit for some lanes',
                'common_failures': ['break_not_honored', 'incorrect_exit_condition']
            },
            'P12': {
                'name': 'For loop with continue',
                'expected_behavior': 'Skip iterations but complete loop',
                'common_failures': ['continue_exits_loop', 'skipped_iterations_execute']
            },
            'P15': {
                'name': 'Nested for loops',
                'expected_behavior': 'Inner loop executes fully for each outer iteration',
                'common_failures': ['inner_loop_skip', 'iteration_count_mismatch']
            },
            
            # While patterns
            'P20': {
                'name': 'Simple while loop',
                'expected_behavior': 'Variable iterations based on condition',
                'common_failures': ['infinite_loop', 'premature_exit']
            },
            'P25': {
                'name': 'Nested while loops',
                'expected_behavior': 'Inner while depends on outer state',
                'common_failures': ['state_corruption', 'incorrect_dependency']
            },
            
            # Switch patterns
            'P30': {
                'name': 'Switch statement',
                'expected_behavior': 'Exactly one case per lane',
                'common_failures': ['fallthrough_error', 'no_case_match']
            },
            'P31': {
                'name': 'Switch with fallthrough',
                'expected_behavior': 'Multiple cases execute in sequence',
                'common_failures': ['incorrect_fallthrough', 'missing_break']
            },
            
            # Complex patterns
            'P40': {
                'name': 'Mixed control flow',
                'expected_behavior': 'Combination of different control structures',
                'common_failures': ['interaction_bugs', 'state_inconsistency']
            },
            
            # Wave-specific patterns
            'P50': {
                'name': 'Wave ops in divergent control',
                'expected_behavior': 'Only active lanes participate',
                'common_failures': ['inactive_participation', 'reconvergence_issue']
            }
        }
    
    def analyze_with_metadata(self, test_name: str, expected: List[int], 
                            observed: List[int], metadata: TestMetadata) -> Dict:
        """Analyze test failure with pattern context"""
        
        # Basic mismatch analysis
        mismatches = self._find_mismatches(expected, observed)
        
        # Pattern-specific analysis
        pattern_analysis = self._analyze_for_pattern(
            metadata.pattern_id, mismatches, expected, observed, metadata
        )
        
        # Mutation impact analysis
        mutation_impact = self._analyze_mutation_impact(
            mismatches, metadata.mutations, metadata
        )
        
        # Control flow analysis
        control_flow_analysis = self._analyze_control_flow_impact(
            mismatches, metadata.control_flow_type, metadata.nesting_depth
        )
        
        # Wave operation correlation
        wave_op_analysis = self._analyze_wave_operations(
            mismatches, metadata.wave_operations, expected, observed
        )
        
        # Synthesize results
        return {
            'test_name': test_name,
            'pattern': {
                'id': metadata.pattern_id,
                'description': metadata.pattern_description,
                'type': metadata.control_flow_type
            },
            'summary': self._generate_summary(expected, observed, mismatches),
            'pattern_analysis': pattern_analysis,
            'mutation_impact': mutation_impact,
            'control_flow_analysis': control_flow_analysis,
            'wave_op_analysis': wave_op_analysis,
            'root_causes': self._infer_root_causes(
                pattern_analysis, mutation_impact, control_flow_analysis, wave_op_analysis
            ),
            'debugging_hints': self._generate_debugging_hints(
                metadata, pattern_analysis, mutation_impact
            )
        }
    
    def _find_mismatches(self, expected: List[int], observed: List[int]) -> List[Dict]:
        """Find all mismatches between expected and observed"""
        mismatches = []
        for i, (e, o) in enumerate(zip(expected, observed)):
            if e != o:
                mismatches.append({
                    'lane': i,
                    'expected': e,
                    'observed': o,
                    'delta': o - e,
                    'wave': i // 32,
                    'lane_in_wave': i % 32
                })
        return mismatches
    
    def _analyze_for_pattern(self, pattern_id: str, mismatches: List[Dict],
                           expected: List[int], observed: List[int], 
                           metadata: TestMetadata) -> Dict:
        """Pattern-specific failure analysis"""
        
        pattern_info = self.pattern_rules.get(pattern_id, {})
        analysis = {
            'pattern_id': pattern_id,
            'pattern_name': pattern_info.get('name', 'Unknown pattern'),
            'expected_behavior': pattern_info.get('expected_behavior', ''),
            'failure_matches_known_issues': False,
            'specific_failure': None
        }
        
        # Check against known failure modes
        known_failures = pattern_info.get('common_failures', [])
        
        # Pattern-specific checks
        if pattern_id in ['P10', 'P11', 'P12', 'P15']:  # For loop patterns
            analysis.update(self._analyze_loop_pattern(
                pattern_id, mismatches, expected, observed, metadata
            ))
        elif pattern_id in ['P20', 'P25']:  # While loop patterns
            analysis.update(self._analyze_while_pattern(
                pattern_id, mismatches, expected, observed, metadata
            ))
        elif pattern_id in ['P30', 'P31']:  # Switch patterns
            analysis.update(self._analyze_switch_pattern(
                pattern_id, mismatches, expected, observed, metadata
            ))
        elif pattern_id in ['P01', 'P02', 'P03']:  # If patterns
            analysis.update(self._analyze_if_pattern(
                pattern_id, mismatches, expected, observed, metadata
            ))
        
        return analysis
    
    def _analyze_loop_pattern(self, pattern_id: str, mismatches: List[Dict],
                            expected: List[int], observed: List[int],
                            metadata: TestMetadata) -> Dict:
        """Analyze for-loop specific failures"""
        
        max_expected = max(expected) if expected else 0
        max_observed = max(observed) if observed else 0
        
        analysis = {
            'loop_type': 'for',
            'expected_iterations': max_expected,
            'max_observed_iterations': max_observed
        }
        
        # Check for incomplete iterations
        if max_observed < max_expected:
            incomplete_lanes = [m['lane'] for m in mismatches if m['delta'] < 0]
            analysis['specific_failure'] = 'INCOMPLETE_ITERATIONS'
            analysis['incomplete_lanes'] = incomplete_lanes
            analysis['average_completion'] = max_observed / max_expected
        
        # Check for break/continue issues
        if pattern_id in ['P11', 'P12']:
            # Look for patterns indicating break/continue problems
            if pattern_id == 'P11' and any(m['observed'] > 0 for m in mismatches):
                analysis['specific_failure'] = 'BREAK_NOT_EFFECTIVE'
            elif pattern_id == 'P12':
                # Continue should skip some iterations but not exit
                if all(m['observed'] == 0 for m in mismatches if m['expected'] > 0):
                    analysis['specific_failure'] = 'CONTINUE_EXITS_LOOP'
        
        # Nested loop analysis
        if pattern_id == 'P15':
            # Check if inner loop iterations are consistent
            if len(set(expected)) > 2:  # More than 2 unique values suggests nested behavior
                analysis['nested_behavior_detected'] = True
                analysis['unique_iteration_counts'] = sorted(set(expected))
        
        return analysis
    
    def _analyze_while_pattern(self, pattern_id: str, mismatches: List[Dict],
                             expected: List[int], observed: List[int],
                             metadata: TestMetadata) -> Dict:
        """Analyze while-loop specific failures"""
        
        analysis = {
            'loop_type': 'while',
            'convergence_issue': False
        }
        
        # While loops often have variable iteration counts
        expected_counts = Counter(expected)
        observed_counts = Counter(observed)
        
        if len(expected_counts) > 1:
            analysis['variable_iterations'] = True
            analysis['iteration_distribution'] = dict(expected_counts)
        
        # Check for infinite loop indicators
        if all(o == 0 for o in observed) and any(e > 0 for e in expected):
            analysis['specific_failure'] = 'LOOP_NEVER_ENTERED'
        elif any(o > e * 2 for o, e in zip(observed, expected) if e > 0):
            analysis['specific_failure'] = 'POSSIBLE_INFINITE_LOOP'
        
        return analysis
    
    def _analyze_switch_pattern(self, pattern_id: str, mismatches: List[Dict],
                              expected: List[int], observed: List[int],
                              metadata: TestMetadata) -> Dict:
        """Analyze switch-statement specific failures"""
        
        analysis = {
            'control_type': 'switch',
            'case_distribution': Counter(expected)
        }
        
        # Check for fallthrough issues
        if pattern_id == 'P31':
            # Fallthrough should result in multiple executions
            if all(o <= 1 for o in observed):
                analysis['specific_failure'] = 'FALLTHROUGH_NOT_WORKING'
        else:
            # Non-fallthrough should execute exactly once
            if any(o > 1 for o in observed if o > 0):
                analysis['specific_failure'] = 'UNEXPECTED_FALLTHROUGH'
        
        # Check for missing case coverage
        if any(e > 0 and o == 0 for e, o in zip(expected, observed)):
            analysis['specific_failure'] = 'CASE_NOT_REACHED'
        
        return analysis
    
    def _analyze_if_pattern(self, pattern_id: str, mismatches: List[Dict],
                          expected: List[int], observed: List[int],
                          metadata: TestMetadata) -> Dict:
        """Analyze if-statement specific failures"""
        
        analysis = {
            'control_type': 'if',
            'branch_coverage': {}
        }
        
        # Simple if (P01)
        if pattern_id == 'P01':
            true_lanes = [i for i, e in enumerate(expected) if e > 0]
            executed_lanes = [i for i, o in enumerate(observed) if o > 0]
            
            if set(executed_lanes) != set(true_lanes):
                analysis['specific_failure'] = 'CONDITION_MISMATCH'
                analysis['extra_lanes'] = list(set(executed_lanes) - set(true_lanes))
                analysis['missing_lanes'] = list(set(true_lanes) - set(executed_lanes))
        
        # If-else (P02)
        elif pattern_id == 'P02':
            # Both branches should be exclusive
            if any(o > 1 for o in observed):
                analysis['specific_failure'] = 'BOTH_BRANCHES_EXECUTE'
            elif any(e > 0 and o == 0 for e, o in zip(expected, observed)):
                analysis['specific_failure'] = 'BRANCH_NOT_TAKEN'
        
        # Nested if (P03)
        elif pattern_id == 'P03':
            # Check for condition hoisting
            if sum(observed) > sum(expected):
                analysis['specific_failure'] = 'CONDITION_HOISTING'
        
        return analysis
    
    def _analyze_mutation_impact(self, mismatches: List[Dict], 
                               mutations: List[Dict], metadata: TestMetadata) -> Dict:
        """Analyze how mutations correlate with failures"""
        
        if not mutations:
            return {'no_mutations': True}
        
        analysis = {
            'mutation_count': len(mutations),
            'mutation_types': [m.get('type', 'unknown') for m in mutations],
            'correlations': []
        }
        
        # Check each mutation's potential impact
        for mutation in mutations:
            mut_type = mutation.get('type', '')
            target = mutation.get('target', '')
            
            correlation = {
                'type': mut_type,
                'target': target,
                'likely_impact': 'unknown'
            }
            
            # Analyze based on mutation type
            if 'WaveParticipantTracking' in mut_type or 'WaveParticipantBitTracking' in mut_type:
                # These mutations add tracking - failures indicate incorrect participation
                correlation['likely_impact'] = 'incorrect_wave_participation'
                
            elif 'ContextAware' in mut_type:
                # These add iteration-specific behavior
                if any(m['delta'] != 0 for m in mismatches):
                    correlation['likely_impact'] = 'iteration_specific_failure'
                    
            elif 'NestedControl' in mut_type:
                # These modify control flow
                correlation['likely_impact'] = 'control_flow_alteration'
            
            analysis['correlations'].append(correlation)
        
        return analysis
    
    def _analyze_control_flow_impact(self, mismatches: List[Dict],
                                   control_type: str, nesting_depth: int) -> Dict:
        """Analyze impact of control flow structure on failures"""
        
        analysis = {
            'control_type': control_type,
            'nesting_depth': nesting_depth,
            'complexity_correlation': None
        }
        
        # Higher nesting often correlates with more failures
        if nesting_depth > 2:
            analysis['high_complexity'] = True
            if len(mismatches) > len(mismatches) * 0.5:
                analysis['complexity_correlation'] = 'high_nesting_high_failure'
        
        # Check for wave boundary issues in nested structures
        wave_boundaries = set()
        for m in mismatches:
            if m['lane_in_wave'] in [0, 31]:  # Wave boundaries
                wave_boundaries.add(m['wave'])
        
        if wave_boundaries:
            analysis['wave_boundary_issues'] = True
            analysis['affected_waves'] = list(wave_boundaries)
        
        return analysis
    
    def _analyze_wave_operations(self, mismatches: List[Dict],
                               wave_operations: List[Dict],
                               expected: List[int], observed: List[int]) -> Dict:
        """Analyze wave operation behavior"""
        
        if not wave_operations:
            return {'no_wave_operations': True}
        
        analysis = {
            'wave_op_count': len(wave_operations),
            'wave_op_types': [op.get('type', 'unknown') for op in wave_operations],
            'placement_analysis': []
        }
        
        # Analyze each wave operation's placement
        for op in wave_operations:
            placement = {
                'type': op.get('type'),
                'location': op.get('placement', 'unknown'),
                'condition': op.get('condition', 'none'),
                'execution_pattern': self._infer_execution_pattern(expected)
            }
            
            # Check if mismatches correlate with wave op placement
            if op.get('placement') == 'loop_body':
                if any(m['delta'] < 0 for m in mismatches):
                    placement['issue'] = 'incomplete_loop_iterations'
            elif op.get('placement') == 'conditional':
                if mismatches:
                    placement['issue'] = 'condition_evaluation_mismatch'
            
            analysis['placement_analysis'].append(placement)
        
        return analysis
    
    def _infer_execution_pattern(self, values: List[int]) -> str:
        """Infer execution pattern from values"""
        unique_values = set(values)
        
        if len(unique_values) == 1:
            return 'uniform'
        elif len(unique_values) == 2 and 0 in unique_values:
            return 'conditional'
        elif max(unique_values) > 1:
            return 'iterative'
        else:
            return 'mixed'
    
    def _infer_root_causes(self, pattern_analysis: Dict, mutation_impact: Dict,
                         control_flow_analysis: Dict, wave_op_analysis: Dict) -> List[Dict]:
        """Synthesize analyses to infer root causes"""
        
        causes = []
        
        # Pattern-specific failure
        if pattern_analysis.get('specific_failure'):
            causes.append({
                'type': pattern_analysis['specific_failure'],
                'confidence': 0.9,
                'description': f"Pattern {pattern_analysis['pattern_id']} shows {pattern_analysis['specific_failure']}",
                'evidence': {
                    'pattern': pattern_analysis['pattern_name'],
                    'expected_behavior': pattern_analysis['expected_behavior']
                }
            })
        
        # Mutation-induced failure
        if mutation_impact.get('correlations'):
            for corr in mutation_impact['correlations']:
                if corr['likely_impact'] != 'unknown':
                    causes.append({
                        'type': f"MUTATION_{corr['type']}",
                        'confidence': 0.8,
                        'description': f"Mutation {corr['type']} caused {corr['likely_impact']}",
                        'evidence': corr
                    })
        
        # Control flow complexity
        if control_flow_analysis.get('complexity_correlation'):
            causes.append({
                'type': 'HIGH_COMPLEXITY',
                'confidence': 0.7,
                'description': f"High nesting depth ({control_flow_analysis['nesting_depth']}) correlates with failures",
                'evidence': control_flow_analysis
            })
        
        # Wave operation issues
        for placement in wave_op_analysis.get('placement_analysis', []):
            if placement.get('issue'):
                causes.append({
                    'type': f"WAVE_OP_{placement['issue'].upper()}",
                    'confidence': 0.85,
                    'description': f"Wave operation in {placement['location']} has {placement['issue']}",
                    'evidence': placement
                })
        
        # Sort by confidence
        causes.sort(key=lambda x: x['confidence'], reverse=True)
        return causes[:3]  # Top 3 causes
    
    def _generate_debugging_hints(self, metadata: TestMetadata,
                                pattern_analysis: Dict, mutation_impact: Dict) -> List[str]:
        """Generate pattern-specific debugging hints"""
        
        hints = []
        pattern_id = metadata.pattern_id
        
        # Pattern-specific hints
        if pattern_id in self.pattern_rules:
            pattern_info = self.pattern_rules[pattern_id]
            hints.append(f"Pattern {pattern_id} ({pattern_info['name']})")
            hints.append(f"Expected: {pattern_info['expected_behavior']}")
        
        # Failure-specific hints
        specific_failure = pattern_analysis.get('specific_failure')
        if specific_failure:
            if specific_failure == 'INCOMPLETE_ITERATIONS':
                hints.extend([
                    "Check for early loop exits (break statements)",
                    "Verify loop bounds are consistent across lanes",
                    "Look for data-dependent conditions affecting iteration count"
                ])
            elif specific_failure == 'BREAK_NOT_EFFECTIVE':
                hints.extend([
                    "Verify break statement is properly placed",
                    "Check if break is inside correct loop level",
                    "Ensure condition for break is evaluated correctly"
                ])
            elif specific_failure == 'CONDITION_MISMATCH':
                hints.extend([
                    "Review the if condition logic",
                    "Check for lane-specific condition evaluation",
                    "Verify condition doesn't have side effects"
                ])
            elif specific_failure == 'CASE_NOT_REACHED':
                hints.extend([
                    "Check switch expression evaluation",
                    "Verify all expected cases are covered",
                    "Look for missing break statements causing fallthrough"
                ])
        
        # Mutation-specific hints
        for corr in mutation_impact.get('correlations', []):
            if 'WaveParticipant' in corr.get('type', ''):
                hints.append("Wave participant tracking detected incorrect participation")
            elif 'ContextAware' in corr.get('type', ''):
                hints.append("Context-aware mutation shows iteration-specific failures")
        
        return hints
    
    def _generate_summary(self, expected: List[int], observed: List[int],
                        mismatches: List[Dict]) -> Dict:
        """Generate summary statistics"""
        
        return {
            'total_lanes': len(expected),
            'mismatch_count': len(mismatches),
            'mismatch_rate': len(mismatches) / len(expected) if expected else 0,
            'first_mismatch_lane': mismatches[0]['lane'] if mismatches else None,
            'affected_waves': sorted(set(m['wave'] for m in mismatches))
        }


class MetadataLoader:
    """Load and parse test metadata files"""
    
    @staticmethod
    def load_metadata(metadata_file: str) -> Optional[TestMetadata]:
        """Load metadata from JSON file"""
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            test_name = Path(metadata_file).stem.replace('.meta', '')
            return TestMetadata.from_json(data, test_name)
        except Exception as e:
            print(f"Error loading metadata from {metadata_file}: {e}")
            return None
    
    @staticmethod
    def find_metadata_for_test(test_name: str, metadata_dir: str) -> Optional[str]:
        """Find metadata file for a test"""
        metadata_path = Path(metadata_dir)
        
        # Try exact match
        meta_file = metadata_path / f"{test_name}.meta.json"
        if meta_file.exists():
            return str(meta_file)
        
        # Try without .test extension
        if test_name.endswith('.test'):
            meta_file = metadata_path / f"{test_name[:-5]}.meta.json"
            if meta_file.exists():
                return str(meta_file)
        
        return None


def analyze_test_with_metadata(test_output: str, metadata_dir: str) -> Dict:
    """Main entry point for pattern-aware analysis"""
    
    # Extract test name from output
    test_name_match = re.match(r'(program_\d+_.*?\.test)', test_output)
    if not test_name_match:
        return {'error': 'Could not extract test name from output'}
    
    test_name = test_name_match.group(1)
    
    # Parse expected and observed arrays
    expected_match = re.search(r'Expected:.*?Data:\s*\[(.*?)\]', test_output, re.DOTALL)
    observed_match = re.search(r'Got:.*?Data:\s*\[(.*?)\]', test_output, re.DOTALL)
    
    if not expected_match or not observed_match:
        return {'error': 'Could not parse expected/observed arrays'}
    
    expected = [int(x.strip()) for x in expected_match.group(1).split(',') if x.strip()]
    observed = [int(x.strip()) for x in observed_match.group(1).split(',') if x.strip()]
    
    # Load metadata
    metadata_file = MetadataLoader.find_metadata_for_test(test_name, metadata_dir)
    if not metadata_file:
        return {
            'error': f'No metadata found for {test_name}',
            'test_name': test_name,
            'basic_analysis': {
                'mismatches': sum(1 for e, o in zip(expected, observed) if e != o),
                'total_lanes': len(expected)
            }
        }
    
    metadata = MetadataLoader.load_metadata(metadata_file)
    if not metadata:
        return {'error': f'Failed to load metadata from {metadata_file}'}
    
    # Perform pattern-aware analysis
    analyzer = PatternAwareAnalyzer()
    return analyzer.analyze_with_metadata(test_name, expected, observed, metadata)


def main():
    """Example usage"""
    
    if len(sys.argv) < 3:
        print("Usage: python3 pattern_aware_analyzer.py <test_output_file> <metadata_dir>")
        print("\nExample:")
        print("  python3 pattern_aware_analyzer.py failed_test.txt metadata/")
        sys.exit(1)
    
    test_output_file = sys.argv[1]
    metadata_dir = sys.argv[2]
    
    # Read test output
    with open(test_output_file, 'r') as f:
        test_output = f.read()
    
    # Analyze
    result = analyze_test_with_metadata(test_output, metadata_dir)
    
    # Output results
    print(json.dumps(result, indent=2))
    
    # Generate human-readable report
    if 'error' not in result:
        print("\n" + "="*60)
        print("PATTERN-AWARE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nTest: {result['test_name']}")
        print(f"Pattern: {result['pattern']['id']} - {result['pattern']['description']}")
        print(f"Control Flow: {result['pattern']['type']}")
        
        summary = result['summary']
        print(f"\nMismatches: {summary['mismatch_count']}/{summary['total_lanes']} lanes")
        
        print("\nRoot Causes:")
        for i, cause in enumerate(result['root_causes'], 1):
            print(f"{i}. {cause['type']} (confidence: {cause['confidence']:.0%})")
            print(f"   {cause['description']}")
        
        print("\nDebugging Hints:")
        for hint in result['debugging_hints']:
            print(f"- {hint}")


if __name__ == "__main__":
    main()