#!/usr/bin/env python3
"""
Advanced Test Result Analyzer V2 - With Wave Participant Tracking Semantics
"""

import re
import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class WaveParticipantMismatch:
    """Represents a mismatch in wave participant tracking"""
    lane_id: int
    expected_count: int
    observed_count: int
    
    @property
    def wave_id(self):
        return self.lane_id // 32
    
    @property
    def lane_in_wave(self):
        return self.lane_id % 32
    
    @property
    def mismatch_type(self):
        if self.expected_count > 0 and self.observed_count == 0:
            return 'missing_all'
        elif 0 < self.observed_count < self.expected_count:
            return 'partial'
        elif self.expected_count == 0 and self.observed_count > 0:
            return 'wrong_set'
        else:
            return 'extra'

class WaveParticipantTrackingAnalyzer:
    """Specialized analyzer for WaveParticipantTracking test failures"""
    
    def __init__(self, wave_size: int = 32):
        self.wave_size = wave_size
    
    def analyze_failure(self, expected: List[int], observed: List[int]) -> Dict:
        """Analyze wave participant tracking failure"""
        
        # Extract mismatches
        mismatches = []
        for i, (exp, obs) in enumerate(zip(expected, observed)):
            if exp != obs:
                mismatches.append(WaveParticipantMismatch(i, exp, obs))
        
        # Categorize mismatches
        mismatch_categories = self._categorize_mismatches(mismatches)
        
        # Analyze patterns
        patterns = self._analyze_patterns(expected, observed, mismatches)
        
        # Infer root causes
        root_causes = self._infer_root_causes(mismatch_categories, patterns)
        
        # Generate analysis
        return {
            'summary': self._generate_summary(expected, observed, mismatches),
            'mismatch_categories': mismatch_categories,
            'patterns': patterns,
            'root_causes': root_causes,
            'debugging_hints': self._generate_debugging_hints(root_causes)
        }
    
    def _categorize_mismatches(self, mismatches: List[WaveParticipantMismatch]) -> Dict:
        """Categorize mismatches by type"""
        categories = {
            'missing_all': [],      # Should participate but didn't at all
            'partial': [],          # Participated but not enough times
            'wrong_set': [],        # Shouldn't participate but did
            'extra': []            # Participated too many times
        }
        
        for m in mismatches:
            category = m.mismatch_type
            categories[category].append({
                'lane': m.lane_id,
                'wave': m.wave_id,
                'expected': m.expected_count,
                'observed': m.observed_count
            })
        
        return categories
    
    def _analyze_patterns(self, expected: List[int], observed: List[int], 
                         mismatches: List[WaveParticipantMismatch]) -> Dict:
        """Analyze patterns in the data"""
        patterns = {}
        
        # Check if this looks like a loop pattern
        max_expected = max(expected) if expected else 0
        if max_expected > 1:
            patterns['suggests_loop'] = True
            patterns['max_iterations'] = max_expected
            
            # Check iteration completion rates
            completion_rates = []
            for exp, obs in zip(expected, observed):
                if exp > 0:
                    rate = obs / exp
                    completion_rates.append(rate)
            
            if completion_rates:
                avg_rate = sum(completion_rates) / len(completion_rates)
                patterns['average_completion_rate'] = avg_rate
                patterns['all_or_nothing'] = all(r == 0 or r == 1 for r in completion_rates)
        
        # Check spatial patterns
        patterns['spatial'] = self._check_spatial_patterns(mismatches)
        
        # Check wave-level patterns
        patterns['wave_patterns'] = self._analyze_wave_patterns(mismatches)
        
        return patterns
    
    def _check_spatial_patterns(self, mismatches: List[WaveParticipantMismatch]) -> Dict:
        """Check for spatial patterns in mismatches"""
        if not mismatches:
            return {'type': 'none'}
        
        lane_ids = [m.lane_id for m in mismatches]
        
        # Check for modulo patterns
        for modulo in [2, 4, 8, 16, 32]:
            remainders = [lid % modulo for lid in lane_ids]
            if len(set(remainders)) == 1:
                return {
                    'type': 'modulo',
                    'modulo': modulo,
                    'remainder': remainders[0],
                    'description': f'All affected lanes have ID % {modulo} == {remainders[0]}'
                }
        
        # Check for contiguous blocks
        sorted_lanes = sorted(lane_ids)
        blocks = []
        if sorted_lanes:
            start = sorted_lanes[0]
            end = sorted_lanes[0]
            
            for i in range(1, len(sorted_lanes)):
                if sorted_lanes[i] == end + 1:
                    end = sorted_lanes[i]
                else:
                    if end - start >= 2:
                        blocks.append((start, end))
                    start = end = sorted_lanes[i]
            
            if end - start >= 2:
                blocks.append((start, end))
        
        if blocks:
            return {
                'type': 'contiguous',
                'blocks': blocks,
                'description': f'{len(blocks)} contiguous block(s) of failures'
            }
        
        return {'type': 'scattered', 'description': 'No clear spatial pattern'}
    
    def _analyze_wave_patterns(self, mismatches: List[WaveParticipantMismatch]) -> Dict:
        """Analyze patterns at wave level"""
        wave_stats = defaultdict(lambda: {'total': 0, 'types': Counter()})
        
        for m in mismatches:
            wave_stats[m.wave_id]['total'] += 1
            wave_stats[m.wave_id]['types'][m.mismatch_type] += 1
        
        # Check if all waves have similar patterns
        wave_signatures = []
        for wave_id in sorted(wave_stats.keys()):
            sig = tuple(sorted(wave_stats[wave_id]['types'].items()))
            wave_signatures.append(sig)
        
        unique_signatures = set(wave_signatures)
        
        return {
            'affected_waves': list(wave_stats.keys()),
            'consistent_across_waves': len(unique_signatures) == 1,
            'unique_patterns': len(unique_signatures),
            'per_wave_stats': dict(wave_stats)
        }
    
    def _infer_root_causes(self, categories: Dict, patterns: Dict) -> List[Dict]:
        """Infer root causes from analysis"""
        causes = []
        
        # Check for wrong participant set (highest priority)
        if categories['wrong_set']:
            causes.append({
                'type': 'INCORRECT_PARTICIPANT_CONDITION',
                'confidence': 0.9,
                'description': 'Wave operations include lanes that should not participate',
                'evidence': {
                    'affected_lanes': len(categories['wrong_set']),
                    'details': 'The condition guarding wave operations is incorrect',
                    'examples': categories['wrong_set'][:3]
                },
                'fix_suggestion': 'Review and correct the if condition around wave operations'
            })
        
        # Check for missing participants
        if categories['missing_all']:
            spatial = patterns.get('spatial', {})
            
            if spatial.get('type') == 'modulo':
                causes.append({
                    'type': 'SYSTEMATIC_EXCLUSION',
                    'confidence': 0.85,
                    'description': f'Systematic exclusion based on lane ID pattern',
                    'evidence': {
                        'pattern': spatial['description'],
                        'affected_lanes': len(categories['missing_all']),
                        'modulo': spatial['modulo'],
                        'remainder': spatial['remainder']
                    },
                    'fix_suggestion': f'Check modulo conditions - may have off-by-one error'
                })
            else:
                causes.append({
                    'type': 'CONTROL_FLOW_DIVERGENCE',
                    'confidence': 0.8,
                    'description': 'Some lanes never reach wave operations due to divergent control flow',
                    'evidence': {
                        'affected_lanes': len(categories['missing_all']),
                        'pattern': spatial.get('description', 'scattered')
                    },
                    'fix_suggestion': 'Check nested conditionals and ensure all intended lanes reach wave ops'
                })
        
        # Check for partial participation (loop issues)
        if categories['partial'] and patterns.get('suggests_loop'):
            avg_rate = patterns.get('average_completion_rate', 0)
            
            causes.append({
                'type': 'INCOMPLETE_LOOP_ITERATIONS',
                'confidence': 0.85,
                'description': 'Loops containing wave operations not completing all iterations',
                'evidence': {
                    'affected_lanes': len(categories['partial']),
                    'max_iterations': patterns.get('max_iterations'),
                    'average_completion': f'{avg_rate:.1%}',
                    'examples': categories['partial'][:3]
                },
                'fix_suggestion': 'Check for early loop exits (break/continue) or data-dependent bounds'
            })
        
        # Check for reconvergence issues
        wave_patterns = patterns.get('wave_patterns', {})
        if not wave_patterns.get('consistent_across_waves') and len(wave_patterns.get('affected_waves', [])) > 1:
            causes.append({
                'type': 'WAVE_RECONVERGENCE_ISSUE',
                'confidence': 0.7,
                'description': 'Inconsistent behavior across waves suggests reconvergence problems',
                'evidence': {
                    'affected_waves': wave_patterns.get('affected_waves'),
                    'unique_patterns': wave_patterns.get('unique_patterns')
                },
                'fix_suggestion': 'Check wave operation placement relative to control flow reconvergence points'
            })
        
        # Sort by confidence
        causes.sort(key=lambda x: x['confidence'], reverse=True)
        return causes
    
    def _generate_summary(self, expected: List[int], observed: List[int], 
                         mismatches: List[WaveParticipantMismatch]) -> Dict:
        """Generate summary statistics"""
        total_expected = sum(expected)
        total_observed = sum(observed)
        
        return {
            'total_lanes': len(expected),
            'mismatched_lanes': len(mismatches),
            'mismatch_rate': len(mismatches) / len(expected) if expected else 0,
            'total_expected_participations': total_expected,
            'total_observed_participations': total_observed,
            'participation_accuracy': total_observed / total_expected if total_expected > 0 else 0,
            'first_mismatch_lane': mismatches[0].lane_id if mismatches else None
        }
    
    def _generate_debugging_hints(self, root_causes: List[Dict]) -> List[str]:
        """Generate debugging hints based on root causes"""
        hints = []
        
        if not root_causes:
            return ["Unable to determine specific cause. Review wave operation conditions and placement."]
        
        # Add hints based on primary cause
        primary = root_causes[0]
        hints.append(f"Primary issue: {primary['description']}")
        hints.append(f"Suggested fix: {primary.get('fix_suggestion', 'Review the code')}")
        
        # Add specific hints based on cause type
        cause_type = primary['type']
        
        if cause_type == 'INCORRECT_PARTICIPANT_CONDITION':
            hints.extend([
                "- Review the if condition before wave operations",
                "- Ensure the condition matches the intended participant pattern",
                "- Check for inverted or missing conditions"
            ])
        elif cause_type == 'SYSTEMATIC_EXCLUSION':
            evidence = primary['evidence']
            hints.extend([
                f"- Lanes with ID % {evidence['modulo']} == {evidence['remainder']} are excluded",
                "- Check for off-by-one errors in modulo conditions",
                "- Verify the intended participant pattern"
            ])
        elif cause_type == 'INCOMPLETE_LOOP_ITERATIONS':
            hints.extend([
                "- Check for break or continue statements in loops",
                "- Verify loop bounds are consistent across lanes",
                "- Look for data-dependent exit conditions"
            ])
        
        return hints


# Simplified generic analyzer for other test types
class GenericTestAnalyzer:
    """Generic analyzer for non-WaveParticipantTracking tests"""
    
    def analyze_failure(self, expected: List[int], observed: List[int]) -> Dict:
        """Basic analysis for generic test failures"""
        
        mismatches = []
        for i, (exp, obs) in enumerate(zip(expected, observed)):
            if exp != obs:
                mismatches.append({
                    'lane': i,
                    'expected': exp,
                    'observed': obs,
                    'delta': obs - exp
                })
        
        return {
            'summary': {
                'total_lanes': len(expected),
                'mismatches': len(mismatches),
                'first_mismatch': mismatches[0]['lane'] if mismatches else None
            },
            'mismatches': mismatches[:10],  # First 10
            'pattern': 'generic',
            'root_causes': [{
                'type': 'GENERIC_MISMATCH',
                'confidence': 0.5,
                'description': 'Value mismatch detected - specific analysis depends on test type'
            }]
        }


class TestResultAnalyzerV2:
    """Main analyzer that routes to appropriate specialized analyzer"""
    
    def __init__(self):
        self.wave_analyzer = WaveParticipantTrackingAnalyzer()
        self.generic_analyzer = GenericTestAnalyzer()
    
    def analyze_test(self, test_name: str, expected: List[int], 
                    observed: List[int], metadata: Dict = None) -> Dict:
        """Analyze test based on type"""
        
        # Determine test type from name or metadata
        if 'WaveParticipantTracking' in test_name:
            analyzer = self.wave_analyzer
        else:
            analyzer = self.generic_analyzer
        
        # Perform analysis
        analysis = analyzer.analyze_failure(expected, observed)
        
        # Add test info
        analysis['test_info'] = {
            'name': test_name,
            'type': 'WaveParticipantTracking' if 'WaveParticipantTracking' in test_name else 'generic',
            'metadata': metadata or {}
        }
        
        return analysis


def parse_test_output(test_output: str) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Parse expected and observed arrays from test output"""
    expected = None
    observed = None
    
    # Find Expected block
    expected_match = re.search(r'Expected:.*?Data:\s*\[(.*?)\]', test_output, re.DOTALL)
    if expected_match:
        expected_str = expected_match.group(1)
        expected = [int(x.strip()) for x in expected_str.split(',') if x.strip()]
    
    # Find Got/Observed block
    observed_match = re.search(r'Got:.*?Data:\s*\[(.*?)\]', test_output, re.DOTALL)
    if observed_match:
        observed_str = observed_match.group(1)
        observed = [int(x.strip()) for x in observed_str.split(',') if x.strip()]
    
    return expected, observed


def generate_report(test_name: str, analysis: Dict) -> str:
    """Generate human-readable report from analysis"""
    
    lines = [f"# Test Analysis: {test_name}\n"]
    
    # Summary
    summary = analysis['summary']
    lines.append("## Summary")
    lines.append(f"- Total lanes: {summary['total_lanes']}")
    lines.append(f"- Mismatched lanes: {summary.get('mismatched_lanes', summary.get('mismatches', 0))}")
    
    if 'participation_accuracy' in summary:
        lines.append(f"- Participation accuracy: {summary['participation_accuracy']:.1%}")
    
    # Root causes
    lines.append("\n## Root Cause Analysis")
    for i, cause in enumerate(analysis['root_causes']):
        lines.append(f"\n### {i+1}. {cause['type']} (Confidence: {cause['confidence']:.0%})")
        lines.append(cause['description'])
        
        if 'evidence' in cause:
            lines.append("\nEvidence:")
            evidence = cause['evidence']
            for key, value in evidence.items():
                if key != 'examples':
                    lines.append(f"- {key}: {value}")
        
        if 'fix_suggestion' in cause:
            lines.append(f"\nSuggested fix: {cause['fix_suggestion']}")
    
    # Debugging hints
    if 'debugging_hints' in analysis:
        lines.append("\n## Debugging Hints")
        for hint in analysis['debugging_hints']:
            lines.append(f"{hint}")
    
    return "\n".join(lines)


def main():
    """Example usage and testing"""
    
    # Test with the example data
    test_name = "program_1735064431123456789_1_increment_0_WaveParticipantTracking.test"
    expected = [1, 0, 2, 9, 2, 2, 1, 0, 1, 2, 2, 1, 0, 0, 2, 1, 1, 2,
                1, 0, 1, 2, 2, 1, 0, 0, 2, 0, 1, 2, 1, 0, 1, 2, 2, 1,
                0, 0, 2, 0, 1, 2, 1, 0, 1, 2, 2, 1, 0, 0, 2, 0, 1, 2,
                1, 0, 1, 2, 2, 1, 0, 0, 2, 0]
    observed = [1, 0, 0, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 9,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Analyze
    analyzer = TestResultAnalyzerV2()
    analysis = analyzer.analyze_test(test_name, expected, observed)
    
    # Generate report
    report = generate_report(test_name, analysis)
    print(report)
    
    # Also output JSON for debugging
    print("\n" + "="*60)
    print("Full analysis (JSON):")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()