#!/usr/bin/env python3
"""
Batch Pattern Analyzer - Process multiple test results with metadata
Generates comprehensive failure pattern reports
"""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from pattern_aware_analyzer import PatternAwareAnalyzer, MetadataLoader, TestMetadata

class BatchPatternAnalyzer:
    """Analyze multiple test results to find pattern correlations"""
    
    def __init__(self, metadata_dir: str):
        self.metadata_dir = metadata_dir
        self.analyzer = PatternAwareAnalyzer()
        self.results = []
        
    def parse_test_results_file(self, results_file: str) -> List[Dict]:
        """Parse a test results file containing multiple tests"""
        
        # Read file (handle UTF-16 if needed)
        with open(results_file, 'rb') as f:
            content = f.read()
        
        try:
            text = content.decode('utf-16-le')
        except:
            text = content.decode('utf-8')
        
        # Split into test blocks
        test_blocks = re.split(r'\n(?=program_)', text)
        
        parsed_tests = []
        for block in test_blocks:
            if not block.strip():
                continue
            
            # Extract test name
            name_match = re.match(r'(program_\d+_.*?\.test)', block)
            if not name_match:
                continue
            
            test_name = name_match.group(1)
            
            # Check if passed or failed
            if 'PASS' in block:
                parsed_tests.append({
                    'test_name': test_name,
                    'status': 'PASS',
                    'raw_output': block
                })
            else:
                # Extract arrays
                expected_match = re.search(r'Expected:.*?Data:\s*\[(.*?)\]', block, re.DOTALL)
                observed_match = re.search(r'Got:.*?Data:\s*\[(.*?)\]', block, re.DOTALL)
                
                if expected_match and observed_match:
                    expected = [int(x.strip()) for x in expected_match.group(1).split(',') if x.strip()]
                    observed = [int(x.strip()) for x in observed_match.group(1).split(',') if x.strip()]
                    
                    parsed_tests.append({
                        'test_name': test_name,
                        'status': 'FAIL',
                        'expected': expected,
                        'observed': observed,
                        'raw_output': block
                    })
        
        return parsed_tests
    
    def analyze_all_tests(self, results_file: str):
        """Analyze all tests in a results file"""
        
        parsed_tests = self.parse_test_results_file(results_file)
        print(f"Found {len(parsed_tests)} tests in results file")
        
        for test_data in parsed_tests:
            if test_data['status'] == 'PASS':
                self.results.append({
                    'test_name': test_data['test_name'],
                    'status': 'PASS',
                    'pattern_id': self._get_pattern_id(test_data['test_name'])
                })
                continue
            
            # Load metadata
            metadata_file = MetadataLoader.find_metadata_for_test(
                test_data['test_name'], self.metadata_dir
            )
            
            if metadata_file:
                metadata = MetadataLoader.load_metadata(metadata_file)
                if metadata:
                    # Analyze with metadata
                    analysis = self.analyzer.analyze_with_metadata(
                        test_data['test_name'],
                        test_data['expected'],
                        test_data['observed'],
                        metadata
                    )
                    self.results.append(analysis)
                else:
                    print(f"Warning: Failed to load metadata for {test_data['test_name']}")
            else:
                # Basic analysis without metadata
                self.results.append({
                    'test_name': test_data['test_name'],
                    'status': 'FAIL',
                    'pattern_id': self._get_pattern_id(test_data['test_name']),
                    'no_metadata': True
                })
    
    def _get_pattern_id(self, test_name: str) -> str:
        """Extract pattern ID from test name"""
        match = re.search(r'P\d{2}', test_name)
        return match.group(0) if match else 'unknown'
    
    def generate_pattern_report(self) -> Dict:
        """Generate comprehensive pattern failure report"""
        
        report = {
            'summary': self._generate_summary(),
            'pattern_analysis': self._analyze_by_pattern(),
            'failure_correlations': self._find_failure_correlations(),
            'mutation_impact': self._analyze_mutation_patterns(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate overall summary statistics"""
        
        total_tests = len(self.results)
        failed_tests = sum(1 for r in self.results if r.get('status') != 'PASS')
        
        # Pattern distribution
        pattern_counts = Counter()
        pattern_failures = Counter()
        
        for result in self.results:
            pattern_id = result.get('pattern', {}).get('id') or result.get('pattern_id', 'unknown')
            pattern_counts[pattern_id] += 1
            
            if result.get('status') != 'PASS':
                pattern_failures[pattern_id] += 1
        
        return {
            'total_tests': total_tests,
            'failed_tests': failed_tests,
            'failure_rate': failed_tests / total_tests if total_tests > 0 else 0,
            'pattern_distribution': dict(pattern_counts),
            'pattern_failure_rates': {
                p: pattern_failures[p] / count
                for p, count in pattern_counts.items()
            }
        }
    
    def _analyze_by_pattern(self) -> Dict:
        """Analyze failures grouped by pattern"""
        
        pattern_groups = defaultdict(list)
        
        for result in self.results:
            if result.get('status') == 'PASS':
                continue
                
            pattern_id = result.get('pattern', {}).get('id') or result.get('pattern_id', 'unknown')
            pattern_groups[pattern_id].append(result)
        
        pattern_analysis = {}
        
        for pattern_id, failures in pattern_groups.items():
            # Collect root causes
            root_causes = []
            for failure in failures:
                if 'root_causes' in failure:
                    root_causes.extend(failure['root_causes'])
            
            # Count cause types
            cause_counts = Counter(cause['type'] for cause in root_causes)
            
            # Find common failure signatures
            signatures = []
            for failure in failures:
                if 'summary' in failure:
                    mismatch_count = failure['summary'].get('mismatch_count', 0)
                    affected_waves = failure['summary'].get('affected_waves', [])
                    signatures.append((mismatch_count, tuple(affected_waves)))
            
            pattern_analysis[pattern_id] = {
                'failure_count': len(failures),
                'common_root_causes': cause_counts.most_common(3),
                'failure_signatures': Counter(signatures).most_common(3),
                'example_tests': [f['test_name'] for f in failures[:3]]
            }
        
        return pattern_analysis
    
    def _find_failure_correlations(self) -> Dict:
        """Find correlations between different failure characteristics"""
        
        correlations = {
            'control_flow_correlations': defaultdict(list),
            'nesting_depth_impact': defaultdict(list),
            'wave_op_placement_impact': defaultdict(list)
        }
        
        for result in self.results:
            if result.get('status') == 'PASS' or 'pattern' not in result:
                continue
            
            pattern = result['pattern']
            summary = result.get('summary', {})
            
            # Control flow type vs failure rate
            control_type = pattern.get('type', 'unknown')
            failure_rate = summary.get('mismatch_rate', 0)
            correlations['control_flow_correlations'][control_type].append(failure_rate)
            
            # Nesting depth impact
            if 'control_flow_analysis' in result:
                depth = result['control_flow_analysis'].get('nesting_depth', 0)
                correlations['nesting_depth_impact'][depth].append(failure_rate)
            
            # Wave op placement
            if 'wave_op_analysis' in result:
                for placement in result['wave_op_analysis'].get('placement_analysis', []):
                    location = placement.get('location', 'unknown')
                    correlations['wave_op_placement_impact'][location].append(failure_rate)
        
        # Calculate averages
        for category in correlations:
            for key, rates in correlations[category].items():
                if rates:
                    correlations[category][key] = {
                        'average_failure_rate': sum(rates) / len(rates),
                        'sample_count': len(rates)
                    }
        
        return correlations
    
    def _analyze_mutation_patterns(self) -> Dict:
        """Analyze impact of different mutation types"""
        
        mutation_impact = defaultdict(lambda: {'total': 0, 'failed': 0})
        
        for result in self.results:
            if 'mutation_impact' not in result:
                continue
            
            for mutation in result['mutation_impact'].get('correlations', []):
                mut_type = mutation.get('type', 'unknown')
                mutation_impact[mut_type]['total'] += 1
                
                if result.get('status') != 'PASS':
                    mutation_impact[mut_type]['failed'] += 1
        
        # Calculate failure rates
        mutation_analysis = {}
        for mut_type, counts in mutation_impact.items():
            if counts['total'] > 0:
                mutation_analysis[mut_type] = {
                    'total_tests': counts['total'],
                    'failed_tests': counts['failed'],
                    'failure_rate': counts['failed'] / counts['total']
                }
        
        return mutation_analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Pattern-specific recommendations
        pattern_analysis = self._analyze_by_pattern()
        for pattern_id, data in pattern_analysis.items():
            if data['failure_count'] > 5:
                causes = data['common_root_causes']
                if causes:
                    top_cause = causes[0][0]
                    recommendations.append(
                        f"Pattern {pattern_id}: Focus on fixing {top_cause} "
                        f"(affects {data['failure_count']} tests)"
                    )
        
        # Control flow recommendations
        correlations = self._find_failure_correlations()
        for control_type, stats in correlations['control_flow_correlations'].items():
            if isinstance(stats, dict) and stats.get('average_failure_rate', 0) > 0.5:
                recommendations.append(
                    f"High failure rate in {control_type} constructs "
                    f"({stats['average_failure_rate']:.0%})"
                )
        
        # Nesting depth recommendations
        for depth, stats in correlations['nesting_depth_impact'].items():
            if isinstance(stats, dict) and depth > 2 and stats.get('average_failure_rate', 0) > 0.3:
                recommendations.append(
                    f"Consider simplifying deeply nested structures "
                    f"(depth {depth} shows {stats['average_failure_rate']:.0%} failure rate)"
                )
        
        return recommendations


def generate_html_report(report: Dict, output_file: str):
    """Generate HTML report from analysis"""
    
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>GPU Test Pattern Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .pass { color: green; }
        .fail { color: red; }
        .warning { color: orange; }
        .recommendation { background-color: #ffffcc; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
"""
    
    # Summary
    summary = report['summary']
    html += f"""
    <h1>GPU Test Pattern Analysis Report</h1>
    
    <h2>Summary</h2>
    <p>Total Tests: {summary['total_tests']}</p>
    <p>Failed Tests: {summary['failed_tests']} ({summary['failure_rate']:.1%})</p>
    
    <h3>Pattern Failure Rates</h3>
    <table>
        <tr><th>Pattern</th><th>Total Tests</th><th>Failure Rate</th></tr>
"""
    
    for pattern, rate in sorted(summary['pattern_failure_rates'].items()):
        status_class = 'fail' if rate > 0.5 else 'warning' if rate > 0.2 else 'pass'
        html += f"""
        <tr>
            <td>{pattern}</td>
            <td>{summary['pattern_distribution'][pattern]}</td>
            <td class="{status_class}">{rate:.1%}</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Pattern-Specific Analysis</h2>
"""
    
    # Pattern analysis
    for pattern_id, data in sorted(report['pattern_analysis'].items()):
        html += f"""
    <h3>Pattern {pattern_id}</h3>
    <p>Failed Tests: {data['failure_count']}</p>
    
    <h4>Common Root Causes:</h4>
    <ul>
"""
        for cause, count in data['common_root_causes']:
            html += f"        <li>{cause}: {count} occurrences</li>\n"
        
        html += """
    </ul>
    
    <h4>Example Tests:</h4>
    <ul>
"""
        for test in data['example_tests']:
            html += f"        <li>{test}</li>\n"
        
        html += "    </ul>\n"
    
    # Recommendations
    html += """
    <h2>Recommendations</h2>
"""
    
    for rec in report['recommendations']:
        html += f'    <div class="recommendation">{rec}</div>\n'
    
    html += """
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML report generated: {output_file}")


def main():
    """Main entry point"""
    
    if len(sys.argv) < 3:
        print("Usage: python3 batch_pattern_analyzer.py <test_results.txt> <metadata_dir> [output_dir]")
        print("\nExample:")
        print("  python3 batch_pattern_analyzer.py test_results.txt metadata/ reports/")
        sys.exit(1)
    
    results_file = sys.argv[1]
    metadata_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Analyze
    print(f"Analyzing test results from {results_file}")
    print(f"Using metadata from {metadata_dir}")
    
    analyzer = BatchPatternAnalyzer(metadata_dir)
    analyzer.analyze_all_tests(results_file)
    
    # Generate report
    report = analyzer.generate_pattern_report()
    
    # Save JSON report
    json_output = f"{output_dir}/pattern_analysis_report.json"
    with open(json_output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved: {json_output}")
    
    # Generate HTML report
    html_output = f"{output_dir}/pattern_analysis_report.html"
    generate_html_report(report, html_output)
    
    # Print summary to console
    print("\n" + "="*60)
    print("PATTERN ANALYSIS SUMMARY")
    print("="*60)
    
    summary = report['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Failed Tests: {summary['failed_tests']} ({summary['failure_rate']:.1%})")
    
    print("\nTop Failing Patterns:")
    for pattern, rate in sorted(summary['pattern_failure_rates'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]:
        if rate > 0:
            print(f"  - {pattern}: {rate:.1%} failure rate")
    
    print("\nRecommendations:")
    for rec in report['recommendations'][:5]:
        print(f"  - {rec}")


if __name__ == "__main__":
    main()