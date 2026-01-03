#!/usr/bin/env python3
"""
Analyze evaluation results from Step-JEPA model.

Usage:
    python analyze_evaluation.py evaluation_results.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_results(result_file):
    """Load evaluation results"""
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def analyze_results(results):
    """Comprehensive analysis of evaluation results"""
    total = len(results)
    correct = sum(1 for r in results if r.get('correct', False))
    incorrect = total - correct
    
    print("=" * 80)
    print("EVALUATION ANALYSIS")
    print("=" * 80)
    
    # Overall accuracy
    print(f"\n📊 Overall Performance:")
    print(f"  Total examples: {total}")
    print(f"  Correct: {correct} ({correct/total*100:.2f}%)")
    print(f"  Incorrect: {incorrect} ({incorrect/total*100:.2f}%)")
    
    # Error analysis
    errors = [r for r in results if not r.get('correct', False)]
    
    if errors:
        print(f"\n❌ Error Analysis ({len(errors)} errors):")
        
        # Categorize errors
        no_answer_extracted = sum(1 for e in errors if e.get('gen_answer') is None)
        wrong_answer = len(errors) - no_answer_extracted
        
        print(f"  Failed to extract answer: {no_answer_extracted}")
        print(f"  Wrong answer: {wrong_answer}")
        
        # Response length analysis
        error_lengths = [len(e.get('generated_response', '')) for e in errors]
        avg_error_length = sum(error_lengths) / len(error_lengths) if error_lengths else 0
        
        correct_responses = [r for r in results if r.get('correct', False)]
        correct_lengths = [len(r.get('generated_response', '')) for r in correct_responses]
        avg_correct_length = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0
        
        print(f"\n📏 Response Length:")
        print(f"  Correct answers: {avg_correct_length:.1f} chars (avg)")
        print(f"  Incorrect answers: {avg_error_length:.1f} chars (avg)")
    
    # Answer distribution
    gen_answers = [r.get('gen_answer') for r in results if r.get('gen_answer') is not None]
    gt_answers = [r.get('gt_answer') for r in results if r.get('gt_answer') is not None]
    
    print(f"\n📈 Answer Statistics:")
    print(f"  Generated answers extracted: {len(gen_answers)}/{total} ({len(gen_answers)/total*100:.1f}%)")
    print(f"  Ground truth answers: {len(gt_answers)}/{total}")
    
    # Numeric vs non-numeric
    numeric_gen = sum(1 for a in gen_answers if a.replace('.', '').replace('-', '').isdigit())
    print(f"  Numeric answers generated: {numeric_gen}/{len(gen_answers)} ({numeric_gen/len(gen_answers)*100:.1f}%)")
    
    # Show some examples
    print("\n" + "=" * 80)
    print("SAMPLE CORRECT PREDICTIONS")
    print("=" * 80)
    correct_samples = [r for r in results if r.get('correct', False)][:3]
    for i, sample in enumerate(correct_samples, 1):
        print(f"\n{i}. Question: {sample['question'][:100]}...")
        print(f"   Answer: {sample['gen_answer']}")
        print(f"   Response preview: {sample['generated_response'][:150]}...")
    
    print("\n" + "=" * 80)
    print("SAMPLE ERRORS")
    print("=" * 80)
    error_samples = errors[:3]
    for i, sample in enumerate(error_samples, 1):
        print(f"\n{i}. Question: {sample['question'][:100]}...")
        print(f"   Expected: {sample.get('gt_answer', 'N/A')}")
        print(f"   Generated: {sample.get('gen_answer', 'N/A')}")
        print(f"   Response preview: {sample.get('generated_response', '')[:150]}...")
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'error_count': len(errors),
        'no_answer_extracted': no_answer_extracted if errors else 0,
        'wrong_answer': wrong_answer if errors else 0,
    }


def compare_models(result_files):
    """Compare multiple model evaluations"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    all_stats = {}
    for result_file in result_files:
        model_name = Path(result_file).stem.replace('evaluation_results', 'model').replace('eval_', '')
        results = load_results(result_file)
        
        correct = sum(1 for r in results if r.get('correct', False))
        total = len(results)
        accuracy = correct / total * 100 if total > 0 else 0
        
        all_stats[model_name] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }
    
    # Print comparison table
    print(f"\n{'Model':<30} {'Accuracy':<15} {'Correct/Total'}")
    print("-" * 80)
    
    for model_name in sorted(all_stats.keys(), key=lambda x: all_stats[x]['accuracy'], reverse=True):
        stats = all_stats[model_name]
        print(f"{model_name:<30} {stats['accuracy']:>6.2f}%        {stats['correct']:>4}/{stats['total']:<4}")
    
    # Find best model
    best_model = max(all_stats.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best model: {best_model[0]} ({best_model[1]['accuracy']:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze Step-JEPA evaluation results")
    parser.add_argument("result_files", nargs='+', help="Evaluation result files (.jsonl)")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    
    args = parser.parse_args()
    
    if len(args.result_files) == 1 and not args.compare:
        # Analyze single file
        results = load_results(args.result_files[0])
        stats = analyze_results(results)
    else:
        # Compare multiple files
        compare_models(args.result_files)
        
        # Also analyze each individually
        for result_file in args.result_files:
            print(f"\n\n{'='*80}")
            print(f"Detailed analysis: {result_file}")
            print('='*80)
            results = load_results(result_file)
            analyze_results(results)


if __name__ == "__main__":
    main()

