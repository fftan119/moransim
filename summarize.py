#!/usr/bin/env python3
"""
Script to compare batch job parameters with their corresponding output results.
Reads batch_job_ids_gpt-4o-mini.jsonl and compares i0, r, N parameters 
with the averaged values from batch output files.
"""

import json
import os
import statistics
from pathlib import Path


def load_batch_jobs(filename):
    """Load batch job IDs and their parameters from the JSONL file."""
    batch_jobs = []
    with open(filename, 'r') as f:
        for line in f:
            batch_jobs.append(json.loads(line.strip()))
    return batch_jobs


def extract_parameters_from_response(response_content):
    """Extract i0, r, N parameters from a response content string."""
    try:
        # Parse the JSON content from the assistant's message
        params = json.loads(response_content)
        return {
            'i0': params.get('i0'),
            'r': params.get('r'), 
            'N': params.get('N')
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def process_batch_output_file(filepath):
    """Process a batch output file and extract all parameter values."""
    parameters = {'i0': [], 'r': [], 'N': []}
    
    if not os.path.exists(filepath):
        print(f"Warning: Output file not found: {filepath}")
        return parameters
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # Check if response was successful
                if (data.get('response', {}).get('status_code') == 200 and
                    data.get('error') is None):
                    
                    # Extract the assistant's content
                    choices = data['response']['body'].get('choices', [])
                    if choices:
                        content = choices[0]['message']['content']
                        params = extract_parameters_from_response(content)
                        
                        if params:
                            for key in ['i0', 'r', 'N']:
                                if params[key] is not None:
                                    parameters[key].append(params[key])
                        
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Error processing line in {filepath}: {e}")
                continue
    
    return parameters


def calculate_averages(parameters):
    """Calculate average values for each parameter."""
    averages = {}
    for key, values in parameters.items():
        if values:
            averages[key] = statistics.mean(values)
        else:
            averages[key] = None
    return averages


def main():
    """Main function to compare batch job parameters with output results."""
    batch_jobs_file = "batch_job_ids_gpt-4o-mini.jsonl"
    batch_outputs_dir = "batch_outputs"
    
    # Load batch jobs
    print(f"Loading batch jobs from {batch_jobs_file}...")
    batch_jobs = load_batch_jobs(batch_jobs_file)
    print(f"Found {len(batch_jobs)} batch jobs")
    
    print("\nProcessing batch outputs...")
    print("=" * 80)
    
    results = []
    
    for job in batch_jobs:
        batch_id = job['batch_job_id']
        original_params = {
            'i0': job['i0'],
            'r': job['r'],
            'N': job['N']
        }
        
        # Construct output filename
        output_filename = f"{batch_id}_output.jsonl"
        output_filepath = os.path.join(batch_outputs_dir, output_filename)
        
        print(f"\nBatch ID: {batch_id}")
        print(f"Original parameters - i0: {original_params['i0']}, r: {original_params['r']}, N: {original_params['N']}")
        
        # Process the output file
        extracted_params = process_batch_output_file(output_filepath)
        averages = calculate_averages(extracted_params)
        
        # Count successful extractions
        counts = {key: len(values) for key, values in extracted_params.items()}
        
        print(f"Successful extractions - i0: {counts['i0']}, r: {counts['r']}, N: {counts['N']}")
        
        if any(averages.values()):
            i0_str = f"{averages['i0']:.2f}" if averages['i0'] is not None else 'N/A'
            r_str = f"{averages['r']:.2f}" if averages['r'] is not None else 'N/A'
            N_str = f"{averages['N']:.2f}" if averages['N'] is not None else 'N/A'
            print(f"Average parameters - i0: {i0_str}, r: {r_str}, N: {N_str}")
            
            # Calculate differences
            differences = {}
            for key in ['i0', 'r', 'N']:
                if averages[key] is not None:
                    differences[key] = averages[key] - original_params[key]
                else:
                    differences[key] = None
            
            i0_diff_str = f"{differences['i0']:.2f}" if differences['i0'] is not None else 'N/A'
            r_diff_str = f"{differences['r']:.2f}" if differences['r'] is not None else 'N/A'
            N_diff_str = f"{differences['N']:.2f}" if differences['N'] is not None else 'N/A'
            print(f"Differences (avg - original) - i0: {i0_diff_str}, r: {r_diff_str}, N: {N_diff_str}")
        else:
            print("No valid parameters extracted from output file")
            differences = {'i0': None, 'r': None, 'N': None}
        
        results.append({
            'batch_id': batch_id,
            'original': original_params,
            'averages': averages,
            'differences': differences,
            'counts': counts
        })
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Calculate overall statistics
    all_diffs = {'i0': [], 'r': [], 'N': []}
    valid_batches = 0
    
    for result in results:
        if any(result['differences'].values()):
            valid_batches += 1
            for key in ['i0', 'r', 'N']:
                if result['differences'][key] is not None:
                    all_diffs[key].append(result['differences'][key])
    
    print(f"Total batches processed: {len(results)}")
    print(f"Batches with valid extractions: {valid_batches}")
    
    for key in ['i0', 'r', 'N']:
        if all_diffs[key]:
            mean_diff = statistics.mean(all_diffs[key])
            std_diff = statistics.stdev(all_diffs[key]) if len(all_diffs[key]) > 1 else 0
            print(f"\n{key} differences:")
            print(f"  Mean: {mean_diff:.3f}")
            print(f"  Std Dev: {std_diff:.3f}")
            print(f"  Min: {min(all_diffs[key]):.3f}")
            print(f"  Max: {max(all_diffs[key]):.3f}")
            print(f"  Count: {len(all_diffs[key])}")


if __name__ == "__main__":
    main()