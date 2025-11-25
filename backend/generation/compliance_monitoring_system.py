"""
Unified Compliance Monitoring System
Compares company database with AIID and Standards databases, then generates
contextualized compliance reports using AI analysis.

This script performs:
1. Vector similarity comparison between databases
2. Generation of compliance assessment reports using AI prompts
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# FAISS Database Functions
# ============================================================================

def load_faiss_database(db_dir: str, index_name: str) -> Tuple[faiss.Index, List[Dict], Dict]:
    """
    Load a FAISS database (index, metadata, and summary).
    
    Args:
        db_dir: Directory containing the FAISS database files
        index_name: Base name of the index files
        
    Returns:
        Tuple of (FAISS index, metadata list, summary dict)
    """
    db_path = Path(db_dir)
    
    # Load index
    index_file = db_path / f"{index_name}.index"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    index = faiss.read_index(str(index_file))
    
    # Load metadata
    metadata_file = db_path / f"{index_name}_metadata.pkl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load summary
    summary_file = db_path / f"{index_name}_summary.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    
    return index, metadata, summary


def load_embeddings_from_json_files(source_files: List[str], base_dir: str) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load embeddings from original JSON files.
    
    Args:
        source_files: List of JSON file names
        base_dir: Base directory to search for files
        
    Returns:
        Tuple of (embeddings array, metadata list)
    """
    all_embeddings = []
    all_metadata = []
    
    # Search for JSON files in processed/vector directories
    search_dirs = [
        Path(base_dir) / "data" / "processed" / "vector" / "company",
        Path(base_dir) / "data" / "processed" / "vector" / "aiid",
        Path(base_dir) / "data" / "processed" / "vector" / "standards"
    ]
    
    for source_file in source_files:
        found = False
        for search_dir in search_dirs:
            json_path = search_dir / source_file
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chunks = data.get('chunks', [])
                for chunk in chunks:
                    embedding = chunk.get('embedding')
                    if embedding:
                        all_embeddings.append(embedding)
                        chunk_metadata = {
                            'chunk_id': chunk.get('chunk_id'),
                            'text': chunk.get('text'),
                            'source_file': data.get('metadata', {}).get('source_file'),
                            'source_name': data.get('metadata', {}).get('source_name'),
                            'embedding_file': source_file
                        }
                        all_metadata.append(chunk_metadata)
                found = True
                break
        
        if not found:
            print(f"Warning: Could not find source file: {source_file}")
    
    if not all_embeddings:
        return None, None
    
    embeddings_array = np.array(all_embeddings, dtype='float32')
    return embeddings_array, all_metadata


def compare_databases_with_embeddings(
    query_db_dir: str,
    query_index_name: str,
    target_db_dir: str,
    target_index_name: str,
    base_dir: str,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Compare two databases by loading embeddings from JSON files and searching.
    
    Args:
        query_db_dir: Directory of query database
        query_index_name: Name of query index
        target_db_dir: Directory of target database
        target_index_name: Name of target index
        base_dir: Base directory for finding JSON files
        top_k: Number of top matches per query
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Comparison results dictionary
    """
    # Load databases
    query_index, query_metadata, query_summary = load_faiss_database(query_db_dir, query_index_name)
    target_index, target_metadata, target_summary = load_faiss_database(target_db_dir, target_index_name)
    
    # Load embeddings from JSON files
    query_source_files = query_summary.get('source_files', [])
    target_source_files = target_summary.get('source_files', [])
    
    print(f"Loading query embeddings from {len(query_source_files)} source file(s)...")
    query_embeddings, query_meta = load_embeddings_from_json_files(query_source_files, base_dir)
    
    print(f"Loading target embeddings from {len(target_source_files)} source file(s)...")
    target_embeddings, target_meta = load_embeddings_from_json_files(target_source_files, base_dir)
    
    if query_embeddings is None or target_embeddings is None:
        raise ValueError("Could not load embeddings from source files")
    
    # Normalize for cosine similarity (if using IndexFlatIP)
    faiss.normalize_L2(query_embeddings)
    faiss.normalize_L2(target_embeddings)
    
    # Search each query vector in target database
    print(f"Searching {len(query_embeddings)} query vectors in target database...")
    similarities, indices = target_index.search(query_embeddings, top_k)
    
    # Diagnostic: Show similarity score distribution
    all_similarities = similarities.flatten()
    print(f"\nSimilarity Score Distribution:")
    print(f"  Max similarity found: {float(np.max(all_similarities)):.4f}")
    print(f"  Min similarity found: {float(np.min(all_similarities)):.4f}")
    print(f"  Mean similarity: {float(np.mean(all_similarities)):.4f}")
    print(f"  Median similarity: {float(np.median(all_similarities)):.4f}")
    print(f"  Threshold used: {similarity_threshold}")
    print(f"  Scores above threshold: {len(all_similarities[all_similarities >= similarity_threshold])} / {len(all_similarities)}")
    
    # Process results
    all_matches = []
    similarity_scores = []
    
    for i, (query_vec_idx, query_meta_item) in enumerate(zip(range(len(query_embeddings)), query_meta)):
        query_similarities = similarities[i]
        query_indices = indices[i]
        
        matches_for_query = []
        for sim_score, target_idx in zip(query_similarities, query_indices):
            if sim_score >= similarity_threshold and target_idx < len(target_meta):
                match = {
                    'query_index': i,
                    'query_chunk_id': query_meta_item.get('chunk_id'),
                    'query_text': query_meta_item.get('text', '')[:200] + '...' if len(query_meta_item.get('text', '')) > 200 else query_meta_item.get('text', ''),
                    'query_source': query_meta_item.get('source_name'),
                    'target_index': int(target_idx),
                    'target_chunk_id': target_meta[int(target_idx)].get('chunk_id'),
                    'target_text': target_meta[int(target_idx)].get('text', '')[:200] + '...' if len(target_meta[int(target_idx)].get('text', '')) > 200 else target_meta[int(target_idx)].get('text', ''),
                    'target_source': target_meta[int(target_idx)].get('source_name'),
                    'similarity': float(sim_score)
                }
                matches_for_query.append(match)
                similarity_scores.append(float(sim_score))
        
        if matches_for_query:
            all_matches.append({
                'query_index': i,
                'query_source': query_meta_item.get('source_name'),
                'matches': matches_for_query
            })
    
    # Calculate statistics
    stats = {
        'total_queries': len(query_embeddings),
        'queries_with_matches': len(all_matches),
        'total_matches': len(similarity_scores),
        'avg_similarity': float(np.mean(similarity_scores)) if similarity_scores else 0.0,
        'max_similarity': float(np.max(similarity_scores)) if similarity_scores else 0.0,
        'min_similarity': float(np.min(similarity_scores)) if similarity_scores else 0.0,
        'median_similarity': float(np.median(similarity_scores)) if similarity_scores else 0.0,
        'matches_above_threshold': len([s for s in similarity_scores if s >= similarity_threshold])
    }
    
    # Group by source
    source_stats = {}
    for match_group in all_matches:
        source = match_group['query_source']
        if source not in source_stats:
            source_stats[source] = {'count': 0, 'matches': 0}
        source_stats[source]['count'] += 1
        source_stats[source]['matches'] += len(match_group['matches'])
    
    return {
        'query_database': query_summary,
        'target_database': target_summary,
        'statistics': stats,
        'source_statistics': source_stats,
        'top_matches': sorted(all_matches, key=lambda x: max(m['similarity'] for m in x['matches']), reverse=True)[:20],
        'all_matches': all_matches[:100]  # Limit to first 100 for report size
    }


def save_comparison_report(comparison_results: Dict[str, Any], output_file: str, comparison_name: str):
    """
    Generate and save a detailed comparison report.
    
    Args:
        comparison_results: Results from compare_databases_with_embeddings
        output_file: Path to save the report
        comparison_name: Name of the comparison (e.g., "Company vs AIID")
    """
    report = {
        'comparison_name': comparison_name,
        'generated_at': datetime.now().isoformat(),
        'query_database': comparison_results['query_database'],
        'target_database': comparison_results['target_database'],
        'statistics': comparison_results['statistics'],
        'source_statistics': comparison_results['source_statistics'],
        'top_matches': comparison_results['top_matches']
    }
    
    # Save JSON report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Also create a human-readable text report
    txt_report_path = output_path.with_suffix('.txt')
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"COMPARISON REPORT: {comparison_name}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Generated at: {report['generated_at']}\n\n")
        
        f.write(f"QUERY DATABASE:\n")
        f.write(f"  Name: {comparison_results['query_database'].get('source_files', ['Unknown'])}\n")
        f.write(f"  Vectors: {comparison_results['query_database'].get('num_vectors', 0)}\n")
        f.write(f"  Dimension: {comparison_results['query_database'].get('dimension', 0)}\n\n")
        
        f.write(f"TARGET DATABASE:\n")
        f.write(f"  Name: {comparison_results['target_database'].get('source_files', ['Unknown'])}\n")
        f.write(f"  Vectors: {comparison_results['target_database'].get('num_vectors', 0)}\n")
        f.write(f"  Dimension: {comparison_results['target_database'].get('dimension', 0)}\n\n")
        
        stats = comparison_results['statistics']
        f.write(f"STATISTICS:\n")
        f.write(f"  Total queries: {stats['total_queries']}\n")
        f.write(f"  Queries with matches: {stats['queries_with_matches']}\n")
        f.write(f"  Total matches: {stats['total_matches']}\n")
        f.write(f"  Average similarity: {stats['avg_similarity']:.4f}\n")
        f.write(f"  Max similarity: {stats['max_similarity']:.4f}\n")
        f.write(f"  Min similarity: {stats['min_similarity']:.4f}\n")
        f.write(f"  Median similarity: {stats['median_similarity']:.4f}\n")
        f.write(f"  Matches above threshold (0.7): {stats['matches_above_threshold']}\n\n")
        
        f.write(f"SOURCE STATISTICS:\n")
        for source, source_stat in comparison_results['source_statistics'].items():
            f.write(f"  {source}:\n")
            f.write(f"    Queries: {source_stat['count']}\n")
            f.write(f"    Matches: {source_stat['matches']}\n")
        f.write("\n")
        
        f.write(f"TOP MATCHES:\n")
        f.write(f"{'-'*80}\n")
        for i, match_group in enumerate(comparison_results['top_matches'][:10], 1):
            f.write(f"\nMatch Group {i}:\n")
            f.write(f"  Query Source: {match_group['query_source']}\n")
            f.write(f"  Query Text: {match_group['matches'][0]['query_text']}\n")
            f.write(f"  Matches found: {len(match_group['matches'])}\n")
            for j, match in enumerate(match_group['matches'][:3], 1):
                f.write(f"    Match {j} (similarity: {match['similarity']:.4f}):\n")
                f.write(f"      Target Source: {match['target_source']}\n")
                f.write(f"      Target Text: {match['target_text']}\n")
            f.write("\n")
    
    print(f"Comparison report saved to {output_path}")
    print(f"Text report saved to {txt_report_path}")
    
    return report


# ============================================================================
# Compliance Report Generation Functions
# ============================================================================

def get_openai_client():
    """Get OpenAI client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    try:
        return OpenAI(api_key=api_key)
    except TypeError:
        os.environ["OPENAI_API_KEY"] = api_key
        return OpenAI()


def load_prompt(prompt_file: str) -> str:
    """
    Load a prompt template from file.
    
    Args:
        prompt_file: Path to prompt file
        
    Returns:
        Prompt text as string
    """
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_report_for_prompt(report_data: Dict[str, Any]) -> str:
    """
    Format comparison report data into a readable format for the prompt.
    
    Args:
        report_data: Comparison report dictionary
        
    Returns:
        Formatted report text
    """
    formatted = []
    
    formatted.append("="*80)
    formatted.append("COMPARISON REPORT DATA")
    formatted.append("="*80)
    formatted.append("")
    
    # Basic information
    formatted.append(f"Comparison: {report_data.get('comparison_name', 'Unknown')}")
    formatted.append(f"Generated: {report_data.get('generated_at', 'Unknown')}")
    formatted.append("")
    
    # Query database info
    query_db = report_data.get('query_database', {})
    formatted.append("COMPANY DATABASE:")
    formatted.append(f"  Source Files: {', '.join(query_db.get('source_files', []))}")
    formatted.append(f"  Total Vectors: {query_db.get('num_vectors', 0)}")
    formatted.append(f"  Dimension: {query_db.get('dimension', 0)}")
    formatted.append("")
    
    # Target database info
    target_db = report_data.get('target_database', {})
    formatted.append("TARGET DATABASE:")
    formatted.append(f"  Source Files: {', '.join(target_db.get('source_files', []))}")
    formatted.append(f"  Total Vectors: {target_db.get('num_vectors', 0)}")
    formatted.append(f"  Dimension: {target_db.get('dimension', 0)}")
    formatted.append("")
    
    # Statistics
    stats = report_data.get('statistics', {})
    formatted.append("STATISTICS:")
    formatted.append(f"  Total Queries: {stats.get('total_queries', 0)}")
    formatted.append(f"  Queries with Matches: {stats.get('queries_with_matches', 0)}")
    formatted.append(f"  Total Matches: {stats.get('total_matches', 0)}")
    formatted.append(f"  Average Similarity: {stats.get('avg_similarity', 0.0):.4f}")
    formatted.append(f"  Max Similarity: {stats.get('max_similarity', 0.0):.4f}")
    formatted.append(f"  Min Similarity: {stats.get('min_similarity', 0.0):.4f}")
    formatted.append(f"  Median Similarity: {stats.get('median_similarity', 0.0):.4f}")
    formatted.append(f"  Matches Above Threshold (0.7): {stats.get('matches_above_threshold', 0)}")
    formatted.append("")
    
    # Source statistics
    source_stats = report_data.get('source_statistics', {})
    if source_stats:
        formatted.append("SOURCE STATISTICS:")
        for source, stat in source_stats.items():
            formatted.append(f"  {source}:")
            formatted.append(f"    Queries: {stat.get('count', 0)}")
            formatted.append(f"    Matches: {stat.get('matches', 0)}")
        formatted.append("")
    
    # Top matches
    top_matches = report_data.get('top_matches', [])
    if top_matches:
        formatted.append("TOP MATCHES:")
        formatted.append("-"*80)
        for i, match_group in enumerate(top_matches[:20], 1):
            formatted.append(f"\nMatch Group {i}:")
            formatted.append(f"  Query Source: {match_group.get('query_source', 'Unknown')}")
            
            matches = match_group.get('matches', [])
            if matches:
                first_match = matches[0]
                formatted.append(f"  Query Text: {first_match.get('query_text', 'N/A')}")
                formatted.append(f"  Number of Matches: {len(matches)}")
                
                for j, match in enumerate(matches[:3], 1):
                    formatted.append(f"\n  Match {j}:")
                    formatted.append(f"    Similarity Score: {match.get('similarity', 0.0):.4f}")
                    formatted.append(f"    Target Source: {match.get('target_source', 'Unknown')}")
                    formatted.append(f"    Target Text: {match.get('target_text', 'N/A')}")
            formatted.append("")
    
    return "\n".join(formatted)


def generate_compliance_analysis(
    report_data: Dict[str, Any],
    base_prompt: str,
    model: str = "gpt-4",
    temperature: float = 0.3
) -> str:
    """
    Generate compliance analysis using OpenAI API.
    
    Args:
        report_data: Comparison report data
        base_prompt: Base prompt template
        model: OpenAI model to use
        temperature: Temperature for generation
        
    Returns:
        Generated compliance analysis text
    """
    client = get_openai_client()
    
    # Format report data
    report_text = format_report_for_prompt(report_data)
    
    # Combine prompt with report data
    full_prompt = f"""{base_prompt}

## Report Data

{report_text}

## Analysis Request

Please analyze this comparison report and provide a comprehensive compliance assessment following the framework outlined in the prompt above. Be specific, actionable, and evidence-based in your analysis.
"""
    
    print("Generating compliance analysis...")
    print(f"Using model: {model}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert compliance analyst and risk assessment specialist."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            temperature=temperature,
            max_tokens=4000
        )
        
        analysis = response.choices[0].message.content
        return analysis
    
    except Exception as e:
        raise Exception(f"Error generating analysis: {str(e)}")


def save_compliance_report(
    analysis: str,
    output_file: str,
    report_type: str,
    comparison_name: str
):
    """
    Save generated compliance report.
    
    Args:
        analysis: Generated analysis text
        output_file: Path to save report
        report_type: Type of report (standards/aiid)
        comparison_name: Name of the comparison
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create report with metadata
    report = {
        "report_type": report_type,
        "comparison_name": comparison_name,
        "generated_at": datetime.now().isoformat(),
        "analysis": analysis
    }
    
    # Save JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"COMPLIANCE REPORT: {comparison_name}\n")
        f.write(f"Report Type: {report_type.upper()}\n")
        f.write(f"Generated: {report['generated_at']}\n")
        f.write("="*80 + "\n\n")
        f.write(analysis)
    
    print(f"Compliance report saved to {output_path}")
    print(f"JSON report saved to {json_path}")


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    """Main function: Compare databases and generate compliance reports."""
    BASE_DIR = r"C:\Users\VectoreCore\OneDrive - The University of Alabama\Documents\Coding\comp_rag"
    BASE_PATH = Path(BASE_DIR)
    
    PROMPTS_DIR = BASE_PATH / "backend" / "generation" / "prompts"
    REPORTS_DIR = BASE_PATH / "backend" / "generation" / "reports"
    
    # Paths to databases
    COMPANY_DB_DIR = BASE_PATH / "backend" / "building_database" / "faiss" / "company"
    AIID_DB_DIR = BASE_PATH / "backend" / "building_database" / "faiss" / "aiid"
    STANDARDS_DB_DIR = BASE_PATH / "backend" / "building_database" / "faiss" / "standards"
    
    # ========================================================================
    # STEP 1: Compare Databases
    # ========================================================================
    
    print("="*80)
    print("STEP 1: COMPARING DATABASES")
    print("="*80)
    
    comparison_results = {}
    
    # Comparison 1: Company vs AIID
    print("\n" + "-"*80)
    print("COMPARISON 1: Company vs AIID")
    print("-"*80)
    try:
        results_company_aiid = compare_databases_with_embeddings(
            str(COMPANY_DB_DIR),
            "company_faiss_index",
            str(AIID_DB_DIR),
            "aiid_faiss_index",
            BASE_DIR,
            top_k=5,
            similarity_threshold=0.5  # Lowered from 0.7 to find more semantic matches
        )
        comparison_report_aiid = save_comparison_report(
            results_company_aiid,
            str(REPORTS_DIR / "company_vs_aiid_comparison.json"),
            "Company vs AIID"
        )
        comparison_results['aiid'] = comparison_report_aiid
    except Exception as e:
        print(f"Error comparing Company vs AIID: {str(e)}")
        import traceback
        traceback.print_exc()
        comparison_results['aiid'] = None
    
    # Comparison 2: Company vs Standards
    print("\n" + "-"*80)
    print("COMPARISON 2: Company vs Standards")
    print("-"*80)
    try:
        results_company_standards = compare_databases_with_embeddings(
            str(COMPANY_DB_DIR),
            "company_faiss_index",
            str(STANDARDS_DB_DIR),
            "standards_faiss_index",
            BASE_DIR,
            top_k=5,
            similarity_threshold=0.5  # Lowered from 0.7 to find more semantic matches
        )
        comparison_report_standards = save_comparison_report(
            results_company_standards,
            str(REPORTS_DIR / "company_vs_standards_comparison.json"),
            "Company vs Standards"
        )
        comparison_results['standards'] = comparison_report_standards
    except Exception as e:
        print(f"Error comparing Company vs Standards: {str(e)}")
        import traceback
        traceback.print_exc()
        comparison_results['standards'] = None
    
    # ========================================================================
    # STEP 2: Generate Compliance Reports
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: GENERATING COMPLIANCE REPORTS")
    print("="*80)
    
    # Load prompts
    try:
        base_prompt = load_prompt(str(PROMPTS_DIR / "compliance_monitoring_prompt.txt"))
        standards_prompt = load_prompt(str(PROMPTS_DIR / "standards_compliance_prompt.txt"))
        aiid_prompt = load_prompt(str(PROMPTS_DIR / "aiid_incident_analysis_prompt.txt"))
    except Exception as e:
        print(f"Error loading prompts: {str(e)}")
        print("Skipping compliance report generation.")
        return
    
    # Generate Compliance Report: Company vs Standards
    if comparison_results.get('standards'):
        print("\n" + "-"*80)
        print("GENERATING COMPLIANCE REPORT: Company vs Standards")
        print("-"*80)
        try:
            combined_standards_prompt = f"""{base_prompt}

{standards_prompt}
"""
            analysis = generate_compliance_analysis(
                comparison_results['standards'],
                combined_standards_prompt,
                model="gpt-4",
                temperature=0.3
            )
            save_compliance_report(
                analysis,
                str(REPORTS_DIR / "company_vs_standards_compliance_report.txt"),
                "standards",
                "Company vs Standards Compliance Assessment"
            )
        except Exception as e:
            print(f"Error generating standards compliance report: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Generate Compliance Report: Company vs AIID
    if comparison_results.get('aiid'):
        print("\n" + "-"*80)
        print("GENERATING COMPLIANCE REPORT: Company vs AIID")
        print("-"*80)
        try:
            combined_aiid_prompt = f"""{base_prompt}

{aiid_prompt}
"""
            analysis = generate_compliance_analysis(
                comparison_results['aiid'],
                combined_aiid_prompt,
                model="gpt-4",
                temperature=0.3
            )
            save_compliance_report(
                analysis,
                str(REPORTS_DIR / "company_vs_aiid_compliance_report.txt"),
                "aiid",
                "Company vs AIID Risk Assessment"
            )
        except Exception as e:
            print(f"Error generating AIID compliance report: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("COMPLIANCE MONITORING SYSTEM COMPLETE!")
    print("="*80)
    print("\nGenerated Reports:")
    print(f"  - Comparison Reports: {REPORTS_DIR}")
    print(f"  - Compliance Reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()

