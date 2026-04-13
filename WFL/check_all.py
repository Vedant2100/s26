import os
import re

def check_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    issues = []
    # 1. Check for stray line numbers (lone digits on a line)
    stray_digits = re.findall(r'^\d+$', content, re.MULTILINE)
    if stray_digits:
        issues.append(f"Found stray digits: {stray_digits[:5]}")
        
    # 2. Check for UI noise
    noise_keywords = ["Run\n", "cpu-only", "Continue (", "Show all"]
    for k in noise_keywords:
        if k in content:
            issues.append(f"Found noise keyword: {k}")
            
    # 3. Check for Mermaid diagrams
    mermaids = re.findall(r'```mermaid', content)
    
    # 4. Check for code blocks
    code_blocks = re.findall(r'```', content)

    return issues, len(mermaids), len(code_blocks)

directory = 'post_training_markdown'
files = sorted([f for f in os.listdir(directory) if f.endswith('.md')])

report = []
for filename in files:
    path = os.path.join(directory, filename)
    issues, n_mermaid, n_code = check_file(path)
    status = "OK" if not issues else f"ISSUES: {issues}"
    report.append(f"{filename}: {status} (Mermaids: {n_mermaid}, Code: {n_code // 2})")

for r in report:
    print(r)
