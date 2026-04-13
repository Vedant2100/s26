import asyncio
from playwright.async_api import async_playwright
import urllib.request
import xml.etree.ElementTree as ET
import os
import re
import markdownify

OUTPUT_DIR = "post_training_markdown"
FINAL_MD = "FrontierLab_Post_Training_Full.md"
PREFIX = "https://workatafrontierlab.com/lessons/post-training/"

async def discover_and_save():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Fetch XML sitemap
    print(f"Fetching sitemap from https://workatafrontierlab.com/sitemap.xml to discover lessons...", flush=True)
    try:
        req = urllib.request.Request("https://workatafrontierlab.com/sitemap.xml", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
    except Exception as e:
        print(f"Failed to fetch sitemap: {e}")
        return

    root = ET.fromstring(xml_data)
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    
    # Discover all links in the sitemap starting with PREFIX
    # Use dict.fromkeys to remove duplicates while strictly preserving the chronological XML order!
    extracted_links = [
        loc.text for loc in root.findall('.//ns:loc', namespace) 
        if loc.text and loc.text.startswith(PREFIX)
    ]
    lesson_links = list(dict.fromkeys(extracted_links))

    if not lesson_links:
        print("Discovery failed. Please ensure the URL is correct.", flush=True)
        return

    print("Launching browser...", flush=True)
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="./playwright_profile",
            headless=False,
            viewport={'width': 1280, 'height': 800}
        )
        page = context.pages[0] if context.pages else await context.new_page()

        print("\n*** ACTION REQUIRED ***", flush=True)
        print("Opening the first lesson link to check authentication status.", flush=True)
        
        await page.goto(lesson_links[0], wait_until="networkidle")
        
        print("If you are faced with a Login Screen in the browser window that popped up, PLEASE LOG IN MULTIPLE NOW.", flush=True)
        try:
            await asyncio.to_thread(input, "\n--> PRESS ENTER HERE IN THE TERMINAL ONCE YOU ARE FULLY LOGGED IN: ")
        except EOFError:
            print("\n[Notice] Terminal input unavailable. You have 60 seconds to log in...", flush=True)
            for remaining in range(60, 0, -10):
                print(f"Waiting for login... {remaining} seconds remaining.", flush=True)
                await asyncio.sleep(10)

        print(f"Found {len(lesson_links)} lessons. Starting Markdown extraction...", flush=True)

        for i, url in enumerate(lesson_links):
            print(f"Processing ({i+1}/{len(lesson_links)}): {url}", flush=True)
            try:
                await page.goto(url, wait_until="load", timeout=60000)
                await asyncio.sleep(2) 

                # Detect if we were unexpectedly logged out or hit a paywall/login modal mid-scrape
                page_text = await page.content()
                if "Sign in to continue" in page_text or "login" in page.url.lower():
                    print(f"\n[!] WARNING: The site asked for authentication on {url}!", flush=True)
                    print("   Please use the open browser window to log in. The script will automatically wait...", flush=True)
                    
                    while True:
                        current_text = await page.content()
                        if "Sign in to continue" not in current_text and "login" not in page.url.lower():
                            break
                        await asyncio.sleep(2)
                        
                    print("   Login cleared! Resuming...", flush=True)
                    if page.url != url:
                        await page.goto(url, wait_until="load")
                        await asyncio.sleep(2)

                # Clean up UI DOM directly to remove fluff (headers, navs, footers)
                await page.evaluate("""() => { 
                    // 1. Target & Clean Code Editors (Monaco)
                    document.querySelectorAll('.monaco-editor').forEach(editor => {
                        try {
                            // Extract lines from the 'view-lines' container
                            const lines = Array.from(editor.querySelectorAll('.view-line'))
                                .map(l => l.innerText || l.textContent || '');
                            const cleanCode = lines.join('\\n');
                            
                            // Find the filename if available in the nearby header
                            let filename = 'code';
                            const header = editor.parentElement?.querySelector('[class*="header"]');
                            if (header) {
                                const title = header.innerText || header.textContent;
                                if (title && title.includes('.')) filename = title.split('\\n')[0].trim();
                            }

                            // Replace the messy editor with a clean <pre> block
                            const pre = document.createElement('pre');
                            const code = document.createElement('code');
                            // Add a hidden hint for markdownify if possible, or just set it
                            code.textContent = cleanCode;
                            pre.appendChild(code);
                            
                            // Replace the high-level container of the editor
                            let container = editor;
                            while (container.parentElement && 
                                   (container.parentElement.className.includes('Editor') || 
                                    container.parentElement.className.includes('monaco'))) {
                                container = container.parentElement;
                            }
                            container.replaceWith(pre);
                        } catch (e) {
                            console.error('Failed to clean editor:', e);
                        }
                    });

                    // 2. Preserve Mermaid Diagrams
                    document.querySelectorAll('.Mermaid').forEach(m => {
                        const source = m.getAttribute('data-mermaid') || m.innerText || m.textContent;
                        if (source) {
                            const pre = document.createElement('pre');
                            const code = document.createElement('code');
                            code.className = 'language-mermaid';
                            code.textContent = source.trim();
                            pre.appendChild(code);
                            m.replaceWith(pre);
                        }
                    });

                    // 3. Remove known UI Noise elements (including stray line numbers/buttons)
                    const noise = [
                        'nav', 'header', 'footer', 'aside', 'script', 'style', 'noscript', 'meta',
                        '[class*="Breadcrumbs"]', '[class*="RunButton"]', '[class*="cpuOnly"]',
                        '[class*="Pagination"]', '[class*="Sidebar"]', 'button',
                        '[class*="Continue"]', '[class*="ShowAll"]', '.line-numbers'
                    ];
                    noise.forEach(sel => { 
                        document.querySelectorAll(sel).forEach(el => el.remove()); 
                    }); 

                    // 4. Force expand all details
                    document.querySelectorAll('details').forEach(el => el.setAttribute('open', ''));
                    document.querySelectorAll('button[aria-expanded="false"]').forEach(el => el.click());
                }""")
                await asyncio.sleep(2)

                # Extract HTML
                html_body = await page.evaluate("""() => {
                    let core = document.querySelector('article') || document.querySelector('main') || document.body;
                    return core.innerHTML;
                }""")
                
                # REFINED MARKDOWN CONVERSION
                md_content = markdownify.markdownify(
                    html_body, 
                    heading_style="ATX", 
                    strip=['script', 'style', 'button']
                ).strip()

                # POST-PROCESSING CLEANUP: Strip UI artifacts that leaked through
                # Remove filenames headers noise (e.g. "behavior.pycpu-only")
                md_content = re.sub(r'\.pycpu-only', '.py', md_content)
                md_content = re.sub(r'Run\n', '', md_content)
                md_content = re.sub(r'cpu-only\n', '', md_content)
                
                # Remove stray lone numbers on lines (Pagination / Code artefacts)
                md_content = re.sub(r'^\s*\d+\s*$', '', md_content, flags=re.MULTILINE)
                
                # Collapse excessive newlines
                md_content = re.sub(r'\n{3,}', '\n\n', md_content)

                filename = os.path.join(OUTPUT_DIR, f'lesson_{i+1:02d}.md')
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                    
                print(f"Saved {filename}")
            except Exception as e:
                print(f"Error on {url}: {e}")

        await context.close()
        print("Scraping completed!", flush=True)

def merge_mds(directory, output_filename):
    print(f"Merging Markdown files into {output_filename}...")
    files = sorted([f for f in os.listdir(directory) if f.endswith('.md')])
    if not files:
        print("No markdown files found to merge.")
        return
        
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for f in files:
            filepath = os.path.join(directory, f)
            with open(filepath, 'r', encoding='utf-8') as infile:
                outfile.write(f"\n\n# --- Lesson Extracted from {f} ---\n\n")
                outfile.write(infile.read())
                
    print(f"Successfully generated final merged Document!")

if __name__ == "__main__":
    asyncio.run(discover_and_save())
    
    if os.path.exists(OUTPUT_DIR):
        merge_mds(OUTPUT_DIR, FINAL_MD)