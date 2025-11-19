#!/usr/bin/env python
import argparse
import json
import sys
import re
import html
from pathlib import Path

# ---------- Helpers ----------

def slugify(title: str) -> str:
    """
    Turn a subject into a filename-friendly slug.
    e.g. "The Big Bang Theory (TV series)" -> "The_Big_Bang_Theory_(TV_series)"
    """
    title = title.strip()
    title = title.replace(" ", "_")
    # keep letters, digits, _, -, ., (, )
    title = re.sub(r"[^\w\-\.\(\)]+", "_", title)
    return title


def load_articles(jsonl_path: str):
    articles = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Expect at least "subject" and "wikitext"
            if "subject" in obj and "wikitext" in obj:
                articles.append(obj)
    return articles


def build_slug_map(articles):
    """
    Map subject -> slug (and lowercase subject -> slug for more tolerant linking).
    """
    slug_map = {}
    for a in articles:
        subject = a["subject"]
        slug = slugify(subject)
        slug_map[subject] = slug
        slug_map[subject.lower()] = slug
    return slug_map


def strip_wiki_markup(wikitext: str) -> str:
    """
    Turn wikitext into plain text (for search snippets).
    Very rough, but good enough.
    """
    # Remove simple templates like {{Reflist}}
    wikitext = re.sub(r"\{\{[^\n]+}}", "", wikitext)

    # Replace links [[Target|Label]] or [[Target]] with just the label
    def repl_link(m):
        content = m.group(1)
        if "|" in content:
            _, label = content.split("|", 1)
        else:
            label = content
        return label

    wikitext = re.sub(r"\[\[(.+?)\]\]", repl_link, wikitext)

    # Remove bold/italic markup
    wikitext = wikitext.replace("'''''", "")
    wikitext = wikitext.replace("'''", "")
    wikitext = wikitext.replace("''", "")

    # Collapse whitespace
    wikitext = re.sub(r"\s+", " ", wikitext)
    return wikitext.strip()


def wikitext_to_html(wikitext: str, slug_map: dict) -> str:
    """
    Very simple wikitext -> HTML:
    - Headings: == H2 ==, === H3 ===, etc.
    - Bold/italic: '''bold''', ''italic''
    - Links: [[Target|Label]] -> <a href="Target.html">Label</a> (only if Target exists)
    - Auto-link http(s) URLs
    - Paragraph-ish formatting
    """

    # Remove simple one-line templates (e.g. {{Reflist}})
    wikitext = re.sub(r"\{\{[^\n]+}}", "", wikitext)

    def process_inline(text: str) -> str:
        # Internal wiki-style links [[Target|Label]] / [[Target]]
        def repl_link(m):
            content = m.group(1)
            if "|" in content:
                target, label = content.split("|", 1)
            else:
                target, label = content, content

            target_page = target.strip()
            label = label.strip()

            # Handle Category:Foo -> try article "Foo" if it exists
            lookup_keys = [target_page, target_page.lower()]
            if target_page.startswith("Category:"):
                cat_name = target_page.split(":", 1)[1].strip()
                lookup_keys.extend([cat_name, cat_name.lower()])

            href = None
            for key in lookup_keys:
                if key in slug_map:
                    href = slug_map[key] + ".html"
                    break

            # Only link to subjects that actually exist in our jsonl
            if href:
                return f'<a href="{href}">{label}</a>'
            else:
                # Show label as plain text if we don't have that article
                return label

        # [[...]] → internal links
        text = re.sub(r"\[\[(.+?)\]\]", repl_link, text)

        # Bold+italic: '''''text'''''
        text = re.sub(r"'''''(.+?)'''''", r"<b><i>\1</i></b>", text)
        # Bold: '''text'''
        text = re.sub(r"'''(.+?)'''", r"<b>\1</b>", text)
        # Italic: ''text''
        text = re.sub(r"''(.+?)''", r"<i>\1</i>", text)

        # Auto-link raw http:// or https:// URLs
        def link_url(m):
            url = m.group(0)
            return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>'

        # Stop at whitespace or "<" so we don't eat closing tags like </ref>
        text = re.sub(r"https?://[^\s<]+", link_url, text)

        return text

    lines = wikitext.splitlines()
    html_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            html_lines.append("")  # blank line (paragraph break)
            continue

        # Headings: == Heading ==
        m = re.match(r"^(=+)\s*(.+?)\s*\1$", stripped)
        if m:
            level = min(len(m.group(1)), 6)
            content = process_inline(m.group(2))
            html_lines.append(f"<h{level}>{content}</h{level}>")
        else:
            html_lines.append(process_inline(line))

    # Turn blocks of text into paragraphs, but leave headings as-is
    result = []
    buffer = []

    def flush_buffer():
        nonlocal buffer, result
        if not buffer:
            return
        paragraph = " ".join(buffer).strip()
        if not paragraph:
            buffer = []
            return
        # Don't wrap block-level tags
        if paragraph.startswith("<h"):
            result.append(paragraph)
        else:
            result.append(f"<p>{paragraph}</p>")
        buffer = []

    for line in html_lines:
        if not line:
            flush_buffer()
            continue
        if line.startswith("<h"):
            flush_buffer()
            result.append(line)
        else:
            buffer.append(line)
    flush_buffer()

    return "\n".join(result)


def write_style_css(out_dir: Path):
    css_path = out_dir / "style.css"
    if css_path.exists():
        return
    css = """
body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f6f6f6;
    color: #202122;
    margin: 0;
}
a {
    color: #0645ad;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
#wrapper {
    max-width: 960px;
    margin: 0 auto;
    padding: 1rem 1.5rem 3rem;
    background: #ffffff;
    box-shadow: 0 0 4px rgba(0,0,0,0.1);
}
header {
    border-bottom: 1px solid #a2a9b1;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
}
header h1 {
    font-size: 1.4rem;
    margin: 0;
}
header .tagline {
    font-size: 0.8rem;
    color: #54595d;
}
main h1 {
    font-size: 1.8rem;
    margin-top: 0.5rem;
}
main h2, main h3, main h4 {
    border-bottom: 1px solid #a2a9b1;
    padding-bottom: 0.2rem;
    margin-top: 1.5rem;
}
footer {
    border-top: 1px solid #a2a9b1;
    margin-top: 2rem;
    padding-top: 0.5rem;
    font-size: 0.8rem;
    color: #54595d;
}
.search-box {
    margin: 0.5rem 0;
}
.search-box input {
    padding: 0.3rem 0.4rem;
    width: 260px;
    max-width: 100%;
}
.result {
    padding: 0.4rem 0;
    border-bottom: 1px solid #eaecf0;
}
.result p {
    margin: 0.1rem 0 0;
    font-size: 0.85rem;
    color: #54595d;
}
.article-list {
    margin-top: 1rem;
    font-size: 0.9rem;
}
.article-list ul {
    columns: 3 200px;
    -webkit-columns: 3 200px;
    -moz-columns: 3 200px;
}
.article-list li {
    list-style: none;
    margin-bottom: 0.2rem;
}
    """.strip()
    css_path.write_text(css, encoding="utf-8")


def make_article_page(article, slug_map, out_dir: Path):
    subject = article["subject"]
    wikitext = article["wikitext"]
    slug = slug_map[subject]  # original-case key exists

    body_html = wikitext_to_html(wikitext, slug_map)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(subject)} - LLMPedia</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
<div id="wrapper">
  <header>
    <div>
      <h1><a href="index.html">LLMPedia</a></h1>
      <div class="tagline">A tiny wiki built from LLMs</div>
    </div>
    <div class="search-box">
      <form action="index.html">
        <input type="text" name="q" placeholder="Search…" />
      </form>
    </div>
  </header>
  <main>
    <h1>{html.escape(subject)}</h1>
    {body_html}
  </main>
  <footer>
    Generated from <code>articles.jsonl</code>. Subject: {html.escape(subject)}
  </footer>
</div>
</body>
</html>
"""
    (out_dir / f"{slug}.html").write_text(page_html, encoding="utf-8")


def make_index_page(search_index, articles, slug_map, out_dir: Path):
    # Embed search index directly into the page as JS data
    import json as _json

    search_json = _json.dumps(search_index, ensure_ascii=False)

    # List all articles
    items_html = []
    for a in sorted(articles, key=lambda x: x["subject"].lower()):
        subject = a["subject"]
        slug = slug_map[subject]
        items_html.append(f'<li><a href="{slug}.html">{html.escape(subject)}</a></li>')

    articles_list = "\n".join(items_html)

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MiniWiki index</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
<div id="wrapper">
  <header>
    <div>
      <h1>LLMPedia</h1>
      <div class="tagline">Search & browse your generated articles</div>
    </div>
    <div class="search-box">
      <input id="searchBox" type="text" placeholder="Search articles…" autocomplete="off" />
    </div>
  </header>

  <main>
    <section id="searchResults" style="display:none;">
      <h2>Search results</h2>
      <div id="results"></div>
    </section>

    <section class="article-list">
      <h2>All articles</h2>
      <ul>
        {articles_list}
      </ul>
    </section>
  </main>

  <footer>
    Generated from <code>articles.jsonl</code>.
  </footer>
</div>

<script>
  const SEARCH_INDEX = {search_json};

  const box = document.getElementById('searchBox');
  const resultsContainer = document.getElementById('results');
  const searchSection = document.getElementById('searchResults');

  function renderResults(query) {{
    const q = query.trim().toLowerCase();
    resultsContainer.innerHTML = "";
    if (!q) {{
      searchSection.style.display = "none";
      return;
    }}
    const matches = SEARCH_INDEX.filter(function(item) {{
      return item.title.toLowerCase().includes(q) ||
             item.snippet.toLowerCase().includes(q);
    }}).slice(0, 50);

    if (!matches.length) {{
      searchSection.style.display = "block";
      resultsContainer.innerHTML = "<p>No results.</p>";
      return;
    }}
    searchSection.style.display = "block";
    matches.forEach(function(item) {{
      var div = document.createElement('div');
      div.className = 'result';
      div.innerHTML =
        '<a href="' + item.url + '"><strong>' +
        item.title + '</strong></a>' +
        '<p>' + item.snippet + '</p>';
      resultsContainer.appendChild(div);
    }});
  }}

  box.addEventListener('input', function(e) {{
    renderResults(e.target.value);
  }});

  var params = new URLSearchParams(window.location.search);
  if (params.has('q')) {{
    var q = params.get('q');
    box.value = q;
    renderResults(q);
  }}
</script>

</body>
</html>
"""
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")


def build_site(jsonl_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading articles from {jsonl_path}...")
    articles = load_articles(jsonl_path)
    print(f"Loaded {len(articles)} articles.")

    slug_map = build_slug_map(articles)

    write_style_css(out_dir)

    # Build search index
    search_index = []
    for a in articles:
        subject = a["subject"]
        slug = slug_map[subject]
        plain = strip_wiki_markup(a["wikitext"])
        snippet = plain[:250] + ("…" if len(plain) > 250 else "")
        search_index.append({
            "title": subject,
            "url": f"{slug}.html",
            "snippet": snippet
        })

    # Write each article page
    for a in articles:
        make_article_page(a, slug_map, out_dir)

    # Index / search page
    make_index_page(search_index, articles, slug_map, out_dir)

    print(f"Done. Open {out_dir/'index.html'} in your browser.")


def main():
    parser = argparse.ArgumentParser(
        description="Build a small HTML wiki from LLMPedia articles.jsonl"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to LLMPedia run directory (must contain articles.jsonl)",
    )
    parser.add_argument(
        "--articles-file",
        default="articles.jsonl",
        help="Name of the articles JSONL file inside run-dir (default: articles.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        default="site",
        help="Name of output folder to create inside run-dir (default: site)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    jsonl_path = run_dir / args.articles_file
    out_dir = run_dir / args.out_dir

    if not jsonl_path.exists():
        print(f"ERROR: could not find {jsonl_path}")
        sys.exit(1)

    print(f"Run dir     : {run_dir}")
    print(f"Articles    : {jsonl_path}")
    print(f"Output (site): {out_dir}")

    build_site(str(jsonl_path), str(out_dir))


if __name__ == "__main__":
    main()
