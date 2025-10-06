import os
import time
import requests
import pandas as pd
from itertools import combinations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import igraph as ig
import networkx as nx

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

import plotly.express as px
from pyvis.network import Network

# ------------------ GitHub Token ------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Set GITHUB_TOKEN in your .env file!")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# ------------------ GitHub API Functions ------------------
def github_get(url, params=None):
    while True:
        r = requests.get(url, headers=HEADERS, params=params)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 403:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time()+60))
            wait = max(5, reset - time.time() + 1)
            print(f"Rate limited. Sleeping {int(wait)}s...")
            time.sleep(wait)
        else:
            print(f"Error {r.status_code} for {url}")
            r.raise_for_status()

def search_repos(query, per_page=100, max_pages=2):
    all_repos = []
    for page in range(1, max_pages+1):
        params = {"q": query, "per_page": per_page, "page": page, "sort":"stars","order":"desc"}
        print(f"Fetching page {page}...")
        data = github_get("https://api.github.com/search/repositories", params)
        items = data.get("items", [])
        if not items: break
        all_repos.extend(items)
        time.sleep(1)
    return all_repos

def get_repo_details(repo):
    full_name = repo['full_name']
    try:
        languages = github_get(f"https://api.github.com/repos/{full_name}/languages")
    except:
        languages = {}
    contributors = []
    page = 1
    while True:
        try:
            data = github_get(f"https://api.github.com/repos/{full_name}/contributors", params={"per_page":100,"page":page})
            if not data: break
            contributors.extend([c.get("login","") for c in data if "login" in c])
            if len(data) < 100: break
            page +=1
            time.sleep(0.2)
        except:
            break
    return {
        "full_name": full_name,
        "owner": repo.get("owner", {}).get("login",""),
        "languages": languages,
        "contributors": contributors,
        "stargazers_count": repo.get("stargazers_count",0)
    }

def scrape_github(query="created:>2020-01-01 stars:>100", per_page=100, max_pages=2, max_workers=10, limit=150):
    repos = search_repos(query, per_page, max_pages)
    dataset = []
    filename = "github_dataset.csv"
    print("Fetching repo details with multithreading...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_repo_details, r) for r in repos[:limit]]
        for idx, fut in enumerate(as_completed(futures),1):
            try:
                dataset.append(fut.result())
            except Exception as e:
                print(f"Error: {e}")
            if idx % 10 == 0 or idx == len(futures):
                pd.DataFrame(dataset).to_csv(filename, index=False)
                print(f"Saved {idx}/{len(futures)} repos to {filename}")
    print(f"Scraping complete! Saved {filename}")
    return pd.DataFrame(dataset)

# ------------------ Build User Graph ------------------
def build_user_graph(df, min_edge_weight=2):
    edges_dict = defaultdict(int)
    user_languages = defaultdict(lambda: defaultdict(int))  # user -> language -> bytes
    print("Building user-user graph...")
    for idx, row in df.iterrows():
        contributors = row.get("contributors", [])
        languages = row.get("languages", {})
        if not contributors or len(contributors)<2: continue
        for u1, u2 in combinations(contributors,2):
            key = tuple(sorted([u1,u2]))
            edges_dict[key] += 1
        for u in contributors:
            for lang, b in languages.items():
                user_languages[u][lang] += b
        if (idx+1)%20==0:
            print(f"Processed {idx+1}/{len(df)} repos...")
    edges = [k for k,v in edges_dict.items() if v>=min_edge_weight]
    weights = [v for k,v in edges_dict.items() if v>=min_edge_weight]
    all_users = set([u for edge in edges for u in edge])
    user_to_idx = {u:i for i,u in enumerate(all_users)}
    edges_idx = [(user_to_idx[u1], user_to_idx[u2]) for u1,u2 in edges]
    G = ig.Graph(edges=edges_idx, directed=False)
    G.vs["name"] = list(all_users)
    G.es["weight"] = weights
    top_languages = []
    for u in G.vs["name"]:
        langs = user_languages[u]
        top = max(langs.items(), key=lambda x: x[1])[0] if langs else "Other"
        top_languages.append(top)
    G.vs["top_language"] = top_languages
    print(f"Graph created: {G.vcount()} nodes, {G.ecount()} edges")
    return G

# ------------------ Save GEXF for Gephi ------------------
def save_gexf(G, filename="user_user_graph_leiden.gexf"):
    print("Converting igraph to NetworkX for Gephi export...")
    language_counts = defaultdict(int)
    for v in G.vs:
        language_counts[v["top_language"]] += 1
    top_global_language = max(language_counts.items(), key=lambda x: x[1])[0]
    Gnx = nx.Graph()
    for v in G.vs:
        Gnx.add_node(
            v["name"],
            top_language=v["top_language"],
            is_top_global_language=v["top_language"]==top_global_language
        )
    for e in G.es:
        u, v_ = e.tuple
        Gnx.add_edge(G.vs[u]["name"], G.vs[v_]["name"], weight=e["weight"])
    nx.write_gexf(Gnx, filename)
    print(f"Saved GEXF file: {filename}")
    print(f"Global top language: {top_global_language}")

# ------------------ Plot Language Clusters ------------------
def plot_language_clusters(G, filename="language_clusters.png"):
    languages = sorted(set(G.vs["top_language"]))
    lang_to_nodes = {lang: [i for i,v in enumerate(G.vs["top_language"]) if v == lang] for lang in languages}
    cols = min(6, len(languages))
    rows = (len(languages) + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(cols*5, rows*5))
    palette = plt.colormaps['tab20']
    lang_colors = {lang: mcolors.to_hex(palette(i % 20)) for i,lang in enumerate(languages)}
    radius = 1.6
    positions = dict()
    cluster_bounds = dict()
    language_counts = defaultdict(int)
    for v in G.vs:
        language_counts[v["top_language"]] += 1
    top_global_language = max(language_counts.items(), key=lambda x: x[1])[0]

    for idx, lang in enumerate(languages):
        center_x = radius*2.5 *(idx%cols)
        center_y = -radius*2.5* (idx//cols)
        nodes = lang_to_nodes[lang]
        n = len(nodes)
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        for i, node in enumerate(nodes):
            x = center_x + radius*np.cos(theta[i])
            y = center_y + radius*np.sin(theta[i])
            positions[node] = (x, y)
        cluster_bounds[lang] = (center_x, center_y, radius+0.4)
    for e in G.es:
        src, dst = e.tuple
        x1, y1 = positions.get(src, (0,0))
        x2, y2 = positions.get(dst, (0,0))
        plt.plot([x1, x2], [y1, y2], color="#bbbbbb", zorder=1, linewidth=0.5, alpha=0.5)
    for node, (x,y) in positions.items():
        lang = G.vs[node]["top_language"]
        edgecolor = "red" if lang==top_global_language else "k"
        plt.scatter(x, y, s=50, color=lang_colors.get(lang,"gray"), edgecolors=edgecolor, linewidths=1.2, zorder=2)
    for lang, (center_x, center_y, rad) in cluster_bounds.items():
        circle = plt.Circle((center_x, center_y), rad, edgecolor=lang_colors[lang], facecolor='none', lw=3, zorder=3)
        ax.add_patch(circle)
        plt.text(center_x, center_y+rad+0.2, lang, fontsize=14, ha='center', weight='bold', color=lang_colors[lang])
    plt.axis('equal')
    plt.axis('off')
    patch_list = [mpatches.Patch(color=lang_colors[lang], label=lang) for lang in languages]
    plt.legend(handles=patch_list, title="Language Cluster", loc="upper left", fontsize='large')
    fig.tight_layout()
    plt.savefig(filename, dpi=320)
    print(f"Saved language cluster PNG as {filename}")

# ------------------ Interactive HTML Dashboard ------------------
def generate_html_dashboard(G, df, filename="github_dashboard.html"):
    # 1. Bar chart of top languages
    language_counts = defaultdict(int)
    for v in G.vs:
        language_counts[v["top_language"]] += 1
    lang_df = pd.DataFrame(language_counts.items(), columns=["Language", "Users"]).sort_values("Users", ascending=False)
    bar_fig = px.bar(lang_df, x="Language", y="Users", title="Number of Users per Top Language")

    # 2. Pie chart of contributions per language
    contrib_counts = defaultdict(int)
    for idx, row in df.iterrows():
        for lang, val in row.get("languages", {}).items():
            contrib_counts[lang] += val
    contrib_df = pd.DataFrame(contrib_counts.items(), columns=["Language", "Bytes"]).sort_values("Bytes", ascending=False)
    pie_fig = px.pie(contrib_df, names="Language", values="Bytes", title="Contributions per Language")

    # 3. Interactive network using pyvis
    net = Network(height="700px", width="100%", notebook=False)
    for v in G.vs:
        color = f"#{np.random.randint(0,0xFFFFFF):06x}"
        size = 10 + G.degree(v.index)
        border = "red" if v["top_language"]==lang_df.iloc[0]["Language"] else "black"
        net.add_node(v["name"], label=v["name"], color=color, title=v["top_language"], size=size, borderWidth=2)
    for e in G.es:
        u, v_ = e.tuple
        net.add_edge(G.vs[u]["name"], G.vs[v_]["name"], value=e["weight"])
    net.show_buttons(filter_=['physics'])
    net.save_graph("network_temp.html")

    # Merge plotly charts and pyvis into one HTML
    with open("network_temp.html", "r") as f:
        network_html = f.read()
    html_content = f"""
    <html><head><title>GitHub Dashboard</title></head><body>
    <h1>GitHub User-Language Dashboard</h1>
    <h2>Interactive Network</h2>
    {network_html}
    <h2>Bar Chart of Top Languages</h2>
    {bar_fig.to_html(full_html=False, include_plotlyjs='cdn')}
    <h2>Pie Chart of Contributions</h2>
    {pie_fig.to_html(full_html=False, include_plotlyjs='cdn')}
    </body></html>
    """
    with open(filename, "w") as f:
        f.write(html_content)
    print(f"Saved interactive dashboard as {filename}")

# ------------------ Main Execution ------------------
if __name__=="__main__":
    df = scrape_github(max_pages=2, max_workers=10, limit=150)
    G = build_user_graph(df)
    save_gexf(G, filename="user_user_graph_leiden.gexf")
    plot_language_clusters(G, filename="language_clusters.png")
    generate_html_dashboard(G, df, filename="github_dashboard.html")
