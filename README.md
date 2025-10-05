aeea# GitHub Language Community Analysis

This project analyzes GitHub repositories to identify **developer communities based on programming language usage**. Using social network analytics (SNA) techniques and community detection algorithms, we explore which languages are most popular and how developers cluster around them.

---

## 🔹 Project Overview

- **Goal:** Determine the most used programming languages on GitHub and uncover communities of developers collaborating on similar tech stacks.
- **Approach:**
  1. Scrape GitHub repositories for contributors and languages.
  2. Build bipartite graphs connecting users and languages.
  3. Project bipartite graphs to user-user networks.
  4. Apply community detection algorithms (Louvain, Leiden) to detect developer communities.
  5. Analyze and visualize communities with Gephi and Python plots.

- **End Result:** Communities of developers labeled by their **dominant programming language**, e.g., Python, JavaScript, Java, etc.

---

## 🔹 Dataset

- **Source:** GitHub API
- **Content per repo:**
  - Repository ID, name, owner
  - Contributors
  - Programming languages (bytes per language)
  - Creation date, topics
  - Stars count
- **Stored in:** `github_dataset.json`

> Note: A GitHub Personal Access Token (PAT) is required to handle API rate limits.

---

## 🔹 Analysis Pipeline

1. **Scrape Data**
   - File: `scrape_github.py`
   - Collects top repositories by stars and fetches contributors and languages.

2. **Aggregate Languages**
   - File: `analyze_languages.py`
   - Computes:
     - Total bytes per language
     - Number of repos per language
     - Number of developers per language

3. **Build Graphs**
   - File: `build_graphs.py`
   - Creates bipartite User ↔ Language graph
   - Projects to User–User network (edges = shared languages)

4. **Community Detection**
   - Louvain: `communities_louvain.py`
   - Leiden: `leiden_demo.py`
   - Each developer node is assigned a `community_id` and can be labeled by dominant language.

5. **Visualization**
   - Gephi: visualize `user_user_graph_louvain.gexf` or `user_user_graph_leiden.gexf`
   - Python plots: bar charts of top languages, community sizes, and optionally geographic distribution.

---

## 🔹 Results

- **Top Languages by Developers / Bytes / Repos**:
  - Python, JavaScript, Java, C++, etc.
- **Communities:**
  - Detected developer clusters naturally grouped by dominant language.
  - Example: Community 1 → Python, Community 2 → JavaScript, Community 3 → Java
- **Visualizations:**
  - Gephi network of developers colored by community
  - Bar charts of language popularity
  - Optional map showing top languages by country

---

## 🔹 Requirements

- Python 3.8+
- Libraries:
- requests
- networkx
- python-louvain
- python-igraph
- leidenalg
- plotly
- pandas
- Gephi (for interactive network visualization)

---

## 🔹 How to Run

1. **Set GitHub Token**
 ```bash
 export GITHUB_TOKEN="your_token_here"
 ```
2. **Scrape GitHub Repositories**
```bash
python scrape_github.py
```
3. **Analyze Language Usage**
```bash
python analyze_languages.py
```
4. **Build User Network Graph**
```bash
python build_graphs.py
```
5. **Run Community Detection**
```bash
python communities_louvain.py
python leiden_demo.py
```

