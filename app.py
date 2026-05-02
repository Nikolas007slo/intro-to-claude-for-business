"""
RAG Knowledge Base - Starter Template
FAMNIT AI Course - Day 3

A simple Retrieval-Augmented Generation (RAG) app built with
Streamlit, LangChain, and ChromaDB. No API keys needed!

Instructions:
  1. Replace the DOCUMENTS list below with your own texts
  2. Update the app title and description
  3. Run locally:  streamlit run app.py
  4. Deploy to Render (see assignment instructions)
"""

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Intro to Claude for Business",
    page_icon="🤖",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────
# YOUR DOCUMENTS — Replace these with your own topic!
# Each string is one "document" that will be chunked, embedded, and
# stored in the vector database for semantic search.
# ──────────────────────────────────────────────────────────────────────
DOCUMENTS = [
    """# **Document 1: Getting better results with prompt engineering**

Prompt engineering is your bridge to good communication with generative AI. The quality of Claude’s output depends on the quality of your input — your prompt. The most common phrase to sum this up is “garbage in, garbage out”. A more detailed explanation of what you want from the LLM is gonna get much better results than basic and generic commands. The six techniques below are the core toolkit.

## **1. Be clear, direct, and specific**

A detailed explanation of what you want from the LLM gets much better results than basic commands. Provide instructions as sequential steps when order matters.

*Example: Instead of a vague prompt like ****“Write a summary of this meeting,”**** use a specific, step-by-step approach: ****“1. Summarize the key decisions made in the transcript below. 2. List the action items in bullet points. 3. Output the final text in under 150 words.”***

## **2. Provide role and context**

Giving Claude context about your goals and assigning a specific role focuses its behavior and tone.

*Example: ****“Context: I am launching a new B2B software startup and need an email template for cold outreach. Role: Act as a senior tech sales executive known for a polite, direct, and non-salesy tone. Task: Write a 3-sentence introductory email.”***

## **3. Few-shot or multishot prompting**

Providing examples of the exact output you expect is one of the most reliable ways to control Claude’s accuracy and consistency.

*Example: If you want output in JSON, paste two or three short JSON examples in your prompt before asking the question.*

## **4. Structure prompts with XML tags**

XML tags help Claude parse complex prompts without ambiguity, clearly separating instructions, context, and inputs.

*Example: ****<******instructions******>******Translate the text below into Slovenian, keeping the tone formal.******<******/instructions******>****** ******<******input******>******The quarterly earnings call will commence at 9:00 AM.******<******/input******>***

## **5. Long context prompting**

When working with large documents, put the long-form data at the very top of your prompt, and place your actual query at the very bottom.

*Example: [Paste 50 pages of PDF interview transcripts here] ****<******query******>******Based on the transcripts provided above, what were the three most common complaints customers had about the new UI?******<******/query******>***

## **6. Chain of thought (CoT)**

Chain of thought is a prompt engineering technique that enhances the output of large language models, particularly for complex tasks involving multistep reasoning, by guiding the model through a step-by-step logical process.

*Example: Instead of asking ****“What’s 23 percent of 847?”****, ask ****“Walk me through calculating 23% of 847 step by step.”***
""",
""" # **Document 2: The Claude Desktop App**

The desktop app gives you everything you can do in the browser, plus access to your local computer — files, tools, and folders — through Local MCP (Model Context Protocol). Because it runs natively on your machine, it can integrate with what’s actually on your disk instead of being limited to what you paste into a chat.

## **Quick-summon and built-in environment**

The desktop app lets Mac users summon Claude from any app without switching windows. Desktop extensions support Node.js, Python, and binary MCP servers, and Claude Desktop includes a built-in Node.js environment, so a separate Node.js installation isn’t required.

## **Cowork for independent work**

Another bonus of the desktop app is Cowork, which is built for more complex and independent work. It gives Claude folder access — Claude reads files from a specific directory and saves finished outputs back to that same location. This is what lets Claude operate on real workflows instead of one-off conversations.

*My example: I have cleaned my disorganized downloads folder with Claude Cowork.*""",

"""# **Document 3: Claude Projects**

Projects are workspaces where you give Claude persistent context — uploaded files and custom instructions that stay loaded across every chat inside that project. Instead of re-explaining yourself in every conversation, you set up the project once and reuse it.

## **Key features**

- **Files: **project knowledge uploaded into project files, retrieved with RAG so Claude has more context and background.

- **Instructions: **guide Claude’s responses with custom tonality, expertise level, and guidelines (for example, a particular brand voice).

- **Share: **Projects can be shared with your team on Claude for Work plans.

## **How it works**

Uploaded files become the project’s persistent knowledge base. You can start separate chats inside the same project and the background of instructions and uploaded files stays the same. In instructions you specify your preferred outputs — tone, level of detail, formatting rules — and Claude applies them every time.

*My example: I used Claude for building a lead scoring program for a university course assignment. I uploaded the assignment instructions and dataset. Inside the project I opened a new chat where Claude helped me polish the dataset for deep learning, then provided a step-by-step guide on how to use Claude Code to execute the project. Once Claude Code finished, I pasted screenshots of its work into the chat where Claude found some mistakes, which I fixed in Claude Code. The project is now finished and got a grade of 10/10.*
""",
""" # **Document 4: How to create mini-apps with Artifacts**

Artifacts are ready-to-use Claude outputs that appear in the side panel next to your chat and can be downloaded immediately. Claude triggers an artifact automatically when the content is significant and self-contained — typically over 15 lines, complex enough to stand on its own, something you’ll edit or reuse, or something you’ll need to refer back to later.

## **Common artifact types**

- **Documents: **docx, ppt, xlsx, pdf, md

- **Code snippets**

- **HTML pages: **forms, demos, quick prototypes

- **SVG images: **logos, icons, illustrations

- **Diagrams: **charts, graphs

- **React components: **interactive UI elements, games, visualizations, calculators

## **Creating artifacts**

Describe what you want and Claude decides whether to present it as an artifact. If it doesn’t make one automatically, just ask it to. The artifact appears in the side panel, where you can view different formats, copy content, download files, or view code. You can also share and publish artifacts with your organization or publicly.

## **Tips for efficient use**

Be specific about what you want, describe the end use, iterate incrementally (one change at a time), and request an artifact explicitly when you need one.

*My example: I asked Claude to make a Standard Operating Procedure for my company role. I gave it the list of my daily tasks with step-by-step descriptions, screenshots, and the company’s color palette. I asked for a Word doc and it produced a professional SOP that I now use.*
""",
""" # **Document 5: Claude Skills**

Claude Skills are pre-written instructions Claude follows for specific task types — they teach Claude how to complete a task in a repeatable, high-performance way. Unlike a static document or one-time prompt, skills get activated automatically based on what you ask: request a financial model or a PDF report, and Claude picks the right skill.

The key thing to understand is that skills are scoped by *what* you’re doing, not *where* you’re doing it. A project is a knowledge hub for reference material; a skill is a task machine that defines methodology, steps, and order of operations. So a single custom workflow — a brand voice review, a compliance checklist — applies consistently no matter which project workspace you’re in.

## **Key features**

- **Task-scoped, not location-scoped: **tied to what you’re doing, not where.

- **Reusable everywhere: **one skill works across all chats and projects.

- **Complementary to projects: **the project gives the “what” (information and context), the skill gives the “how” (the process).

- **Built-in or custom: **use Anthropic’s pre-built skills, or build your own.

## **Building custom skills**

You can create your own skills through a simple conversation with Claude. Claude interviews you about your workflow and desired outcomes, then structures the instructions into a file. Once saved, the skill stays a persistent part of Claude’s capabilities, so your specific way of doing things gets applied every time.

*Example: A ****“customer call prep”**** skill can be invoked to process data stored in a specific project’s knowledge base. The project provides the customer history and notes; the skill defines how to summarize and structure the prep document. Or, instead of pasting your formatting rules in every new chat, create a ****“weekly report”**** skill once. Next time you ask Claude to draft a weekly report, it will automatically apply your sections, tonality, and length without you re-explaining anything.*

*My example: When I ask Claude to make me a Word doc, it’s visible from the reasoning trace that it activated the docx skill.*
""",
""" # **Document 6: Claude Models**

Before diving into Claude models, one baseline: all current Claude models support text and image input, text output, multilingual capabilities, and vision. The differences come down to intelligence, speed, cost, and which thinking modes are available.

## **Claude Opus 4.7**

Currently the most recent and most capable model for solving complex tasks. It doesn’t feature Extended thinking, but you can use Adaptive thinking instead. The trade-off is that responses are slower and more expensive.

*My example: I personally use Opus 4.7 for my projects — building a website, deep learning models, and learning.*

## **Claude Sonnet 4.6**

The best combo of speed and intelligence, with Extended thinking available. Cheaper than Opus 4.7. Great for navigating complex codebases, iterative development, deep reasoning, and advanced computer use, with consistently high-quality output.

*My example: When I started using Claude, I tried Sonnet first because it was available on the free plan. Later I built my deep learning lead scoring assignment in Opus.*

## **Claude Haiku 4.5**

The fastest model with near-frontier intelligence. It has Extended thinking but you cannot use Adaptive thinking. The fastest comparative latency of the three. Specialty: lightweight tasks at high volume — real-time responses, automation, short-form outputs.

*My example: I use Haiku 4.5 for quick factual questions and easier tasks like ****“create me a menu for today’s guest event.”***
""",
""" # **Document 7: What is agentic research**

Agentic research is when an AI doesn’t just answer your question, it plans the research, runs the searches, reads the sources, and comes back with a structured report. Instead of one question and one answer, the model breaks your prompt into smaller sub-questions, runs multiple searches that build on each other, and pulls the threads together at the end. You set the goal, the AI handles the legwork.

The shift from regular chat is significant. A normal Claude conversation is single-turn: you ask, it answers from what it knows or one quick search. Agentic research is multi-step and autonomous: the model plans an approach, fetches information from many sources, examines different angles, and iterates when something is missing.

## **Key features**

- **Planning: **the model figures out what to search for and in what order before it starts.

- **Tool use: **it actively uses web search, fetches pages, and pulls data from connected sources.

- **Iteration: **if the first round of results is shallow, it goes deeper or tries a different angle.

- **Citations: **every claim in the final report is linked to its source so you can verify it.

- **Extended thinking: **the model reasons through the problem step by step, and you can expand the thinking section to read its logic.

Because the work is autonomous, agentic research takes longer than a normal chat — anywhere from 5 to 45 minutes depending on complexity. The trade-off is that you walk away and come back to a finished report instead of babysitting tabs. It is best for tasks that would normally cost you a half day of manual searching: market analysis, competitor research, technical documentation synthesis, product roadmap planning.

*Example: Instead of opening 8 tabs to compare pricing across 5 competitor SaaS tools, you give Claude one prompt with the competitors named and the metrics you care about. Claude plans the search, visits each pricing page, pulls the data, and returns a structured comparison with citations to every source — while you go work on something else.*
""",
""" # **Document 8: How to use Claude’s research feature**

Research is an advanced feature in Claude that turns a single prompt into a comprehensive report with in-depth analysis from multiple sources. Instead of you opening tabs and copy-pasting, Claude does the searching, reading, and synthesis, then hands you a structured document with citations.

## **How to enable it**

Find the Research button on the bottom left of Claude’s chat window, click it, and submit your prompt. That’s the whole switch — same chat, just a different mode.

## **How to prompt it well**

A research run can take anywhere from 5 to 45 minutes, so it pays to write a detailed prompt upfront. Outline the sections you want in the report, the source types you prefer (academic papers, news, official docs), and any context that narrows the scope. If your prompt is too vague, Claude will ask follow-up questions before it begins. Most reports finish in 5 to 15 minutes; complex ones go longer.

## **What happens after you submit**

Claude operates independently, running multiple searches that build on each other and examining different angles in parallel. Extended thinking is automatically enabled, so you will see a thinking indicator with a timer and an expandable section above the response where you can read Claude’s reasoning step by step. You can close the tab and come back later — the report keeps building in the background.

## **What you get back**

A structured report organized around the sections you asked for, with easy-to-check citations on every claim so you can verify sources. Best for market analysis, technical documentation synthesis, competitor research, and product roadmap planning — work that would normally cost you half a day.

*My example: When I was planning a team offsite, I asked Claude to research and compare venue options, meal catering, and team activities for a two-day trip. I outlined the three sections I wanted in the report, the budget range, and the city. Claude came back 12 minutes later with a comparison table and citations to every venue website — I just had to pick.*
""",
""" # **Document 9: What is MCP (Model Context Protocol)**

MCP, or Model Context Protocol, is an open standard that lets Claude connect to the systems where your data actually lives — files on your computer, Google Drive, Gmail, GitHub, Slack, databases, and more. It was introduced by Anthropic and is now used across the industry as the universal way to plug AI assistants into real tools and real data.

## **The problem it solves**

Even the smartest model is limited if it can’t see your stuff. Before MCP, every integration between an AI and an external system had to be custom-built, which made connected workflows expensive and slow to scale. MCP replaces that mess with one shared protocol: any tool that speaks MCP can talk to any AI that speaks MCP. Build the connector once, use it everywhere.

## **Two flavors**

- **Local MCP **runs on your own computer through the Claude Desktop app. It gives Claude access to your local files, folders, and apps — for example reading a folder of PDFs or saving outputs back to disk. This is what powers Cowork.

- **Remote MCP **connects Claude to online services like Google Drive, Gmail, GitHub, Asana, or Slack. You enable these through the Connectors menu inside Claude, log in once, and Claude can read and act in those tools from any chat.

## **What it unlocks**

Once an MCP connection is live, Claude can pull context from your actual work without you copy-pasting into the chat. It can summarize a Drive doc, draft a reply to a specific Gmail thread, query a database, or pull open issues from GitHub — all from inside one conversation.

*My example: I connected Claude to my Google Drive through Remote MCP. Now when I ask ****“summarize the Q3 strategy doc and pull the three biggest risks,”**** Claude opens the file directly, reads it, and answers — no upload, no copy-paste.*
""",
""" # **Document 10: Using Claude for cold outbound campaigns**

Cold outbound is one of the highest-leverage use cases for Claude in sales work. It compresses hours of research, writing, and rehearsal into minutes, and the personalization stays sharp because Claude actually reads what you give it. The trick is setting Claude up once with the right context, then reusing it across every prospect.

## **1. Prospect research**

Drop a company name, website, or LinkedIn URL into the chat and ask Claude for a quick brief: what they do, recent news, likely pain points, and which decision-maker to target. With Research mode on, Claude pulls from multiple sources and gives you something usable in 5–10 minutes.

## **2. Set up an ICP project**

Create a Claude Project for your outbound campaign and upload your ICP description, product one-pager, and 2–3 of your best-performing emails as examples. Now every chat inside that project starts with full context — you stop re-explaining your product in every prompt.

## **3. Drafting the cold email**

Apply Doc 1’s prompt engineering basics: assign a role (**“act as a sales rep at a B2B SaaS company”**), specify tone (direct, non-salesy), length (under 90 words), and structure (hook, value, soft CTA). Few-shot it with the winning emails you uploaded.

## **4. Personalization at scale**

Paste a list of 10 prospects with one-line notes each and ask Claude for 10 tailored opening lines plus a suggested angle per prospect — not 10 full emails. This keeps the human voice in the body and saves the AI for the part that actually scales: the personalization.

## **5. Cold call scripts and role-play**

Ask Claude to draft a cold call script with the opener, discovery questions, and objection handling for your ICP. Then flip it: tell Claude to act as a skeptical CFO who has been pitched 5 similar tools this month, and run a live role-play. After each round ask for feedback on what landed and what didn’t. This is the cheapest sales coaching you’ll ever get.

## **6. Follow-up sequencing**

For each prospect ask for 2–3 follow-up variants spaced over two weeks, each with a different angle: value-add, breakup, soft re-engage. Claude keeps the thread coherent so you don’t sound like a robot on touch number four.

*My example: I loaded my product one-pager and ICP into a project, dropped in 10 prospects, and got back 10 personalized opening lines in under 3 minutes. Then I role-played the discovery call with Claude playing the prospect — caught two objections I had no good answer for before I ever got on a real call.*
""",
""" # **Document 11: Using Claude for automating content creation**

Content creation eats time mostly because of two things: starting from a blank page, and reformatting the same idea for five different channels. Claude is built to remove both of those frictions — but only if you set it up properly. Without context and voice samples, you get generic AI slop. With them, you get content that actually sounds like you.

## **1. Set up a content project**

Create one Claude Project per channel — LinkedIn, blog, newsletter — and upload three things: your brand voice guide, 2–3 of your best-performing past posts, and a one-paragraph description of your audience. Now every chat inside that project starts with full context. You stop re-explaining who you are in every prompt.

## **2. Generate ideas in bulk, not one polished post**

Feed Claude a topic, your audience, and a few hooks that worked before. Ask for 10 different angles — not one finished post. Quantity first, then you pick the angle that lands and ask Claude to develop only that one. This is faster and produces sharper output than asking for **“a great post about X.”**

## **3. Draft in your voice with few-shot prompting**

Few-shot prompting is the single biggest lever for voice quality. Include 2–3 of your real writing samples directly in the prompt or project, and tell Claude to match the rhythm, sentence length, and tone. Without samples Claude defaults to a polished but bland register. With them it tracks your style closely.

## **4. Repurpose across formats**

One long-form piece is a content week if you let Claude do the conversion. Take a blog post or webinar transcript and ask for: 5 LinkedIn posts with different angles, one newsletter section with a hook and CTA, 10 short tweets, a YouTube video description. Claude handles the format change. You keep the strategy and final approvals.

## **5. Use Claude as your editor**

Before you publish, paste the draft back and ask for a critical review: weak hooks, clichés, unclear CTAs, sentences that don’t earn their place. Treat Claude as a second pair of eyes, not a rubber stamp. The first draft is rarely the one you should ship.

*My example: I loaded my LinkedIn project with my voice guide and three top-performing posts, then dropped in a 40-minute webinar transcript. Claude returned 5 LinkedIn posts, a newsletter teaser, and a short blog summary — all in my voice, all from the same source material. What used to take me a full afternoon now takes about 30 minutes of editing.*
""",
""" # **Document 12: Claude and Finance**

Finance professionals can use Claude to build models, draft investment documents, and make sense of complex spreadsheets. With financial data connectors like Daloopa and S&P Global, Claude pulls live data instead of you copy-pasting from research platforms, and Claude for Excel lets you keep working in the spreadsheet you already use. Below are the three highest-leverage workflows.

## **1. Build financial models**

Tell Claude about the deal, the entry and exit assumptions, the comparables you want pulled, and the format you need. Claude pulls historical financials from Daloopa, comparables from S&P Global, runs base/upside/downside scenarios with sensitivity analysis, and outputs a multi-sheet Excel model with working formulas, conditional formatting, and PE-standard color coding (blue inputs, black calculations, green cross-sheet references). What used to take an analyst days collapses into one conversation. Full walkthrough: [Build financial models](https://claude.com/resources/use-cases/build-financial-models).

## **2. Draft investment memos**

Once the model is built, Claude can turn the analysis into an IC-ready memo. Give it the company, the metrics that drive the decision, and the deliverable format. Claude pulls the financial data, benchmarks against competitors named in SEC filings, calculates segment growth and FCF conversion, flags customer concentration risk, and produces a Word document structured as executive summary → business overview → financial performance → competitive positioning → valuation → risks. Source citations on every claim so reviewers can verify. Full walkthrough: [Draft investment memos](https://claude.com/resources/use-cases/draft-investment-memos).

## **3. Understand and extend an inherited spreadsheet**

Inheriting a model from someone who left the team is one of finance worst rituals — formulas referencing other formulas, assumptions buried across tabs, undocumented logic. Upload the file and Claude reads the formulas, traces cross-sheet dependencies, explains what each tab does, and adds cell comments to document the complex parts. Then it extends the model forward following the original conventions, so the file stays coherent. For models with cascading assumptions, Opus 4.5 outperforms Sonnet because it catches edge cases like seasonality double-counting and tax MAX functions. Full walkthrough: [Understand and extend an inherited spreadsheet](https://claude.com/resources/use-cases/understand-and-extend-an-inherited-spreadsheet).

*My example: When I needed a quick valuation comparison for a B2B SaaS company, I asked Claude to pull recent healthcare SaaS comparables from S**&**P Global, calculate trading multiples and growth rates, and benchmark our target against the set. Got a clean comp table with citations in under 10 minutes — work that would have eaten an afternoon of tab-switching.* """
]

# ──────────────────────────────────────────────────────────────────────
# Cached heavy resources (loaded once, reused across reruns)
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building vector database...")
def build_vector_store(_documents: tuple):
    """Chunk documents, embed them, and store in ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import chromadb
    from chromadb.utils import embedding_functions

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _documents:
        chunks.extend(splitter.split_text(doc))

    # --- ONNX-based embeddings (lighter than PyTorch) ---
    # Uses all-MiniLM-L6-v2 under the hood via ONNX Runtime
    embedding_fn = embedding_functions.ONNXMiniLM_L6_V2()

    # --- Store in ChromaDB ---
    client = chromadb.Client()
    # Reset collection if it exists (for clean rebuilds)
    try:
        client.delete_collection("knowledge_base")
    except Exception:
        pass

    collection = client.create_collection(
        name="knowledge_base",
        embedding_function=embedding_fn,
    )
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    return collection, chunks


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("🤖 Claude for Business")
page = st.sidebar.radio("Navigate", ["Home", "Ask a question", "Explore Chunks"])

# ──────────────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────────────
if page == "Home":
    st.title("Intro to Claude for Business")
    st.markdown("""
    A practical, searchable knowledge base about working with Claude — covering
    Claude's products, features, and the workflows that actually save time at work.

    ### What's inside
    Twelve short documents on prompt engineering, the Claude desktop app, Projects,
    Artifacts, Skills, the Claude model lineup, agentic research, MCP, and concrete
    use cases for sales, content creation, and finance.

    ### How it works
    1. Each document is split into small **chunks**
    2. Every chunk is converted to an **embedding** — a vector that captures its meaning
    3. Chunks are stored in a **vector database** (ChromaDB)
    4. When you ask a question, your query is embedded and compared to every chunk
    5. The most **semantically similar** chunks come back as your answer

    Translation: you can ask things in your own words. "How do I write a better prompt?"
    finds the prompt engineering doc even though it never uses the word "better."

    ### Get started
    - Go to **Ask a question** to query the knowledge base
    - Go to **Explore Chunks** to see how documents are split and how chunking shapes search

    ---
    *Built with Streamlit, LangChain, ChromaDB, and the all-MiniLM-L6-v2 embedding model.*
    """)

    st.info(f"Knowledge base contains **{len(DOCUMENTS)} documents**.")


# ──────────────────────────────────────────────────────────────────────
# SEARCH PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Ask a question":
    st.title("Ask a question")
    st.markdown("Ask a question and the app will find the most relevant chunks from the knowledge base.")

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    query = st.text_input(
        "Your question",
        placeholder="e.g. How do I write a better prompt?",
    )
    num_results = st.slider("Number of results", 1, 10, 3)

    if query:
        with st.spinner("Searching..."):
            results = vector_store.query(
                query_texts=[query],
                n_results=num_results,
            )

        documents = results["documents"][0]
        distances = results["distances"][0]

        st.subheader(f"Top {len(documents)} results")
        for i, (doc_text, distance) in enumerate(zip(documents, distances), 1):
            # ChromaDB returns distance; lower = more similar
            similarity = max(0, 1 - distance)
            with st.container():
                st.markdown(f"**Result {i}** — relevance: `{similarity:.2f}`")
                st.markdown(f"> {doc_text}")
                st.divider()

    st.markdown("---")
    st.caption("Powered by all-MiniLM-L6-v2 embeddings + ChromaDB")
# ──────────────────────────────────────────────────────────────────────
# EXPLORE CHUNKS PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Explore Chunks":
    st.title("Explore Chunks")
    st.markdown(
        "See how the 12 documents get split into chunks by the recursive text splitter. "
        f"Current settings: **chunk_size=600**, **chunk_overlap=80**."
    )
    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    st.metric("Total chunks", len(chunks))

    lengths = [len(c) for c in chunks]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg chunk size", f"{np.mean(lengths):.0f} chars")
    col2.metric("Min chunk size", f"{min(lengths)} chars")
    col3.metric("Max chunk size", f"{max(lengths)} chars")

    st.subheader("All chunks")
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} ({len(chunk)} chars)"):
            st.text(chunk)
