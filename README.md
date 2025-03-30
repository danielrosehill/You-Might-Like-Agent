# You Might Like

An AI-powered recommendation assistant that suggests content based on your personal preferences.

## Overview

This tool analyzes your entertainment preferences (stored in the `I-like.md` file) and generates personalized recommendations across various categories:

- Movies
- Netflix shows
- Documentaries
- YouTube channels
- Podcasts
- Books
- Online courses
- People to follow
- Blogs

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   ```
   cp .env.example .env
   ```
4. Edit the `.env` file with your API keys:
   - OPENROUTER_API_KEY or OPENAI_API_KEY
   - TMDB_API_KEY
   - PERPLEXITY_API_KEY (optional)

5. Update your preferences in `I-like.md`

## Usage

Run the assistant:
```
python recommend.py
```

The assistant will:
1. Ask which content categories you want recommendations for
2. Ask how many recommendations you want (default: 10)
3. Generate markdown files with personalized recommendations in category-specific folders

## How It Works

The assistant uses AI models to analyze your preferences and generate tailored recommendations. It validates movie and TV show suggestions against TMDB to ensure they exist, and can use Perplexity Sonar for up-to-date information on other content types.

Each recommendation is saved as a markdown file with relevant details and links.
